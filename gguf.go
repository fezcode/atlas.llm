package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
)

// Minimal GGUF metadata reader — just enough to learn a model's trained
// context length, so `/set ctx_size` can validate against what the model
// actually supports instead of guessing.
//
// Format: a fixed header followed by key/value pairs. Tensor data comes
// after, and we never read that far.
// https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

const ggufMagic = 0x46554747 // "GGUF" little-endian

// GGUF metadata value types.
const (
	ggufUint8 uint32 = iota
	ggufInt8
	ggufUint16
	ggufInt16
	ggufUint32
	ggufInt32
	ggufFloat32
	ggufBool
	ggufString
	ggufArray
	ggufUint64
	ggufInt64
	ggufFloat64
)

// ggufMaxKVPairs bounds how much of a header we're willing to walk, so a
// corrupt or hostile file can't spin us forever.
const ggufMaxKVPairs = 1 << 16

// ggufReader walks a GGUF header. It holds the file rather than a plain
// Reader so uninteresting values can be seeked past instead of decoded: the
// tokenizer vocabulary is a six-figure array of strings, and materializing
// it made reading one header take the better part of a second.
type ggufReader struct {
	r *bufio.Reader
	f *os.File
}

// skip advances n bytes without allocating.
func (g *ggufReader) skip(n int64) error {
	if n < 0 {
		return fmt.Errorf("negative skip")
	}
	// Discard is cheap for small hops and avoids invalidating the buffer.
	if n <= 4096 {
		_, err := g.r.Discard(int(n))
		return err
	}
	if _, err := g.f.Seek(-int64(g.r.Buffered())+n, io.SeekCurrent); err != nil {
		return err
	}
	g.r.Reset(g.f)
	return nil
}

// skipValue advances past a value without decoding it.
func (g *ggufReader) skipValue(t uint32) error {
	switch t {
	case ggufUint8, ggufInt8, ggufBool:
		return g.skip(1)
	case ggufUint16, ggufInt16:
		return g.skip(2)
	case ggufUint32, ggufInt32, ggufFloat32:
		return g.skip(4)
	case ggufUint64, ggufInt64, ggufFloat64:
		return g.skip(8)
	case ggufString:
		n, err := g.u64()
		if err != nil {
			return err
		}
		return g.skip(int64(n))
	case ggufArray:
		et, err := g.u32()
		if err != nil {
			return err
		}
		n, err := g.u64()
		if err != nil {
			return err
		}
		// Fixed-width elements can be skipped in a single hop.
		if w := ggufFixedWidth(et); w > 0 {
			return g.skip(int64(n) * w)
		}
		for i := uint64(0); i < n; i++ {
			if err := g.skipValue(et); err != nil {
				return err
			}
		}
		return nil
	default:
		return fmt.Errorf("unknown GGUF value type %d", t)
	}
}

// ggufFixedWidth returns the byte width of a fixed-size value type, or 0
// for variable-length ones.
func ggufFixedWidth(t uint32) int64 {
	switch t {
	case ggufUint8, ggufInt8, ggufBool:
		return 1
	case ggufUint16, ggufInt16:
		return 2
	case ggufUint32, ggufInt32, ggufFloat32:
		return 4
	case ggufUint64, ggufInt64, ggufFloat64:
		return 8
	}
	return 0
}

func (g *ggufReader) u32() (uint32, error) {
	var v uint32
	err := binary.Read(g.r, binary.LittleEndian, &v)
	return v, err
}

func (g *ggufReader) u64() (uint64, error) {
	var v uint64
	err := binary.Read(g.r, binary.LittleEndian, &v)
	return v, err
}

func (g *ggufReader) str() (string, error) {
	n, err := g.u64()
	if err != nil {
		return "", err
	}
	// Keys and string values are short; anything huge means we've lost sync.
	if n > 1<<20 {
		return "", fmt.Errorf("implausible string length %d", n)
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(g.r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

// skipValue advances past a value of the given type, returning it as an
// int64 when it is an integer (which is all we need).
func (g *ggufReader) value(t uint32) (int64, error) {
	switch t {
	case ggufUint8, ggufInt8, ggufBool:
		var v uint8
		err := binary.Read(g.r, binary.LittleEndian, &v)
		return int64(v), err
	case ggufUint16, ggufInt16:
		var v uint16
		err := binary.Read(g.r, binary.LittleEndian, &v)
		return int64(v), err
	case ggufUint32, ggufInt32, ggufFloat32:
		v, err := g.u32()
		return int64(v), err
	case ggufUint64, ggufInt64, ggufFloat64:
		v, err := g.u64()
		return int64(v), err
	case ggufString:
		_, err := g.str()
		return 0, err
	case ggufArray:
		et, err := g.u32()
		if err != nil {
			return 0, err
		}
		n, err := g.u64()
		if err != nil {
			return 0, err
		}
		for i := uint64(0); i < n; i++ {
			if _, err := g.value(et); err != nil {
				return 0, err
			}
		}
		return 0, nil
	default:
		return 0, fmt.Errorf("unknown GGUF value type %d", t)
	}
}

// ggufMeta is the subset of GGUF metadata atlas.llm uses: the trained
// context length, plus the shape parameters needed to size the KV cache.
type ggufMeta struct {
	ContextLength   int
	BlockCount      int // transformer layers
	HeadCountKV     int // key/value heads (< head count on GQA models)
	HeadCount       int
	EmbeddingLength int
	KeyLength       int // head dim, when stated explicitly

	// FullAttentionInterval marks a hybrid model where only every Nth layer
	// is full attention and the rest are state-space (SSM) layers holding a
	// fixed-size recurrent state. Qwen3.5 reports 4: 8 of its 32 layers keep
	// a KV cache, not all 32.
	FullAttentionInterval int
	// SlidingWindow caps how many tokens most layers attend over, so their
	// KV stops growing past it. Gemma 3 reports 512.
	SlidingWindow int
}

// kvLayers returns how many layers actually hold a full-context KV cache.
//
// Assuming every layer does overestimates badly on hybrid models: measured
// against Qwen3.5-4B, treating all 32 layers as attention predicted about 4x
// the memory that llama-server actually used, which matches its
// full_attention_interval of 4.
func (m ggufMeta) kvLayers() int {
	if m.BlockCount <= 0 {
		return 0
	}
	if m.FullAttentionInterval > 1 {
		n := m.BlockCount / m.FullAttentionInterval
		if n < 1 {
			n = 1
		}
		return n
	}
	return m.BlockCount
}

// kvCacheBytes estimates the KV cache at a given context size.
//
// Two tensors (K and V) per attending layer, each holding tokens *
// n_kv_heads * head_dim elements at 2 bytes for the default f16 cache.
//
// This is an estimate. It ignores the fixed state of SSM layers and runtime
// overhead, and a model whose metadata omits these hints is treated as plain
// attention. Both choices err high, which is the right direction for a
// figure someone uses to decide whether a model will fit.
//
// Checked against llama-server holding Qwen3.5-4B: predicted 0.54 GB at ctx
// 16384 and 2.15 GB at 65536, a 1.61 GB delta against 1.45 GB measured.
func (m ggufMeta) kvCacheBytes(ctx int) int64 {
	hd := m.headDim()
	layers := m.kvLayers()
	if layers <= 0 || m.HeadCountKV <= 0 || hd <= 0 || ctx <= 0 {
		return 0
	}
	// Sliding-window models (Gemma 3) cap most layers at the window but keep
	// a few global layers at full context, and the metadata does not say how
	// many. Applying the window to every layer would understate the cache
	// several times over, so it is deliberately not applied at all: for a
	// "will this fit" figure, erring high is the safe direction.
	tokens := ctx
	const bytesPerElem = 2 // f16
	return 2 * int64(layers) * int64(tokens) * int64(m.HeadCountKV) * int64(hd) * bytesPerElem
}

// complete reports whether enough shape metadata was found to size the KV
// cache at all.
func (m ggufMeta) complete() bool {
	return m.kvLayers() > 0 && m.HeadCountKV > 0 && m.headDim() > 0
}

// headDim returns the per-head dimension, preferring the explicit key
// length and falling back to embedding / head count.
func (m ggufMeta) headDim() int {
	if m.KeyLength > 0 {
		return m.KeyLength
	}
	if m.HeadCount > 0 && m.EmbeddingLength > 0 {
		return m.EmbeddingLength / m.HeadCount
	}
	return 0
}

// readGGUFMeta walks a GGUF header and extracts the fields above. Keys are
// architecture-prefixed (qwen35.block_count, gemma3.block_count, …), so we
// match on the suffix.
func readGGUFMeta(path string) (ggufMeta, error) {
	var meta ggufMeta
	f, err := os.Open(path)
	if err != nil {
		return meta, err
	}
	defer f.Close()

	g := &ggufReader{r: bufio.NewReaderSize(f, 1<<16), f: f}
	magic, err := g.u32()
	if err != nil {
		return meta, err
	}
	if magic != ggufMagic {
		return meta, fmt.Errorf("%s is not a GGUF file", path)
	}
	if _, err := g.u32(); err != nil { // version
		return meta, err
	}
	if _, err := g.u64(); err != nil { // tensor count
		return meta, err
	}
	nkv, err := g.u64()
	if err != nil {
		return meta, err
	}
	if nkv > ggufMaxKVPairs {
		return meta, fmt.Errorf("implausible GGUF metadata count %d", nkv)
	}

	for i := uint64(0); i < nkv; i++ {
		key, err := g.str()
		if err != nil {
			return meta, err
		}
		t, err := g.u32()
		if err != nil {
			return meta, err
		}
		// Decode only what we use. Everything else — above all the
		// tokenizer arrays — is skipped without allocating.
		if !ggufWantedKey(key) {
			if err := g.skipValue(t); err != nil {
				return meta, err
			}
			continue
		}
		v, err := g.value(t)
		if err != nil {
			return meta, err
		}
		switch {
		case strings.HasSuffix(key, ".context_length"):
			meta.ContextLength = int(v)
		case strings.HasSuffix(key, ".block_count"):
			meta.BlockCount = int(v)
		case strings.HasSuffix(key, ".attention.head_count_kv"):
			meta.HeadCountKV = int(v)
		case strings.HasSuffix(key, ".attention.head_count"):
			meta.HeadCount = int(v)
		case strings.HasSuffix(key, ".embedding_length"):
			meta.EmbeddingLength = int(v)
		case strings.HasSuffix(key, ".attention.key_length"):
			meta.KeyLength = int(v)
		case strings.HasSuffix(key, ".full_attention_interval"):
			meta.FullAttentionInterval = int(v)
		case strings.HasSuffix(key, ".attention.sliding_window"):
			meta.SlidingWindow = int(v)
		}
	}
	if meta.ContextLength <= 0 {
		return meta, fmt.Errorf("no context_length in %s", path)
	}
	return meta, nil
}

// ggufWantedKey reports whether a metadata key is one readGGUFMeta uses.
func ggufWantedKey(key string) bool {
	for _, suffix := range []string{
		".context_length", ".block_count", ".attention.head_count_kv",
		".attention.head_count", ".embedding_length", ".attention.key_length",
		".full_attention_interval", ".attention.sliding_window",
	} {
		if strings.HasSuffix(key, suffix) {
			return true
		}
	}
	return false
}

// GGUF headers never change for a given file, so parse each one once.
// Keyed by path plus size and mtime, so a re-download is picked up.
var (
	ggufCacheMu sync.Mutex
	ggufCache   = map[string]ggufMeta{}
)

// readGGUFMetaCached is what callers on a render path should use: repainting
// the model picker asks for every model's metadata on every keypress.
func readGGUFMetaCached(path string) (ggufMeta, error) {
	info, err := os.Stat(path)
	if err != nil {
		return ggufMeta{}, err
	}
	key := fmt.Sprintf("%s|%d|%d", path, info.Size(), info.ModTime().UnixNano())

	ggufCacheMu.Lock()
	meta, ok := ggufCache[key]
	ggufCacheMu.Unlock()
	if ok {
		return meta, nil
	}

	meta, err = readGGUFMeta(path)
	if err != nil {
		return meta, err
	}
	ggufCacheMu.Lock()
	ggufCache[key] = meta
	ggufCacheMu.Unlock()
	return meta, nil
}

// currentModelMeta returns the active model's GGUF metadata, or ok=false
// when it isn't downloaded or can't be read.
func currentModelMeta() (ggufMeta, bool) {
	m, err := currentModel()
	if err != nil {
		return ggufMeta{}, false
	}
	p, err := modelPath(m)
	if err != nil {
		return ggufMeta{}, false
	}
	meta, err := readGGUFMetaCached(p)
	if err != nil {
		return ggufMeta{}, false
	}
	return meta, true
}

// modelTrainedContext returns the context length a GGUF model was trained
// for — the hard ceiling on what llama-server can be asked to allocate.
// The key is architecture-prefixed (qwen35.context_length,
// gemma3.context_length, …), so we match on the suffix.
func modelTrainedContext(path string) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	g := &ggufReader{r: bufio.NewReaderSize(f, 1<<16), f: f}
	magic, err := g.u32()
	if err != nil {
		return 0, err
	}
	if magic != ggufMagic {
		return 0, fmt.Errorf("%s is not a GGUF file", path)
	}
	if _, err := g.u32(); err != nil { // version
		return 0, err
	}
	if _, err := g.u64(); err != nil { // tensor count
		return 0, err
	}
	nkv, err := g.u64()
	if err != nil {
		return 0, err
	}
	if nkv > ggufMaxKVPairs {
		return 0, fmt.Errorf("implausible GGUF metadata count %d", nkv)
	}

	for i := uint64(0); i < nkv; i++ {
		key, err := g.str()
		if err != nil {
			return 0, err
		}
		t, err := g.u32()
		if err != nil {
			return 0, err
		}
		v, err := g.value(t)
		if err != nil {
			return 0, err
		}
		if strings.HasSuffix(key, ".context_length") && v > 0 {
			return int(v), nil
		}
	}
	return 0, fmt.Errorf("no context_length in %s", path)
}

// currentModelTrainedContext returns the active model's trained context
// length, or 0 when it can't be determined (model not downloaded yet, or an
// unreadable header). Callers treat 0 as "unknown" rather than an error,
// since this only ever refines a limit we already have a default for.
func currentModelTrainedContext() int {
	m, err := currentModel()
	if err != nil {
		return 0
	}
	p, err := modelPath(m)
	if err != nil {
		return 0
	}
	meta, err := readGGUFMetaCached(p)
	if err != nil {
		return 0
	}
	return meta.ContextLength
}
