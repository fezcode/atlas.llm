package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
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

type ggufReader struct {
	r io.Reader
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

	g := &ggufReader{r: f}
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
	n, err := modelTrainedContext(p)
	if err != nil {
		return 0
	}
	return n
}
