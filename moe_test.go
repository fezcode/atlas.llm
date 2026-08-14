package main

import (
	"math"
	"os/exec"
	"strings"
	"testing"
)

// qwenCoderMeta is the shape of Qwen3-Coder-30B-A3B: 48 layers, 128 experts
// of 768 hidden each, GQA with 4 KV heads.
func qwenCoderMeta() ggufMeta {
	return ggufMeta{
		ContextLength: 262144, BlockCount: 48,
		HeadCount: 32, HeadCountKV: 4, KeyLength: 128, EmbeddingLength: 2048,
		ExpertCount: 128, ExpertUsedCount: 8, ExpertFeedForwardLength: 768,
	}
}

func TestIsMoE(t *testing.T) {
	if !qwenCoderMeta().isMoE() {
		t.Error("a 128-expert model is not being recognised as MoE")
	}
	dense := ggufMeta{BlockCount: 40, HeadCount: 32, HeadCountKV: 8, KeyLength: 128, EmbeddingLength: 5120}
	if dense.isMoE() {
		t.Error("a dense model claims to be MoE")
	}
	// expert_count without the hidden size is not enough to size anything.
	if (ggufMeta{ExpertCount: 128}).isMoE() {
		t.Error("expert_count alone should not be treated as usable MoE metadata")
	}
}

// The point of --n-cpu-moe is that experts are most of an MoE layer. If the
// estimate said otherwise, moving them would free nothing and the plan would
// be pointless.
func TestExpertBytesDominateAnMoELayer(t *testing.T) {
	m := qwenCoderMeta()
	const weights = 12_848_766_112 // UD-IQ3_XXS on disk
	perLayer := int64(weights / m.BlockCount)

	got := m.expertBytesPerLayer(weights)
	if got <= 0 {
		t.Fatal("no estimate produced for a well-described MoE model")
	}
	if got >= perLayer {
		t.Errorf("expert estimate %d is not less than the whole layer %d", got, perLayer)
	}
	// 128 experts against four attention projections: experts are the layer.
	// The safety factor shades it down, so the floor is well below 1.0.
	if share := float64(got) / float64(perLayer); share < 0.75 {
		t.Errorf("experts estimated at %.0f%% of a layer; for 128 experts that is far too low", share*100)
	}

	// More experts, same attention: a larger share must be movable.
	wide := m
	wide.ExpertCount = 256
	if wide.expertBytesPerLayer(weights) <= got {
		t.Error("doubling the expert count did not increase the movable share")
	}

	// A dense model has nothing to move.
	dense := ggufMeta{BlockCount: 40, HeadCount: 32, HeadCountKV: 8, KeyLength: 128, EmbeddingLength: 5120}
	if dense.expertBytesPerLayer(weights) != 0 {
		t.Error("a dense model reported movable expert bytes")
	}
}

// Hand-computed against fixed inputs, deliberately not by re-running the
// formula: budget = 16000 − 1000 used − 1024 headroom − 1024 KV = 12952 MiB,
// deficit = 18000 − 12952 = 5048 MiB, at 300 MiB of experts per layer that
// is 17 layers (16 would leave 13200 MiB, still over).
func TestCPUMoELayers(t *testing.T) {
	const mib = 1024 * 1024
	weights := int64(18000) * mib
	expertPer := int64(300) * mib
	kv := int64(1024) * mib

	n, fits := cpuMoELayers(weights, expertPer, kv, 48, 1000, 16000)
	if !fits {
		t.Fatal("reported as impossible when 17 layers is enough")
	}
	if n != 17 {
		t.Errorf("n = %d, want 17", n)
	}
	// And that it is really minimal-and-sufficient, in MiB.
	budget := 16000 - 1000 - 1024 - 1024
	if remaining := 18000 - n*300; remaining > budget {
		t.Errorf("%d layers still leaves %d MiB against a %d MiB budget", n, remaining, budget)
	}
	if remaining := 18000 - (n-1)*300; remaining <= budget {
		t.Errorf("%d layers would have been enough — n is not minimal", n-1)
	}
}

func TestCPUMoELayersEdges(t *testing.T) {
	const mib = 1024 * 1024

	// Already fits whole: nothing moves.
	if n, fits := cpuMoELayers(int64(6000)*mib, int64(100)*mib, int64(512)*mib, 48, 1000, 16000); !fits || n != 0 {
		t.Errorf("a model that fits gave n=%d fits=%v, want 0/true", n, fits)
	}
	// Hopeless: even every layer's experts in RAM leaves too much behind.
	if _, fits := cpuMoELayers(int64(60000)*mib, int64(100)*mib, int64(512)*mib, 48, 1000, 16000); fits {
		t.Error("a 60GB model claims to fit in 16GB with expert offload")
	}
	// No VRAM left at all after headroom and KV.
	if _, fits := cpuMoELayers(int64(6000)*mib, int64(100)*mib, int64(20000)*mib, 48, 1000, 16000); fits {
		t.Error("claims to fit with a KV cache larger than the card")
	}
	// Missing inputs must not produce a confident answer.
	for _, c := range []struct {
		name                    string
		weights, expert, kv     int64
		blocks, usedMiB, totMiB int
	}{
		{"no expert size", 1 << 30, 0, 0, 48, 0, 16000},
		{"no blocks", 1 << 30, 1 << 20, 0, 0, 0, 16000},
		{"no card", 1 << 30, 1 << 20, 0, 48, 0, 0},
	} {
		if _, fits := cpuMoELayers(c.weights, c.expert, c.kv, c.blocks, c.usedMiB, c.totMiB); fits {
			t.Errorf("%s: reported a confident fit", c.name)
		}
	}
}

// The estimate has to cover the base-flag fallback as well as the optimized
// launch: one f16 slot rather than kvCacheSlots q8_0 ones. Undercounting
// either means a model that loads on one path fails on the other.
//
// There is no exact-value assertion here on purpose — that would just be the
// formula written twice. TestKVEstimateMatchesMeasuredQwen anchors the
// magnitude against a real llama-server measurement.
func TestKVEstimateCoversBothLaunchPaths(t *testing.T) {
	m := ggufMeta{BlockCount: 32, FullAttentionInterval: 4, HeadCountKV: 4, KeyLength: 256}
	const ctx = 16384

	got := m.kvCacheBytes(ctx)
	oneSlotF16 := int64(2) * int64(m.kvLayers()) * ctx * int64(m.HeadCountKV) * int64(m.KeyLength) * 2
	if got < oneSlotF16 {
		t.Errorf("estimate %d is below the f16 fallback's %d — the fallback would not fit",
			got, oneSlotF16)
	}
	// It must also grow with the slot count, or the optimized launch — which
	// allocates kvCacheSlots of them — is being undercounted.
	if kvCacheSlots > 1 && got <= oneSlotF16/2*int64(kvCacheSlots)/2 {
		t.Errorf("estimate %d does not reflect %d slots", got, kvCacheSlots)
	}
}

// The restart loop: ensureServer used to decide whether a running server
// still matched by re-resolving gpu_layers, which reads live VRAM — and once
// a server is up, live VRAM includes the gigabytes it is holding. The
// estimate concluded the model no longer fit, the server was torn down,
// which freed that VRAM, so the replacement loaded at full offload and the
// next message did it again. Every tool round reloaded the model from disk.
func TestServerIdentityIgnoresResolvedLayers(t *testing.T) {
	baseTuning := tuningFingerprint(Config{})
	// A non-nil cmd is what makes a server local rather than remote; no
	// process is started for it.
	s := &llamaServer{
		cmd:   &exec.Cmd{},
		model: Model{Name: "qwen3.5-9b"}, ctxN: 16384, askedCtx: 16384,
		gpuLayer: maxGPULayers, gpuSetting: gpuSettingAuto, tuning: baseTuning,
	}
	if !serverMatches(s, "qwen3.5-9b", 16384, gpuSettingAuto, baseTuning) {
		t.Fatal("a server matching its own settings was rejected")
	}
	// The estimator now says 14 layers, because the model it is measuring is
	// loaded. That must not evict the server that loaded it.
	s.gpuLayer = 14
	if !serverMatches(s, "qwen3.5-9b", 16384, gpuSettingAuto, baseTuning) {
		t.Error("identity depends on the resolved layer count — this is the restart loop")
	}
	// Nor may the per-slot share: a parallel setting above the default
	// shrinks ctxN below what was asked, and comparing the share would be
	// the same loop through a different field.
	s.gpuLayer = maxGPULayers
	s.ctxN = 8192
	if !serverMatches(s, "qwen3.5-9b", 16384, gpuSettingAuto, baseTuning) {
		t.Error("identity depends on the per-slot share instead of the asked context")
	}
	s.ctxN = 16384

	// Things the user actually changed must still force a restart.
	if serverMatches(s, "ministral-3-14b-instruct", 16384, gpuSettingAuto, baseTuning) {
		t.Error("a different model reused the server")
	}
	if serverMatches(s, "qwen3.5-9b", 32768, gpuSettingAuto, baseTuning) {
		t.Error("a context change reused the server")
	}
	if serverMatches(s, "qwen3.5-9b", 16384, 20, baseTuning) {
		t.Error("switching gpu_layers from auto to 20 reused the server")
	}
	if serverMatches(s, "qwen3.5-9b", 16384, gpuSettingAuto, tuningFingerprint(Config{KVOffload: "off"})) {
		t.Error("a tuning change reused the server")
	}
	// A remote server (no subprocess) must never satisfy local identity, or
	// clearing the endpoint would leave the session pointed at the remote.
	remote := &llamaServer{model: Model{Name: "qwen3.5-9b"}, ctxN: 16384, askedCtx: 16384,
		gpuSetting: gpuSettingAuto, tuning: baseTuning}
	if !remote.isRemote() {
		t.Fatal("test setup: a server with no cmd should read as remote")
	}
	if serverMatches(remote, "qwen3.5-9b", 16384, gpuSettingAuto, baseTuning) {
		t.Error("a remote server satisfied local identity")
	}
	if serverMatches(nil, "qwen3.5-9b", 16384, gpuSettingAuto, baseTuning) {
		t.Error("nil matched")
	}
}

func TestGPULayersSetting(t *testing.T) {
	if got := gpuLayersSetting(Config{}); got != gpuSettingAuto {
		t.Errorf("unset gpu_layers = %d, want auto sentinel", got)
	}
	for in, want := range map[int]int{0: 0, 20: 20, -5: 0} {
		v := in
		if got := gpuLayersSetting(Config{GPULayers: &v}); got != want {
			t.Errorf("gpu_layers %d resolved to %d, want %d", in, got, want)
		}
	}
	// The sentinel must not collide with any real setting.
	v := 0
	if gpuLayersSetting(Config{GPULayers: &v}) == gpuSettingAuto {
		t.Error("explicit 0 is indistinguishable from auto")
	}
}

// The two MoE entries are the reason --n-cpu-moe exists; they have to be
// described well enough for the estimator to work on them.
func TestMoECatalogEntries(t *testing.T) {
	for _, name := range []string{"qwen3-coder-30b-a3b", "gemma-4-26b-a4b-it"} {
		m, ok := findModel(name)
		if !ok {
			t.Errorf("%s is not in the registry", name)
			continue
		}
		if !strings.HasSuffix(m.Filename, ".gguf") {
			t.Errorf("%s: filename %q is not a gguf", name, m.Filename)
		}
		if !strings.Contains(m.URL, m.Filename) {
			t.Errorf("%s: URL does not point at %s", name, m.Filename)
		}
		// Sized for a 16GB card: bigger than the 14B it supersedes, small
		// enough to leave room for the KV cache.
		gb := float64(parseModelSize(m.Size)) / 1e9
		if gb < 10 || gb > 14 {
			t.Errorf("%s: %s (%.1f GB) is outside the window these entries exist to fill",
				name, m.Size, gb)
		}
	}
}

// planOffload divides by expertBytesPerLayer; a zero would be a division by
// zero or a nonsense plan, and metadata is not guaranteed.
func TestExpertEstimateGuardsIncompleteMetadata(t *testing.T) {
	for _, m := range []ggufMeta{
		{ExpertCount: 128, ExpertFeedForwardLength: 768}, // no shape at all
		{BlockCount: 48, ExpertCount: 128, ExpertFeedForwardLength: 768},
		{BlockCount: 48, HeadCount: 32, HeadCountKV: 4, KeyLength: 128,
			ExpertCount: 128, ExpertFeedForwardLength: 768}, // no embedding length
	} {
		if got := m.expertBytesPerLayer(1 << 34); got != 0 {
			t.Errorf("%+v produced an estimate of %d from incomplete metadata", m, got)
		}
	}
	// And a sane one does produce something finite.
	if got := qwenCoderMeta().expertBytesPerLayer(1 << 34); got <= 0 || math.IsInf(float64(got), 0) {
		t.Errorf("complete metadata gave %d", got)
	}
}
