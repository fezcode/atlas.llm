package main

import (
	"strings"
	"testing"
)

const gib = 1024 * 1024 * 1024

// The clamp decides whether a model loads at all, so its arithmetic is
// checked directly rather than through a GPU and a GGUF on disk.
func TestLayersThatFit(t *testing.T) {
	cases := []struct {
		name       string
		weights    int64
		kv         int64
		blocks     int
		used, tot  int
		wantLayers int
		wantOK     bool
	}{
		{
			// 7.7GB of weights on an idle 16GB card: everything fits.
			name:    "14B on an empty 16GB card",
			weights: 7700 * 1024 * 1024, kv: 2 * gib, blocks: 40,
			used: 300, tot: 16303, wantLayers: 40, wantOK: true,
		},
		{
			// 15.9GB of weights cannot fit in 16GB alongside a KV cache —
			// this is the case that used to fail at load with -ngl 999.
			name:    "30B on a 16GB card",
			weights: 15900 * 1024 * 1024, kv: 2 * gib, blocks: 48,
			used: 300, tot: 16303, wantOK: true,
		},
		{
			// Another process holding most of the card leaves nothing.
			name:    "card already full",
			weights: 7700 * 1024 * 1024, kv: 1 * gib, blocks: 40,
			used: 15500, tot: 16303, wantLayers: 0, wantOK: true,
		},
		{
			name:    "unknown block count is not guessable",
			weights: 7700 * 1024 * 1024, kv: 1 * gib, blocks: 0,
			used: 0, tot: 16303, wantOK: false,
		},
		{
			name:    "unknown VRAM is not guessable",
			weights: 7700 * 1024 * 1024, kv: 1 * gib, blocks: 40,
			used: 0, tot: 0, wantOK: false,
		},
	}
	for _, tc := range cases {
		got, ok := layersThatFit(tc.weights, tc.kv, tc.blocks, tc.used, tc.tot)
		if ok != tc.wantOK {
			t.Errorf("%s: ok = %v, want %v", tc.name, ok, tc.wantOK)
			continue
		}
		if !ok {
			continue
		}
		if tc.wantLayers > 0 && got != tc.wantLayers {
			t.Errorf("%s: %d layers, want %d", tc.name, got, tc.wantLayers)
		}
		if got > tc.blocks {
			t.Errorf("%s: %d layers exceeds the model's %d", tc.name, got, tc.blocks)
		}
		if got < 0 {
			t.Errorf("%s: negative layer count %d", tc.name, got)
		}
	}

	// The oversized case must land strictly between "none" and "all":
	// clamping to 0 would drop to CPU, and to all would fail at load.
	n, ok := layersThatFit(15900*1024*1024, 2*gib, 48, 300, 16303)
	if !ok || n <= 0 || n >= 48 {
		t.Errorf("30B on 16GB gave %d of 48 layers (ok=%v), want a partial offload", n, ok)
	}
	t.Logf("30B on a 16GB card: %d of 48 layers", n)
}

// Headroom must actually be withheld, or the estimate fits a model into
// memory the CUDA context is about to take.
func TestLayersThatFitReservesHeadroom(t *testing.T) {
	// Weights exactly equal to total VRAM must never come out as "all fits".
	n, ok := layersThatFit(16303*1024*1024, 0, 40, 0, 16303)
	if !ok {
		t.Fatal("expected a usable estimate")
	}
	if n >= 40 {
		t.Errorf("offloaded all %d layers with no headroom left", n)
	}
}

func TestParseGPUMemory(t *testing.T) {
	used, total, ok := parseGPUMemory("2303, 16303\n")
	if !ok || used != 2303 || total != 16303 {
		t.Errorf("got %d/%d ok=%v", used, total, ok)
	}
	// Multi-GPU: device 0 is the one llama.cpp uses.
	used, _, ok = parseGPUMemory("100, 8192\n900, 24576\n")
	if !ok || used != 100 {
		t.Errorf("multi-GPU picked the wrong row: used=%d", used)
	}
	for _, in := range []string{"", "N/A, N/A", "junk", "5"} {
		if _, _, ok := parseGPUMemory(in); ok {
			t.Errorf("parseGPUMemory(%q) succeeded, want failure", in)
		}
	}
}

// A client's local ctx_size says nothing about the server's. Computing the
// max_tokens ceiling from it let a client accept a value larger than the
// remote's entire context.
func TestMaxTokensCeilingFollowsTheRemote(t *testing.T) {
	withTempHome(t)
	noGPU(t)
	t.Cleanup(clearRemoteStatus)

	local := Config{CurrentModel: defaultModel, CtxSize: 16384}
	if err := saveConfig(local); err != nil {
		t.Fatal(err)
	}
	localCeiling := maxTokensCeiling(local)

	remote := Config{CurrentModel: defaultModel, CtxSize: 16384, Endpoint: "gpu-box:8080"}
	if err := saveConfig(remote); err != nil {
		t.Fatal(err)
	}
	setRemoteStatus(remoteStatus{
		Endpoint: "http://gpu-box:8080", State: remoteHealthy, HaveInfo: true,
		Info: atlasServerInfo{Service: "atlas.llm", CtxPerSlot: 8192},
	})
	remoteCeiling := maxTokensCeiling(remote)

	if remoteCeiling >= localCeiling {
		t.Errorf("ceiling did not follow the smaller remote context: local %d, remote %d",
			localCeiling, remoteCeiling)
	}
	if remoteCeiling > 8192 {
		t.Errorf("ceiling %d exceeds the server's entire %d-token context",
			remoteCeiling, 8192)
	}
}

// A GPU-less machine must not be told about hardware it doesn't have.
func TestGPUSectionEmptyWithoutGPU(t *testing.T) {
	withTempHome(t)
	noGPU(t)
	if got := renderGPUSection(Config{CurrentModel: defaultModel}); got != "" {
		t.Errorf("rendered a GPU section with no GPU:\n%s", got)
	}
}

// A partial offload must say whether it was chosen or estimated — an
// estimate that halves throughput should never look like a deliberate
// setting.
func TestGPUSectionDistinguishesExplicitFromEstimated(t *testing.T) {
	withTempHome(t)
	withStubbedGPU(t, "NVIDIA GeForce RTX 5070 Ti, 12.0, 16303\n", nil)
	installFakeEngine(t, engineVariantCUDA)

	twenty := 20
	out := renderGPUSection(Config{CurrentModel: defaultModel, GPULayers: &twenty})
	if !strings.Contains(out, "set explicitly") {
		t.Errorf("an explicit layer count is not labelled as one:\n%s", out)
	}
}
