package main

import (
	"runtime"
	"testing"
)

func TestResolveEngineVariant(t *testing.T) {
	tests := []struct{ in, want string }{
		{"", engineVariantCPU},
		{"auto", engineVariantCPU},
		{"cpu", engineVariantCPU},
		{"vulkan", engineVariantVulkan},
		{"VULKAN", engineVariantVulkan},
		{" vulkan ", engineVariantVulkan},
		// An unrecognised value falls back to the safe build rather than
		// failing to start.
		{"nonsense", engineVariantCPU},
	}
	for _, tt := range tests {
		if got := resolveEngineVariant(tt.in); got != tt.want {
			t.Errorf("resolveEngineVariant(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

// Every variant/platform combination we advertise must map to a real asset
// suffix, and unsupported combinations must fail loudly rather than silently
// downloading the wrong archive.
func TestEngineAssetSuffix(t *testing.T) {
	if _, err := engineAssetSuffix(engineVariantCPU); err != nil {
		t.Errorf("cpu variant unavailable on %s/%s: %v", runtime.GOOS, runtime.GOARCH, err)
	}
	if _, err := engineAssetSuffix("banana"); err == nil {
		t.Error("expected unknown variant to error")
	}
	// llama.cpp publishes no Vulkan build for macOS — Metal covers it.
	if runtime.GOOS == "darwin" {
		if _, err := engineAssetSuffix(engineVariantVulkan); err == nil {
			t.Error("expected no vulkan build for darwin")
		}
	}
}

func TestResolveGPULayers(t *testing.T) {
	zero, forty := 0, 40

	// Explicit values win over auto-detection.
	if got := resolveGPULayers(Config{GPULayers: &zero}); got != 0 {
		t.Errorf("explicit 0 = %d, want 0 (CPU-only must be expressible)", got)
	}
	if got := resolveGPULayers(Config{GPULayers: &forty}); got != 40 {
		t.Errorf("explicit 40 = %d, want 40", got)
	}

	// Auto: macOS ships Metal in the standard archive, so offload by default.
	auto := resolveGPULayers(Config{})
	if runtime.GOOS == "darwin" {
		if auto != maxGPULayers {
			t.Errorf("auto on darwin = %d, want %d (Metal is built in)", auto, maxGPULayers)
		}
	} else if auto != 0 {
		t.Errorf("auto on %s = %d, want 0 (default engine is CPU-only)", runtime.GOOS, auto)
	}

	// A Vulkan engine means offload by default on any platform.
	if got := resolveGPULayers(Config{EngineVariant: engineVariantVulkan}); got != maxGPULayers {
		t.Errorf("auto with vulkan = %d, want %d", got, maxGPULayers)
	}
}

// A negative gpu_layers would make llama-server reject the argument.
func TestResolveGPULayersClampsNegative(t *testing.T) {
	neg := -5
	if got := resolveGPULayers(Config{GPULayers: &neg}); got != 0 {
		t.Errorf("negative gpu_layers = %d, want 0", got)
	}
}

// gpu_layers must survive a config round-trip, including an explicit 0 —
// the case a plain int field would have silently turned back into "auto".
func TestGPULayersConfigRoundTrip(t *testing.T) {
	withTempHome(t)
	zero := 0
	if err := saveConfig(Config{CurrentModel: defaultModel, GPULayers: &zero, EngineVariant: "vulkan"}); err != nil {
		t.Fatal(err)
	}
	got, err := loadConfig()
	if err != nil {
		t.Fatal(err)
	}
	if got.GPULayers == nil {
		t.Fatal("explicit gpu_layers=0 was lost on round-trip")
	}
	if *got.GPULayers != 0 {
		t.Errorf("gpu_layers = %d, want 0", *got.GPULayers)
	}
	if got.EngineVariant != "vulkan" {
		t.Errorf("engine_variant = %q, want vulkan", got.EngineVariant)
	}
	if resolveGPULayers(got) != 0 {
		t.Error("explicit 0 should resolve to CPU-only even with a vulkan engine")
	}

	// Absent field must mean auto, not zero.
	if err := saveConfig(Config{CurrentModel: defaultModel}); err != nil {
		t.Fatal(err)
	}
	got, _ = loadConfig()
	if got.GPULayers != nil {
		t.Error("unset gpu_layers should stay nil (auto)")
	}
}
