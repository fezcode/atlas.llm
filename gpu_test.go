package main

import (
	"runtime"
	"strings"
	"testing"
)

// Non-GPU inputs always normalise to the CPU build. GPU names are
// platform-dependent, so they're covered by
// TestResolveEngineVariantFallsBackWhenUnavailable instead.
func TestResolveEngineVariantNormalisesNonGPU(t *testing.T) {
	for _, in := range []string{"", "auto", "AUTO", " cpu ", "cpu", "nonsense"} {
		if got := resolveEngineVariant(in); got != engineVariantCPU {
			t.Errorf("resolveEngineVariant(%q) = %q, want %q", in, got, engineVariantCPU)
		}
	}
}

// Case and surrounding whitespace must not change which build is selected.
func TestResolveEngineVariantIgnoresCaseAndSpace(t *testing.T) {
	for _, v := range engineVariantNames() {
		if v == engineVariantAuto {
			continue
		}
		want := resolveEngineVariant(v)
		for _, variant := range []string{strings.ToUpper(v), " " + v + " "} {
			if got := resolveEngineVariant(variant); got != want {
				t.Errorf("resolveEngineVariant(%q) = %q, want %q", variant, got, want)
			}
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

// The CUDA engine archive and its runtime companion share a filename
// suffix, so matching on the tail alone picks whichever came first in the
// release listing. Prefix disambiguation must keep them apart.
func TestFindReleaseAssetDisambiguatesCUDA(t *testing.T) {
	rel := githubRelease{
		TagName: "b10280",
		Assets: []githubAsset{
			// Deliberately listed runtime-first, the order that breaks a
			// naive HasSuffix match.
			{Name: "cudart-llama-bin-win-cuda-12.4-x64.zip", BrowserDownloadURL: "RUNTIME"},
			{Name: "llama-b10280-bin-win-cuda-12.4-x64.zip", BrowserDownloadURL: "ENGINE"},
		},
	}
	if got := findReleaseAsset(rel, "win-cuda-12.4-x64.zip", enginePrefix); got != "ENGINE" {
		t.Errorf("engine lookup returned %q, want ENGINE", got)
	}
	if got := findReleaseAsset(rel, "cudart-llama-bin-win-cuda-12.4-x64.zip", ""); got != "RUNTIME" {
		t.Errorf("runtime lookup returned %q, want RUNTIME", got)
	}
	if got := findReleaseAsset(rel, "win-vulkan-x64.zip", enginePrefix); got != "" {
		t.Errorf("absent asset returned %q, want empty", got)
	}
}

// Every advertised variant must name a real asset for the platform it's
// listed under, and CUDA must always carry its runtime companion.
func TestEngineAssetTableIsCoherent(t *testing.T) {
	for variant, byPlatform := range llamacppAssetSuffix {
		for platform, asset := range byPlatform {
			if asset.Suffix == "" {
				t.Errorf("%s/%s has an empty suffix", variant, platform)
			}
			if asset.Size == "" {
				t.Errorf("%s/%s has no download size for the UI", variant, platform)
			}
			// CUDA links against DLLs shipped separately; without the
			// companion llama-server won't start.
			if variant == engineVariantCUDA && asset.Companion == "" {
				t.Errorf("%s/%s must declare a cudart companion", variant, platform)
			}
			if asset.Companion != "" && !strings.HasPrefix(asset.Companion, "cudart-") {
				t.Errorf("%s/%s companion %q is not a cudart archive", variant, platform, asset.Companion)
			}
		}
	}
}

// A GPU variant with no build for this platform must fall back rather than
// persist a selection that can never download.
func TestResolveEngineVariantFallsBackWhenUnavailable(t *testing.T) {
	if runtime.GOOS == "darwin" {
		for _, v := range []string{engineVariantVulkan, engineVariantCUDA, engineVariantHIP} {
			if got := resolveEngineVariant(v); got != engineVariantCPU {
				t.Errorf("resolveEngineVariant(%q) on darwin = %q, want cpu", v, got)
			}
		}
	}
	// Whatever this platform advertises must resolve to itself.
	for _, v := range engineVariantNames() {
		if v == engineVariantAuto {
			continue
		}
		if got := resolveEngineVariant(v); got != v {
			t.Errorf("advertised variant %q resolved to %q", v, got)
		}
	}
}

func TestEngineVariantIsGPU(t *testing.T) {
	for _, v := range []string{engineVariantVulkan, engineVariantCUDA, engineVariantHIP} {
		if !engineVariantIsGPU(v) {
			t.Errorf("%q should be a GPU variant", v)
		}
	}
	for _, v := range []string{engineVariantCPU, engineVariantAuto, ""} {
		if engineVariantIsGPU(v) {
			t.Errorf("%q should not be a GPU variant", v)
		}
	}
}

// The help block must render without panicking on any platform and mention
// the setting it documents.
func TestGPUHelpRows(t *testing.T) {
	withTempHome(t)
	rows := gpuHelpRows()
	if len(rows) < 2 {
		t.Fatalf("expected at least 2 help rows, got %d", len(rows))
	}
	var joined string
	for _, r := range rows {
		joined += r[0] + " " + r[1] + "\n"
	}
	if !strings.Contains(joined, "gpu_layers") {
		t.Errorf("help text never mentions gpu_layers:\n%s", joined)
	}
	if runtime.GOOS == "darwin" && !strings.Contains(joined, "Metal") {
		t.Errorf("macOS help should mention Metal:\n%s", joined)
	}
	if runtime.GOOS == "windows" && !strings.Contains(joined, "engine_variant") {
		t.Errorf("Windows help should point at engine_variant:\n%s", joined)
	}
}

// Windows GPU support is the whole point of the cuda/hip/vulkan variants,
// and it can't be exercised by running the suite on macOS — so pin the
// table entries directly.
func TestWindowsHasGPUVariants(t *testing.T) {
	const win = "windows/amd64"
	for _, v := range []string{engineVariantVulkan, engineVariantCUDA, engineVariantHIP} {
		asset, ok := llamacppAssetSuffix[v][win]
		if !ok {
			t.Errorf("no %s build registered for %s", v, win)
			continue
		}
		if !strings.HasPrefix(asset.Suffix, "win-") {
			t.Errorf("%s/%s suffix %q is not a Windows asset", v, win, asset.Suffix)
		}
	}
	// Linux keeps Vulkan; CUDA/HIP aren't published as Linux binaries.
	if _, ok := llamacppAssetSuffix[engineVariantVulkan]["linux/amd64"]; !ok {
		t.Error("no vulkan build registered for linux/amd64")
	}
}

// The /help block must render end to end without panicking.
func TestWelcomeTextIncludesPerformanceBlock(t *testing.T) {
	withTempHome(t)
	out := welcomeText()
	if !strings.Contains(out, "Performance") {
		t.Error("welcome text is missing the Performance block")
	}
	if !strings.Contains(out, "gpu_layers") {
		t.Error("welcome text never mentions gpu_layers")
	}
	t.Logf("\n%s", out)
}
