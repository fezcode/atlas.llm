package main

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// installFakeEngine makes isEngineDownloaded() true by planting the binary
// findEngineExecutable looks for, then records the variant marker.
func installFakeEngine(t *testing.T, variant string) {
	t.Helper()
	dir, err := engineDir()
	if err != nil {
		t.Fatal(err)
	}
	name := "llama-cli"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	for _, n := range []string{name, strings.Replace(name, "llama-cli", "llama-server", 1)} {
		if err := os.WriteFile(filepath.Join(dir, n), []byte("stub"), 0755); err != nil {
			t.Fatal(err)
		}
	}
	if err := writeEngineVariant(variant); err != nil {
		t.Fatal(err)
	}
}

// withStubbedGPU replaces the nvidia-smi probe and clears the process-wide
// detection cache, restoring both afterwards so one test's fake hardware
// can't leak into the next.
func withStubbedGPU(t *testing.T, out string, err error) {
	t.Helper()
	prev := nvidiaSmiOutput
	nvidiaSmiOutput = func() (string, error) { return out, err }
	resetGPUDetection()
	t.Cleanup(func() {
		nvidiaSmiOutput = prev
		resetGPUDetection()
	})
}

// noGPU stubs the probe as "nvidia-smi is not installed".
func noGPU(t *testing.T) { withStubbedGPU(t, "", errors.New("executable not found")) }

// Explicit non-GPU inputs always normalise to the CPU build. "auto" is
// deliberately excluded — it depends on detected hardware and is covered by
// TestAutoEngineVariant.
func TestResolveEngineVariantNormalisesNonGPU(t *testing.T) {
	noGPU(t)
	for _, in := range []string{"", "auto", "AUTO", " cpu ", "cpu", "nonsense"} {
		if got := resolveEngineVariant(in); got != engineVariantCPU {
			t.Errorf("resolveEngineVariant(%q) = %q, want %q", in, got, engineVariantCPU)
		}
	}
}

// auto only selects a GPU build when the hardware is positively identified.
// Guessing is worse than CPU: a CUDA build without a working driver fails at
// model load, after a ~510MB download.
func TestAutoEngineVariant(t *testing.T) {
	noGPU(t)
	if got := resolveEngineVariant(engineVariantAuto); got != engineVariantCPU {
		t.Errorf("auto with no GPU = %q, want cpu", got)
	}

	withStubbedGPU(t, "NVIDIA GeForce RTX 5070 Ti, 12.0, 16303\n", nil)
	want := engineVariantCPU
	if _, ok := cudaArchives[platformKey()]; ok && !isDarwin() {
		want = engineVariantCUDA
	}
	if got := resolveEngineVariant(engineVariantAuto); got != want {
		t.Errorf("auto with an NVIDIA GPU on %s = %q, want %q", platformKey(), got, want)
	}
}

// A GPU too old for every published CUDA archive must not resolve to CUDA —
// the download would succeed and then fail to load.
func TestAutoEngineVariantRejectsUncoveredGPU(t *testing.T) {
	withStubbedGPU(t, "NVIDIA GeForce GTX 480, 2.0, 1536\n", nil)
	if got := resolveEngineVariant(engineVariantAuto); got != engineVariantCPU {
		t.Errorf("auto with a compute-2.0 GPU = %q, want cpu", got)
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

// Hermetic on purpose: auto now consults the *installed* engine, so without
// a temp home this asserted against whatever the developer's machine
// happened to have downloaded — and started failing the moment a real CUDA
// engine was installed.
func TestResolveGPULayers(t *testing.T) {
	withTempHome(t)
	noGPU(t)
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
		t.Errorf("auto on %s = %d, want 0 (no GPU engine installed)", runtime.GOOS, auto)
	}

	// An explicitly selected Vulkan engine means offload by default,
	// wherever llama.cpp publishes one.
	want := maxGPULayers
	if !engineVariantAvailable(engineVariantVulkan, platformKey()) && !isDarwin() {
		want = 0
	}
	if isDarwin() {
		want = maxGPULayers // Metal, regardless of the variant asked for
	}
	if got := resolveGPULayers(Config{EngineVariant: engineVariantVulkan}); got != want {
		t.Errorf("explicit vulkan = %d, want %d", got, want)
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
	noGPU(t) // CUDA must be offered even when this machine has no NVIDIA card.
	const win = "windows/amd64"
	for _, v := range []string{engineVariantVulkan, engineVariantCUDA, engineVariantHIP} {
		asset, ok := engineAssetFor(v, win)
		if !ok {
			t.Errorf("no %s build registered for %s", v, win)
			continue
		}
		if !strings.HasPrefix(asset.Suffix, "win-") {
			t.Errorf("%s/%s suffix %q is not a Windows asset", v, win, asset.Suffix)
		}
	}
	// Linux keeps Vulkan and ROCm; CUDA isn't published as a Linux binary.
	for _, v := range []string{engineVariantVulkan, engineVariantHIP} {
		if _, ok := engineAssetFor(v, "linux/amd64"); !ok {
			t.Errorf("no %s build registered for linux/amd64", v)
		}
	}
	if _, ok := engineAssetFor(engineVariantCUDA, "linux/amd64"); ok {
		t.Error("linux/amd64 should have no CUDA archive — llama.cpp publishes none")
	}
}

// engineVariantNames() is evaluated while package-level vars initialise, so
// probing hardware from it makes every invocation — including `--version`
// and `-c` in a shell pipeline — spawn nvidia-smi and log a line.
func TestVariantListingDoesNotProbeHardware(t *testing.T) {
	calls := 0
	prev := nvidiaSmiOutput
	nvidiaSmiOutput = func() (string, error) {
		calls++
		return "NVIDIA GeForce RTX 5070 Ti, 12.0, 16303\n", nil
	}
	resetGPUDetection()
	t.Cleanup(func() {
		nvidiaSmiOutput = prev
		resetGPUDetection()
	})

	_ = engineVariantNames()
	_ = resolveEngineVariant(engineVariantCPU)
	_ = resolveEngineVariant(engineVariantVulkan)
	if calls != 0 {
		t.Errorf("listing variants probed the GPU %d time(s) — startup must stay cheap", calls)
	}

	// Resolving auto is where detection genuinely belongs.
	_ = resolveEngineVariant(engineVariantAuto)
	if calls == 0 {
		t.Error("auto resolution never probed the GPU")
	}
}

// CUDA must be offered wherever llama.cpp publishes it, so a user can select
// it before installing a card — and so the error for a missing driver names
// the real problem instead of silently reverting to CPU.
func TestCUDAIsOfferedRegardlessOfInstalledHardware(t *testing.T) {
	noGPU(t)
	if !engineVariantAvailable(engineVariantCUDA, "windows/amd64") {
		t.Error("cuda should be listed for windows/amd64 with no GPU present")
	}
	if engineVariantAvailable(engineVariantCUDA, "darwin/arm64") {
		t.Error("cuda must not be listed for macOS")
	}
	if runtime.GOOS == "windows" && runtime.GOARCH == "amd64" {
		if got := resolveEngineVariant(engineVariantCUDA); got != engineVariantCUDA {
			t.Errorf("explicit cuda with no GPU = %q, want cuda", got)
		}
	}
}

// Blackwell is the case that motivated windowed selection: CUDA 12.4 has no
// sm_120 kernels, so the pinned archive installed ~510MB that couldn't run.
func TestSelectCUDAArchiveByCapability(t *testing.T) {
	const win = "windows/amd64"
	cases := []struct {
		name string
		cap  int
		want string
		ok   bool
	}{
		{"Blackwell RTX 5070 Ti", 120, "win-cuda-13.3-x64.zip", true},
		{"Ada RTX 4090", 89, "win-cuda-13.3-x64.zip", true},
		{"Turing RTX 2080, the CUDA 13 floor", 75, "win-cuda-13.3-x64.zip", true},
		{"Pascal GTX 1080, dropped by CUDA 13", 61, "win-cuda-12.4-x64.zip", true},
		{"Maxwell GTX 970, the 12.4 floor", 50, "win-cuda-12.4-x64.zip", true},
		{"Fermi, below every archive", 20, "", false},
	}
	for _, tc := range cases {
		asset, ok := selectCUDAArchive(win, tc.cap)
		if ok != tc.ok {
			t.Errorf("%s (cap %d): ok = %v, want %v", tc.name, tc.cap, ok, tc.ok)
			continue
		}
		if ok && asset.Suffix != tc.want {
			t.Errorf("%s (cap %d): got %q, want %q", tc.name, tc.cap, asset.Suffix, tc.want)
		}
	}
	if _, ok := selectCUDAArchive("linux/amd64", 120); ok {
		t.Error("linux has no CUDA archives, but selection returned one")
	}
}

// Every CUDA archive must carry its runtime companion, and the windows must
// leave no gap between the oldest supported card and the newest.
func TestCUDAArchivesAreCoherent(t *testing.T) {
	for platform, list := range cudaArchives {
		if len(list) == 0 {
			t.Errorf("%s has an empty CUDA archive list", platform)
			continue
		}
		for _, a := range list {
			if a.MinCap > a.MaxCap {
				t.Errorf("%s: archive %q has an inverted window %d..%d",
					platform, a.Asset.Suffix, a.MinCap, a.MaxCap)
			}
			if a.Asset.Suffix == "" || a.Asset.Size == "" {
				t.Errorf("%s: archive %+v is missing a suffix or size", platform, a.Asset)
			}
			if !strings.HasPrefix(a.Asset.Companion, "cudart-") {
				t.Errorf("%s: archive %q companion %q is not a cudart archive",
					platform, a.Asset.Suffix, a.Asset.Companion)
			}
		}
		// Walking down from the newest, each archive must reach at least as
		// low as the previous one's floor, or cards in between get nothing.
		for i := 1; i < len(list); i++ {
			if list[i].MaxCap < list[i-1].MinCap-1 {
				t.Errorf("%s: gap between %q (floor %d) and %q (ceiling %d)",
					platform, list[i-1].Asset.Suffix, list[i-1].MinCap,
					list[i].Asset.Suffix, list[i].MaxCap)
			}
		}
	}
}

// With no detected GPU we must fall back to the archive supporting the
// oldest hardware: too-new fails at load, too-old only costs performance.
func TestWidestCUDAArchivePrefersOldestHardware(t *testing.T) {
	asset, ok := widestCUDAArchive("windows/amd64")
	if !ok {
		t.Fatal("windows/amd64 should have CUDA archives")
	}
	if asset.Suffix != "win-cuda-12.4-x64.zip" {
		t.Errorf("widest archive = %q, want the 12.4 build", asset.Suffix)
	}
	if _, ok := widestCUDAArchive("linux/amd64"); ok {
		t.Error("linux/amd64 has no CUDA archives")
	}
}

func TestParseComputeCap(t *testing.T) {
	good := map[string]int{"12.0": 120, "8.9": 89, "6.1": 61, " 7.5 ": 75, "5.0": 50}
	for in, want := range good {
		got, ok := parseComputeCap(in)
		if !ok || got != want {
			t.Errorf("parseComputeCap(%q) = %d, %v; want %d, true", in, got, ok, want)
		}
	}
	// A malformed reading must fail rather than resolve to a plausible
	// capability — a wrong archive only surfaces after a ~510MB download.
	for _, in := range []string{"", "12", "N/A", "x.y", "-1.0", "12.", ".0", "0.0", "12.55"} {
		if got, ok := parseComputeCap(in); ok {
			t.Errorf("parseComputeCap(%q) = %d, true; want failure", in, got)
		}
	}
}

func TestParseNvidiaSmi(t *testing.T) {
	info, ok := parseNvidiaSmi("NVIDIA GeForce RTX 5070 Ti, 12.0, 16303\n")
	if !ok {
		t.Fatal("failed to parse a well-formed nvidia-smi row")
	}
	if info.Name != "NVIDIA GeForce RTX 5070 Ti" {
		t.Errorf("name = %q", info.Name)
	}
	if info.ComputeCap != 120 {
		t.Errorf("compute cap = %d, want 120", info.ComputeCap)
	}
	if info.VRAMMiB != 16303 {
		t.Errorf("VRAM = %d, want 16303", info.VRAMMiB)
	}
	if info.Vendor != gpuVendorNVIDIA {
		t.Errorf("vendor = %q", info.Vendor)
	}

	// Multi-GPU: device 0 is the one llama.cpp uses by default.
	multi, ok := parseNvidiaSmi("NVIDIA A100, 8.0, 40960\nNVIDIA GeForce GT 1030, 6.1, 2048\n")
	if !ok || multi.ComputeCap != 80 {
		t.Errorf("multi-GPU parse = %+v, %v; want the first row", multi, ok)
	}

	for _, in := range []string{"", "\n", "no devices were found", "garbage, N/A, N/A"} {
		if info, ok := parseNvidiaSmi(in); ok {
			t.Errorf("parseNvidiaSmi(%q) = %+v, true; want failure", in, info)
		}
	}
}

// ROCm archives are version-pinned upstream, and the previous unversioned
// name stopped matching anything at all — leaving the variant selectable but
// permanently un-downloadable.
func TestHIPUsesVersionedROCmAssets(t *testing.T) {
	for _, platform := range []string{"windows/amd64", "linux/amd64"} {
		asset, ok := engineAssetFor(engineVariantHIP, platform)
		if !ok {
			t.Errorf("no hip build registered for %s", platform)
			continue
		}
		if !strings.Contains(asset.Suffix, "rocm") {
			t.Errorf("%s hip suffix %q does not name a rocm archive", platform, asset.Suffix)
		}
	}
}

// -ngl must follow the engine that is actually installed. Under auto, a
// detected-but-not-downloaded CUDA build would otherwise hand -ngl to a
// CPU-only binary.
func TestAutoGPULayersFollowsInstalledEngine(t *testing.T) {
	withTempHome(t)
	withStubbedGPU(t, "NVIDIA GeForce RTX 5070 Ti, 12.0, 16303\n", nil)

	if isDarwin() {
		t.Skip("macOS always offloads; the interesting case is Windows/Linux")
	}
	installFakeEngine(t, engineVariantCPU)

	// A CPU engine is installed, so auto must not offload even though a
	// CUDA-capable GPU was detected.
	if got := resolveGPULayers(Config{}); got != 0 {
		t.Errorf("auto with a cpu engine installed = %d, want 0", got)
	}
	// Once the CUDA engine is actually installed, auto offloads.
	installFakeEngine(t, engineVariantCUDA)
	if got := resolveGPULayers(Config{}); got != maxGPULayers {
		t.Errorf("auto with a cuda engine installed = %d, want %d", got, maxGPULayers)
	}
}

// The two-step variant switch (`/set engine_variant` then `/download
// engine`) was impossible: the download command skipped whenever any engine
// existed, so the second step silently did nothing.
func TestEngineNeedsDownloadOnVariantSwitch(t *testing.T) {
	withTempHome(t)
	noGPU(t)

	if !engineNeedsDownload() {
		t.Error("a machine with no engine installed needs a download")
	}
	installFakeEngine(t, engineVariantCPU)
	if err := saveConfig(Config{CurrentModel: defaultModel}); err != nil {
		t.Fatal(err)
	}
	if engineNeedsDownload() {
		t.Error("a matching installed engine needs no download")
	}

	// Ask for a different variant: this is the case that used to no-op.
	want := engineVariantVulkan
	if _, ok := engineAssetFor(want, platformKey()); !ok {
		t.Skipf("no %s build for %s", want, platformKey())
	}
	if err := saveConfig(Config{CurrentModel: defaultModel, EngineVariant: want}); err != nil {
		t.Fatal(err)
	}
	if !engineNeedsDownload() {
		t.Errorf("switching to %s must trigger a download", want)
	}
}

// Under auto we never replace a working engine on detection alone — the user
// asked to be prompted rather than have ~510MB spent for them.
func TestAutoNeverReplacesInstalledEngine(t *testing.T) {
	withTempHome(t)
	withStubbedGPU(t, "NVIDIA GeForce RTX 5070 Ti, 12.0, 16303\n", nil)
	if isDarwin() {
		t.Skip("no GPU variants to switch to on macOS")
	}
	installFakeEngine(t, engineVariantCPU)
	if err := saveConfig(Config{CurrentModel: defaultModel}); err != nil {
		t.Fatal(err)
	}
	if engineNeedsDownload() {
		t.Error("auto must not schedule a ~510MB replacement of a working engine")
	}
	if got := plannedEngineVariant(Config{}); got != engineVariantCPU {
		t.Errorf("planned variant = %q, want the installed cpu build", got)
	}
	// But it must say something, or the GPU goes unused forever.
	if hint := engineUpgradeHint(); !strings.Contains(hint, "engine_variant cuda") {
		t.Errorf("no actionable upgrade hint for a detected GPU: %q", hint)
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

// A registry entry with a typo'd URL or filename only fails at download
// time, after the user has waited. Pin the invariants instead.
func TestModelRegistryIsWellFormed(t *testing.T) {
	seenName := map[string]bool{}
	seenFile := map[string]bool{}
	for _, m := range availableModels {
		if m.Name == "" || m.Filename == "" || m.URL == "" || m.Size == "" {
			t.Errorf("incomplete registry entry: %+v", m)
			continue
		}
		if seenName[m.Name] {
			t.Errorf("duplicate model name %q", m.Name)
		}
		if seenFile[m.Filename] {
			t.Errorf("duplicate filename %q — models would overwrite each other", m.Filename)
		}
		seenName[m.Name] = true
		seenFile[m.Filename] = true

		if !strings.HasSuffix(m.Filename, ".gguf") {
			t.Errorf("%s: filename %q is not a .gguf", m.Name, m.Filename)
		}
		// The URL's basename must match Filename, or the downloaded file
		// lands under a name isModelDownloaded() will never find.
		if !strings.HasSuffix(m.URL, "/"+m.Filename) {
			t.Errorf("%s: URL does not end in /%s: %s", m.Name, m.Filename, m.URL)
		}
		if !strings.HasPrefix(m.URL, "https://") {
			t.Errorf("%s: URL is not https: %s", m.Name, m.URL)
		}
	}
	// The default must exist, or every fresh install starts broken.
	if _, ok := findModel(defaultModel); !ok {
		t.Errorf("defaultModel %q is not in the registry", defaultModel)
	}
	// At least one model has to be able to drive /tools and /mcp.
	if _, ok := findModel("qwen3.5-4b"); !ok {
		t.Error("registry has no lightweight tool-calling model")
	}
}

// The first dense ~27B sized to sit entirely in 16GB. Unlike the MoE pair
// there is no expert offload to save an oversized quant — every spilled
// layer costs every token — so the quant has to fit outright, with room
// for the KV cache on top.
func TestQwen38CatalogEntry(t *testing.T) {
	m, ok := findModel("qwen3.8-27b")
	if !ok {
		t.Fatal("qwen3.8-27b is not in the registry")
	}
	gb := float64(parseModelSize(m.Size)) / 1e9
	if gb < 12 || gb > 15 {
		t.Errorf("qwen3.8-27b: %s (%.1f GB) is outside the fits-16GB-fully window",
			m.Size, gb)
	}
}
