package engine

import (
	"fmt"
	"runtime"
	"strings"

	"atlas.llm/internal/config"
)

// llamacppLatestURL is the GitHub API endpoint that always returns the latest
// ggml-org/llama.cpp release. We resolve it at download time to pick the
// correct prebuilt archive for the current OS/arch.
const llamacppLatestURL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

// Engine build variants. The macOS releases always ship the Metal backend,
// so there is no separate GPU archive for darwin — "cpu" there still gets
// GPU offload. On Windows/Linux the default archives are CPU-only, and
// Vulkan is the portable GPU option (works on NVIDIA, AMD, and Intel from
// one archive, unlike CUDA which also needs a matching runtime package).
const (
	EngineVariantAuto   = "auto"
	EngineVariantCPU    = "cpu"
	EngineVariantVulkan = "vulkan"
	EngineVariantCUDA   = "cuda"
	EngineVariantHIP    = "hip"
)

// engineAsset describes the release archive(s) for one variant on one
// platform.
type engineAsset struct {
	// Suffix matches the tail of the asset filename so the build tag can
	// vary between releases.
	Suffix string
	// Companion is a second archive extracted into the same directory.
	// CUDA needs one: the engine archive links against CUDA runtime DLLs
	// that ship separately in `cudart-llama-bin-*`.
	Companion string
	// Size is a rough human-readable download size, shown before a large
	// GPU download starts.
	Size string
}

// llamacppAssetSuffix maps variant -> GOOS/GOARCH -> asset. Assets are named
// like `llama-b8892-bin-win-cpu-x64.zip`.
//
// CUDA is deliberately absent: one variant maps to several archives there,
// selected by the GPU's compute capability. See cudaArchives. Look assets up
// through engineAssetFor rather than indexing this map, or CUDA goes missing.
//
// Sizes are the compressed download, measured against release b10375. They
// drift slowly and only feed "this is about to be a big download" messages,
// so being a few MB out is fine — being 2x out is not.
var LlamacppAssetSuffix = map[string]map[string]engineAsset{
	EngineVariantCPU: {
		"windows/amd64": {Suffix: "win-cpu-x64.zip", Size: "~17MB"},
		"windows/arm64": {Suffix: "win-cpu-arm64.zip", Size: "~11MB"},
		"darwin/amd64":  {Suffix: "macos-x64.tar.gz", Size: "~10MB"},
		"darwin/arm64":  {Suffix: "macos-arm64.tar.gz", Size: "~10MB"},
		"linux/amd64":   {Suffix: "ubuntu-x64.tar.gz", Size: "~15MB"},
		"linux/arm64":   {Suffix: "ubuntu-arm64.tar.gz", Size: "~12MB"},
	},
	EngineVariantVulkan: {
		"windows/amd64": {Suffix: "win-vulkan-x64.zip", Size: "~32MB"},
		"linux/amd64":   {Suffix: "ubuntu-vulkan-x64.tar.gz", Size: "~31MB"},
		"linux/arm64":   {Suffix: "ubuntu-vulkan-arm64.tar.gz", Size: "~25MB"},
	},
	EngineVariantHIP: {
		// Upstream renamed these from `win-hip-radeon-x64.zip` when it moved
		// to versioned ROCm archives, which left the old suffix matching
		// nothing at all. Version-pinned suffixes break on every ROCm bump;
		// PLANS #7 tracks making this matching tolerant.
		"windows/amd64": {Suffix: "win-rocm-7.14-x64.zip", Size: "~187MB"},
		"linux/amd64":   {Suffix: "ubuntu-rocm-7.14-x64.tar.gz", Size: "~195MB"},
	},
}

// cudaArchive is one CUDA engine build and the compute-capability window it
// supports, scaled by ten to match gpuInfo.ComputeCap.
type cudaArchive struct {
	MinCap int
	MaxCap int
	Asset  engineAsset
}

// cudaArchives lists CUDA builds newest-first per platform. Neither archive
// is a superset of the other, which is why this is a windowed list and not a
// single pinned entry:
//
//   - CUDA 13 dropped Maxwell, Pascal and Volta, so its floor is Turing (75).
//   - CUDA 12.4 predates Blackwell, so it carries no sm_120 kernels and its
//     ceiling is Hopper (90).
//
// A GTX 1080 (61) therefore needs 12.4 and an RTX 5070 Ti (120) needs 13.3.
// Selection walks the list in order and takes the first window that fits.
//
// Adding a future CUDA release means prepending a row; the newest row's open
// ceiling already covers cards newer than anything released today. The only
// forced edit is when a CUDA version drops architectures — then the previous
// row gets a real MaxCap.
var CudaArchives = map[string][]cudaArchive{
	"windows/amd64": {
		{
			MinCap: 75, MaxCap: 9999,
			Asset: engineAsset{
				Suffix:    "win-cuda-13.3-x64.zip",
				Companion: "cudart-llama-bin-win-cuda-13.3-x64.zip",
				Size:      "~510MB (139MB engine + 372MB CUDA runtime)",
			},
		},
		{
			MinCap: 50, MaxCap: 90,
			Asset: engineAsset{
				Suffix:    "win-cuda-12.4-x64.zip",
				Companion: "cudart-llama-bin-win-cuda-12.4-x64.zip",
				Size:      "~610MB (239MB engine + 373MB CUDA runtime)",
			},
		},
	},
}

// selectCUDAArchive returns the newest archive whose capability window
// contains cap.
func SelectCUDAArchive(platform string, cap int) (engineAsset, bool) {
	for _, a := range CudaArchives[platform] {
		if cap >= a.MinCap && cap <= a.MaxCap {
			return a.Asset, true
		}
	}
	return engineAsset{}, false
}

// widestCUDAArchive returns the build supporting the oldest hardware. It's
// the choice when we know CUDA exists for the platform but not which GPU is
// present — an archive that's too new fails at load on an older card, while
// an older archive at worst leaves performance on the table.
func WidestCUDAArchive(platform string) (engineAsset, bool) {
	list := CudaArchives[platform]
	if len(list) == 0 {
		return engineAsset{}, false
	}
	widest := list[0]
	for _, a := range list[1:] {
		if a.MinCap < widest.MinCap {
			widest = a
		}
	}
	return widest.Asset, true
}

// platformKey is the GOOS/GOARCH key used by both asset tables.
func PlatformKey() string { return runtime.GOOS + "/" + runtime.GOARCH }

// engineVariantAvailable reports whether llama.cpp publishes this variant
// for this platform. Deliberately hardware-independent: the CUDA build is
// offered on Windows x64 whether or not an NVIDIA card is present right now,
// and the check must stay cheap because engineVariantNames() is evaluated
// while package-level vars initialise — probing hardware there would make
// even `--version` spawn nvidia-smi.
func EngineVariantAvailable(variant, platform string) bool {
	if variant == EngineVariantCUDA {
		return len(CudaArchives[platform]) > 0
	}
	_, ok := LlamacppAssetSuffix[variant][platform]
	return ok
}

// engineAssetFor resolves the concrete archive for a variant on a platform,
// and reports whether one exists. CUDA resolves through capability
// selection — and so may fail where engineVariantAvailable succeeds, when a
// detected GPU falls outside every archive's window.
func EngineAssetFor(variant, platform string) (engineAsset, bool) {
	if variant == EngineVariantCUDA {
		if len(CudaArchives[platform]) == 0 {
			return engineAsset{}, false
		}
		if info, ok := DetectGPU(); ok {
			// A detected GPU that no archive covers means CUDA genuinely
			// isn't an option here — better to report that than to install
			// ~510MB that can't load the model.
			return SelectCUDAArchive(platform, info.ComputeCap)
		}
		return WidestCUDAArchive(platform)
	}
	asset, ok := LlamacppAssetSuffix[variant][platform]
	return asset, ok
}

// engineVariantNames lists the selectable variants for this platform, for
// help text and `/set` validation messages.
func EngineVariantNames() []string {
	key := PlatformKey()
	out := []string{EngineVariantAuto}
	for _, v := range []string{EngineVariantCPU, EngineVariantVulkan, EngineVariantCUDA, EngineVariantHIP} {
		if EngineVariantAvailable(v, key) {
			out = append(out, v)
		}
	}
	return out
}

// engineVariantIsGPU reports whether a variant's build carries a GPU
// backend. macOS is special-cased elsewhere: its "cpu" archive ships Metal.
func EngineVariantIsGPU(variant string) bool {
	switch variant {
	case EngineVariantVulkan, EngineVariantCUDA, EngineVariantHIP:
		return true
	}
	return false
}

// resolveEngineVariant turns "auto"/"" into a concrete variant.
func ResolveEngineVariant(v string) string {
	name := strings.ToLower(strings.TrimSpace(v))
	if name == EngineVariantAuto || name == "" {
		return autoEngineVariant()
	}
	if !EngineVariantIsGPU(name) {
		return EngineVariantCPU
	}
	// A GPU variant with no build for this platform would fail at download
	// time; fall back rather than persisting an unusable selection. This
	// checks availability, not the resolved asset: an explicit `cuda` on a
	// machine with no NVIDIA card stays `cuda` so the error names the real
	// problem instead of silently reverting to CPU.
	if !EngineVariantAvailable(name, PlatformKey()) {
		return EngineVariantCPU
	}
	return name
}

// autoEngineVariant picks a build with no user input. macOS needs nothing:
// its archive already carries Metal. Elsewhere we only choose a GPU build
// when the hardware is positively identified and an archive covers it — a
// GPU build without a working driver fails at load, so guessing is strictly
// worse than staying on CPU. Vulkan is never auto-selected for that reason:
// there's no probe that tells us a usable Vulkan driver is installed.
func autoEngineVariant() string {
	if IsDarwin() {
		return EngineVariantCPU
	}
	if info, ok := DetectGPU(); ok && info.Vendor == GpuVendorNVIDIA {
		if _, ok := EngineAssetFor(EngineVariantCUDA, PlatformKey()); ok {
			return EngineVariantCUDA
		}
	}
	return EngineVariantCPU
}

// engineAssetSuffix returns the release-asset suffix for a variant on this
// platform.
func EngineAssetSuffix(variant string) (engineAsset, error) {
	key := PlatformKey()
	if variant != EngineVariantCPU && variant != EngineVariantCUDA &&
		LlamacppAssetSuffix[variant] == nil {
		return engineAsset{}, fmt.Errorf("unknown engine variant %q (expected %s)",
			variant, strings.Join(EngineVariantNames(), ", "))
	}
	asset, ok := EngineAssetFor(variant, key)
	if !ok {
		if variant == EngineVariantCUDA {
			// Distinguish "no CUDA on this platform" from "your specific GPU
			// is outside every archive's range" — the second is actionable
			// only as "use vulkan instead", and the first isn't actionable.
			if info, found := DetectGPU(); found && len(CudaArchives[key]) > 0 {
				return engineAsset{}, fmt.Errorf(
					"no CUDA build covers %s (compute %.1f) — try `/set engine_variant vulkan`",
					info.Name, float64(info.ComputeCap)/10)
			}
		}
		if variant != EngineVariantCPU {
			return engineAsset{}, fmt.Errorf("no %s llama.cpp build available for %s (available here: %s)",
				variant, key, strings.Join(EngineVariantNames(), ", "))
		}
		return engineAsset{}, fmt.Errorf("no llama.cpp prebuilt available for %s", key)
	}
	return asset, nil
}

// maxGPULayers is the "offload everything" sentinel. llama.cpp clamps it to
// the model's actual layer count, so it doesn't need to be exact.
const MaxGPULayers = 999

// autoGPULayers decides the default -ngl when the user hasn't set one.
// macOS builds always carry Metal, so offloading is free there. On
// Windows/Linux only a GPU-enabled engine build can use it.
// A model larger than VRAM fails at load with everything offloaded, so auto
// picks the share that fits instead. Only when the estimate is confident: an
// unknown answer means offload everything and let a real failure be loud,
// rather than quietly halving performance forever.
//
// planOffload owns that arithmetic, including the mixture-of-experts case
// where the answer is a full offload plus --n-cpu-moe rather than a reduced
// -ngl. This returns only the -ngl half, for the callers that render it.
func autoGPULayers(variant string) int {
	m, err := config.CurrentModel()
	if err != nil {
		if runtime.GOOS != "darwin" && !EngineVariantIsGPU(effectiveEngineVariant(variant)) {
			return 0
		}
		return MaxGPULayers
	}
	cfg, _ := config.LoadConfig()
	// The caller's variant wins over the persisted one — resolveGPULayers
	// passes cfg.EngineVariant anyway, but tests probe other variants.
	cfg.EngineVariant = variant
	return planOffload(m, ResolveCtxSize(cfg), cfg).NGL
}

// effectiveEngineVariant is the build inference will actually run against,
// which is not always the one the setting resolves to. An explicit setting
// wins — the user is expected to /download it. Under auto we report the
// installed build instead of the detected one, so a machine where CUDA was
// detected but not yet downloaded doesn't hand -ngl to a CPU binary.
func effectiveEngineVariant(variant string) string {
	if explicit := strings.TrimSpace(variant); explicit != "" &&
		!strings.EqualFold(explicit, EngineVariantAuto) {
		return ResolveEngineVariant(explicit)
	}
	if IsEngineDownloaded() {
		return InstalledEngineVariant()
	}
	return ResolveEngineVariant(EngineVariantAuto)
}

// resolveGPULayers returns the -ngl value to pass to llama-server.
func ResolveGPULayers(cfg config.Config) int {
	if cfg.GPULayers != nil {
		if *cfg.GPULayers < 0 {
			return 0
		}
		return *cfg.GPULayers
	}
	return autoGPULayers(cfg.EngineVariant)
}

// gpuLayersDisplay renders the setting for `/set`, distinguishing an
// explicit value from the auto-detected default.
func GpuLayersDisplay(cfg config.Config) string {
	n := ResolveGPULayers(cfg)
	label := "all layers"
	if n == 0 {
		label = "CPU only"
	} else if n < MaxGPULayers {
		label = fmt.Sprintf("%d layers", n)
	}
	if cfg.GPULayers == nil {
		return fmt.Sprintf("auto (%s)", label)
	}
	return fmt.Sprintf("%d (%s)", n, label)
}

// isDarwin is spelled out here because the macOS engine archive always
// carries Metal, which changes the advice for two settings.
func IsDarwin() bool { return runtime.GOOS == "darwin" }
