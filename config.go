package main

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

type Model struct {
	Name     string `json:"name"`
	Filename string `json:"filename"`
	URL      string `json:"url"`
	Size     string `json:"size"`
}

var availableModels = []Model{
	{
		Name:     "gemma-3-1b-it",
		Filename: "gemma-3-1b-it-Q4_K_M.gguf",
		URL:      "https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf",
		Size:     "~700MB",
	},
	{
		Name:     "gemma-3-4b-it",
		Filename: "gemma-3-4b-it-Q4_K_M.gguf",
		URL:      "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf",
		Size:     "~2.5GB",
	},
	{
		// The lightest model in the registry that reliably emits tool calls,
		// so /tools and /mcp actually work without pulling the 5.7GB 9B.
		Name:     "qwen3.5-4b",
		Filename: "Qwen3.5-4B-Q4_K_M.gguf",
		URL:      "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
		Size:     "~2.7GB",
	},
	{
		Name:     "gemma-4-e2b-it",
		Filename: "gemma-4-E2B-it-Q4_K_M.gguf",
		URL:      "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf",
		Size:     "~2.9GB",
	},
	{
		// Gemma 4, unlike Gemma 3, ships a tool-calling chat template, so
		// this family can drive /tools and /mcp.
		Name:     "gemma-4-e4b-it",
		Filename: "gemma-4-E4B-it-Q4_K_M.gguf",
		URL:      "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf",
		Size:     "~5.0GB",
	},
	{
		Name:     "gemma-4-12b-it",
		Filename: "gemma-4-12b-it-Q4_K_M.gguf",
		URL:      "https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/main/gemma-4-12b-it-Q4_K_M.gguf",
		Size:     "~7.1GB",
	},
	{
		// Fills the gap between qwen3.5-4b and qwen3.5-9b. Same family as
		// the 14B below, which already tool-calls reliably.
		Name:     "ministral-3-8b-instruct",
		Filename: "Ministral-3-8B-Instruct-2512-Q4_K_M.gguf",
		URL:      "https://huggingface.co/unsloth/Ministral-3-8B-Instruct-2512-GGUF/resolve/main/Ministral-3-8B-Instruct-2512-Q4_K_M.gguf",
		Size:     "~5.2GB",
	},
	{
		Name:     "qwen3.5-9b",
		Filename: "Qwen3.5-9B-Q4_K_M.gguf",
		URL:      "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
		Size:     "~5.7GB",
	},
	{
		Name:     "ministral-3-14b-instruct",
		Filename: "Ministral-3-14B-Instruct-2512-Q4_K_M.gguf",
		URL:      "https://huggingface.co/unsloth/Ministral-3-14B-Instruct-2512-GGUF/resolve/main/Ministral-3-14B-Instruct-2512-Q4_K_M.gguf",
		Size:     "~8.2GB",
	},
	{
		// First mixture-of-experts entry, and the first model aimed at code.
		// 30B of weights with 3B active per token, so it generates at roughly
		// 4B speed while knowing what a 30B knows — which is the trade a
		// 16GB card wants. IQ3 rather than the nicer Q4_K_M (18.6GB) because
		// it sits almost entirely in 16GB — measured against a 5070 Ti with
		// a desktop running, five of its 48 layers spill their experts to
		// system RAM. Q4_K_M works too and is the better model; it spills
		// about twenty.
		Name:     "qwen3-coder-30b-a3b",
		Filename: "Qwen3-Coder-30B-A3B-Instruct-UD-IQ3_XXS.gguf",
		URL:      "https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct-UD-IQ3_XXS.gguf",
		Size:     "~12.8GB",
	},
	{
		// The general-purpose counterpart to the coder above: 26B of weights,
		// 4B active. Same family as the gemma-4 entries, so it carries the
		// same tool-calling chat template.
		Name:     "gemma-4-26b-a4b-it",
		Filename: "gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf",
		URL:      "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf",
		Size:     "~12.9GB",
	},
	{
		// Meta's local-agent model (Aug 2026), built around tool calling.
		// First entry aimed at 24/32GB machines — the fit column reads
		// "too big" on 16GB, and no quant of this dense 30B fits there.
		// unsloth ships only UD dynamic quants, hence no Q4_K_M. Text-only:
		// vision needs the separate mmproj GGUF, which we don't download.
		Name:     "muse-glimmer-30b",
		Filename: "Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
		URL:      "https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF/resolve/main/Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
		Size:     "~15.9GB",
	},
	{
		// Qwen's dense 27B (Aug 2026, Apache 2.0), aimed squarely at agentic
		// work — the strongest /tools driver in the registry. UD-Q3_K_XL
		// rather than Q4_K_M (17.1GB) because a dense model pays for every
		// spilled layer on every token; this quant sits entirely in 16GB
		// with room for the KV cache, which its hybrid attention keeps
		// small (most layers are linear-attention, like Qwen3.5). Text-only
		// here: vision needs the separate mmproj GGUF, which we don't
		// download. Needs a llama.cpp build recent enough to know the arch
		// (`/download engine` refreshes an older install).
		Name:     "qwen3.8-27b",
		Filename: "Qwen3.8-27B-UD-Q3_K_XL.gguf",
		URL:      "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q3_K_XL.gguf",
		Size:     "~13.4GB",
	},
	{
		// The 2-bit cut of the dense 27B above, kept for context rather
		// than quality: at ~9GB the weights leave a 12GB card room for a
		// ~150K-token q4_0 KV cache entirely on the GPU — affordable only
		// because the hybrid attention holds KV in a fraction of the
		// layers. 2-bit is audibly lossy; when 150K isn't the point, the
		// Q3_K_XL entry above is the better model.
		Name:     "qwen3.8-27b-iq2",
		Filename: "Qwen3.8-27B-UD-IQ2_XXS.gguf",
		URL:      "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-IQ2_XXS.gguf",
		Size:     "~9.0GB",
	},
	{
		// Abliterated cut of the dense 27B above — refusal directions ablated
		// so it answers where the stock model declines. Of the abliterations
		// published for qwen3.8, this Heretic ARA build is the one that
		// hardly costs quality: KL divergence 0.0085 against the base
		// (huihui's first-party attempt measures 6x worse) with refusals at
		// ~0/100. Q3_K_M so it sits entirely in 16GB like the stock entry;
		// the RVN- filename is the quantizer's own prefix, which the URL
		// must match. Text-only — this abliteration drops the vision stack.
		Name:     "qwen3.8-27b-heretic",
		Filename: "RVN-Q3_K_M.gguf",
		URL:      "https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF/resolve/main/RVN-Q3_K_M.gguf",
		Size:     "~13.3GB",
	},
	{
		// The Q4_K_M cut of the Heretic abliteration above, for 24GB cards —
		// on 16GB a dense model pays for every spilled layer on every token,
		// so the Q3_K_M entry is the right pick there. Same repo, same
		// weights, one quant step nicer.
		Name:     "qwen3.8-27b-heretic-q4",
		Filename: "Qwen3.8-27B-Heretic-Q4_K_M.gguf",
		URL:      "https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF/resolve/main/Qwen3.8-27B-Heretic-Q4_K_M.gguf",
		Size:     "~16.5GB",
	},
}

const defaultModel = "gemma-3-1b-it"

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
	engineVariantAuto   = "auto"
	engineVariantCPU    = "cpu"
	engineVariantVulkan = "vulkan"
	engineVariantCUDA   = "cuda"
	engineVariantHIP    = "hip"
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
var llamacppAssetSuffix = map[string]map[string]engineAsset{
	engineVariantCPU: {
		"windows/amd64": {Suffix: "win-cpu-x64.zip", Size: "~17MB"},
		"windows/arm64": {Suffix: "win-cpu-arm64.zip", Size: "~11MB"},
		"darwin/amd64":  {Suffix: "macos-x64.tar.gz", Size: "~10MB"},
		"darwin/arm64":  {Suffix: "macos-arm64.tar.gz", Size: "~10MB"},
		"linux/amd64":   {Suffix: "ubuntu-x64.tar.gz", Size: "~15MB"},
		"linux/arm64":   {Suffix: "ubuntu-arm64.tar.gz", Size: "~12MB"},
	},
	engineVariantVulkan: {
		"windows/amd64": {Suffix: "win-vulkan-x64.zip", Size: "~32MB"},
		"linux/amd64":   {Suffix: "ubuntu-vulkan-x64.tar.gz", Size: "~31MB"},
		"linux/arm64":   {Suffix: "ubuntu-vulkan-arm64.tar.gz", Size: "~25MB"},
	},
	engineVariantHIP: {
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
var cudaArchives = map[string][]cudaArchive{
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
func selectCUDAArchive(platform string, cap int) (engineAsset, bool) {
	for _, a := range cudaArchives[platform] {
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
func widestCUDAArchive(platform string) (engineAsset, bool) {
	list := cudaArchives[platform]
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
func platformKey() string { return runtime.GOOS + "/" + runtime.GOARCH }

// engineVariantAvailable reports whether llama.cpp publishes this variant
// for this platform. Deliberately hardware-independent: the CUDA build is
// offered on Windows x64 whether or not an NVIDIA card is present right now,
// and the check must stay cheap because engineVariantNames() is evaluated
// while package-level vars initialise — probing hardware there would make
// even `--version` spawn nvidia-smi.
func engineVariantAvailable(variant, platform string) bool {
	if variant == engineVariantCUDA {
		return len(cudaArchives[platform]) > 0
	}
	_, ok := llamacppAssetSuffix[variant][platform]
	return ok
}

// engineAssetFor resolves the concrete archive for a variant on a platform,
// and reports whether one exists. CUDA resolves through capability
// selection — and so may fail where engineVariantAvailable succeeds, when a
// detected GPU falls outside every archive's window.
func engineAssetFor(variant, platform string) (engineAsset, bool) {
	if variant == engineVariantCUDA {
		if len(cudaArchives[platform]) == 0 {
			return engineAsset{}, false
		}
		if info, ok := detectGPU(); ok {
			// A detected GPU that no archive covers means CUDA genuinely
			// isn't an option here — better to report that than to install
			// ~510MB that can't load the model.
			return selectCUDAArchive(platform, info.ComputeCap)
		}
		return widestCUDAArchive(platform)
	}
	asset, ok := llamacppAssetSuffix[variant][platform]
	return asset, ok
}

// engineVariantNames lists the selectable variants for this platform, for
// help text and `/set` validation messages.
func engineVariantNames() []string {
	key := platformKey()
	out := []string{engineVariantAuto}
	for _, v := range []string{engineVariantCPU, engineVariantVulkan, engineVariantCUDA, engineVariantHIP} {
		if engineVariantAvailable(v, key) {
			out = append(out, v)
		}
	}
	return out
}

// engineVariantIsGPU reports whether a variant's build carries a GPU
// backend. macOS is special-cased elsewhere: its "cpu" archive ships Metal.
func engineVariantIsGPU(variant string) bool {
	switch variant {
	case engineVariantVulkan, engineVariantCUDA, engineVariantHIP:
		return true
	}
	return false
}

// resolveEngineVariant turns "auto"/"" into a concrete variant.
func resolveEngineVariant(v string) string {
	name := strings.ToLower(strings.TrimSpace(v))
	if name == engineVariantAuto || name == "" {
		return autoEngineVariant()
	}
	if !engineVariantIsGPU(name) {
		return engineVariantCPU
	}
	// A GPU variant with no build for this platform would fail at download
	// time; fall back rather than persisting an unusable selection. This
	// checks availability, not the resolved asset: an explicit `cuda` on a
	// machine with no NVIDIA card stays `cuda` so the error names the real
	// problem instead of silently reverting to CPU.
	if !engineVariantAvailable(name, platformKey()) {
		return engineVariantCPU
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
	if isDarwin() {
		return engineVariantCPU
	}
	if info, ok := detectGPU(); ok && info.Vendor == gpuVendorNVIDIA {
		if _, ok := engineAssetFor(engineVariantCUDA, platformKey()); ok {
			return engineVariantCUDA
		}
	}
	return engineVariantCPU
}

// engineAssetSuffix returns the release-asset suffix for a variant on this
// platform.
func engineAssetSuffix(variant string) (engineAsset, error) {
	key := platformKey()
	if variant != engineVariantCPU && variant != engineVariantCUDA &&
		llamacppAssetSuffix[variant] == nil {
		return engineAsset{}, fmt.Errorf("unknown engine variant %q (expected %s)",
			variant, strings.Join(engineVariantNames(), ", "))
	}
	asset, ok := engineAssetFor(variant, key)
	if !ok {
		if variant == engineVariantCUDA {
			// Distinguish "no CUDA on this platform" from "your specific GPU
			// is outside every archive's range" — the second is actionable
			// only as "use vulkan instead", and the first isn't actionable.
			if info, found := detectGPU(); found && len(cudaArchives[key]) > 0 {
				return engineAsset{}, fmt.Errorf(
					"no CUDA build covers %s (compute %.1f) — try `/set engine_variant vulkan`",
					info.Name, float64(info.ComputeCap)/10)
			}
		}
		if variant != engineVariantCPU {
			return engineAsset{}, fmt.Errorf("no %s llama.cpp build available for %s (available here: %s)",
				variant, key, strings.Join(engineVariantNames(), ", "))
		}
		return engineAsset{}, fmt.Errorf("no llama.cpp prebuilt available for %s", key)
	}
	return asset, nil
}

// maxGPULayers is the "offload everything" sentinel. llama.cpp clamps it to
// the model's actual layer count, so it doesn't need to be exact.
const maxGPULayers = 999

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
	m, err := currentModel()
	if err != nil {
		if runtime.GOOS != "darwin" && !engineVariantIsGPU(effectiveEngineVariant(variant)) {
			return 0
		}
		return maxGPULayers
	}
	cfg, _ := loadConfig()
	// The caller's variant wins over the persisted one — resolveGPULayers
	// passes cfg.EngineVariant anyway, but tests probe other variants.
	cfg.EngineVariant = variant
	return planOffload(m, resolveCtxSize(cfg), cfg).NGL
}

// effectiveEngineVariant is the build inference will actually run against,
// which is not always the one the setting resolves to. An explicit setting
// wins — the user is expected to /download it. Under auto we report the
// installed build instead of the detected one, so a machine where CUDA was
// detected but not yet downloaded doesn't hand -ngl to a CPU binary.
func effectiveEngineVariant(variant string) string {
	if explicit := strings.TrimSpace(variant); explicit != "" &&
		!strings.EqualFold(explicit, engineVariantAuto) {
		return resolveEngineVariant(explicit)
	}
	if isEngineDownloaded() {
		return installedEngineVariant()
	}
	return resolveEngineVariant(engineVariantAuto)
}

// resolveGPULayers returns the -ngl value to pass to llama-server.
func resolveGPULayers(cfg Config) int {
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
func gpuLayersDisplay(cfg Config) string {
	n := resolveGPULayers(cfg)
	label := "all layers"
	if n == 0 {
		label = "CPU only"
	} else if n < maxGPULayers {
		label = fmt.Sprintf("%d layers", n)
	}
	if cfg.GPULayers == nil {
		return fmt.Sprintf("auto (%s)", label)
	}
	return fmt.Sprintf("%d (%s)", n, label)
}

// Config carries both tag sets on every field: the active config.json is
// JSON, while named profiles are piml (the Atlas suite's format). A field
// with one tag missing would silently vanish from one of the two files.
type Config struct {
	CurrentModel string `json:"current_model" piml:"current_model"`
	MaxTokens    int    `json:"max_tokens,omitempty" piml:"max_tokens,omitempty"`
	ToolsEnabled bool   `json:"tools_enabled,omitempty" piml:"tools_enabled,omitempty"`

	// GPULayers is the -ngl value handed to llama-server. nil means "auto"
	// — a pointer rather than an int so an explicit 0 (force CPU) is
	// distinguishable from an absent setting.
	GPULayers *int `json:"gpu_layers,omitempty" piml:"gpu_layers,omitempty"`

	// EngineVariant selects which llama.cpp release archive to download:
	// "cpu" (default) or "vulkan". Empty means auto.
	EngineVariant string `json:"engine_variant,omitempty" piml:"engine_variant,omitempty"`

	// Reasoning controls the model's internal <think> block: "on", "off",
	// or "" / "auto" for the default. Only affects models that have one.
	Reasoning string `json:"reasoning,omitempty" piml:"reasoning,omitempty"`

	// ShowThinking streams the think block's text into the transcript,
	// dimmed, instead of the byte counter. Display-only: the think text
	// never joins the history sent back to the model.
	ShowThinking bool `json:"show_thinking,omitempty" piml:"show_thinking,omitempty"`

	// MaxToolRounds caps tool-call rounds per message. 0 means the default;
	// a negative value means no cap.
	MaxToolRounds int `json:"max_tool_rounds,omitempty" piml:"max_tool_rounds,omitempty"`

	// CtxSize is the context window llama-server is started with (-c).
	// 0 means the default. The ceiling is whatever the model was trained
	// for, which is read from its GGUF metadata.
	CtxSize int `json:"ctx_size,omitempty" piml:"ctx_size,omitempty"`

	// Endpoint points inference at a llama-server someone else is running,
	// typically another atlas.llm in --serve mode. When set, this install
	// needs no engine and no model file: it becomes a client. Empty means
	// run inference locally.
	Endpoint string `json:"endpoint,omitempty" piml:"endpoint,omitempty"`

	// EndpointKey is the bearer token for Endpoint, for a server started
	// with --api-key. Empty is the normal case on a trusted LAN.
	EndpointKey string `json:"endpoint_key,omitempty" piml:"endpoint_key,omitempty"`

	// AMAEnabled toggles /ama: whether the agent may ask the user questions
	// through the interactive ask_user picker instead of deciding alone.
	AMAEnabled bool `json:"ama_enabled,omitempty" piml:"ama_enabled,omitempty"`

	// Engine tuning. Each field maps onto one llama-server flag; the zero
	// value always means "auto" — launch exactly as before the setting
	// existed. Pointer types are for settings where an explicit zero is a
	// real choice distinct from auto (the GPULayers lesson).
	KVOffload      string   `json:"kv_offload,omitempty" piml:"kv_offload,omitempty"`           // "off" → --no-kv-offload
	FlashAttn      string   `json:"flash_attn,omitempty" piml:"flash_attn,omitempty"`           // "on"/"off" → -fa; "" = auto
	CacheTypeK     string   `json:"cache_type_k,omitempty" piml:"cache_type_k,omitempty"`       // --cache-type-k; "" = auto
	CacheTypeV     string   `json:"cache_type_v,omitempty" piml:"cache_type_v,omitempty"`       // --cache-type-v; "" = auto
	Threads        int      `json:"threads,omitempty" piml:"threads,omitempty"`                 // -t; 0 = auto
	BatchSize      int      `json:"batch_size,omitempty" piml:"batch_size,omitempty"`           // -b; 0 = auto
	UBatchSize     int      `json:"ubatch_size,omitempty" piml:"ubatch_size,omitempty"`         // -ub; 0 = auto
	Parallel       int      `json:"parallel,omitempty" piml:"parallel,omitempty"`               // --parallel; 0 = auto
	CacheReuse     *int     `json:"cache_reuse,omitempty" piml:"cache_reuse,omitempty"`         // --cache-reuse; nil = auto, 0 = off
	Mmap           string   `json:"mmap,omitempty" piml:"mmap,omitempty"`                       // "off" → --no-mmap
	Mlock          string   `json:"mlock,omitempty" piml:"mlock,omitempty"`                     // "on" → --mlock
	Seed           *int     `json:"seed,omitempty" piml:"seed,omitempty"`                       // --seed; nil = auto
	Temperature    *float64 `json:"temperature,omitempty" piml:"temperature,omitempty"`         // per-request; nil = 0.2
	OverrideTensor string   `json:"override_tensor,omitempty" piml:"override_tensor,omitempty"` // -ot
	ContextShift   string   `json:"context_shift,omitempty" piml:"context_shift,omitempty"`     // "on" → --context-shift
}

// Reasoning settings.
const (
	reasoningAuto = "auto"
	reasoningOn   = "on"
	reasoningOff  = "off"
)

// reasoningEnabledFor reports whether to let the model think, given whether
// this is an agentic turn.
//
// auto splits the two, because the cost differs enormously by task. Measured
// on qwen3.5-4b with "in one sentence, what is a KV cache?": thinking gave a
// first token at 62.1s and finished at 63.9s; without it, 0.3s and 3.2s. A
// twentyfold difference for a question that needed none of it.
//
// So auto turns thinking off for plain chat and leaves it on for tool-driven
// turns, where it measurably improves which tool gets called. `on` and `off`
// override both. One-shot commands never think either way.
func reasoningEnabledFor(cfg Config, agentic bool) bool {
	switch strings.ToLower(strings.TrimSpace(cfg.Reasoning)) {
	case reasoningOff:
		return false
	case reasoningOn:
		return true
	default:
		return agentic
	}
}

// reasoningEnabled is the conversational-turn case.
func reasoningEnabled(cfg Config) bool { return reasoningEnabledFor(cfg, false) }

// reasoningDisplay renders the setting for /set and /config.
func reasoningDisplay(cfg Config) string {
	switch strings.ToLower(strings.TrimSpace(cfg.Reasoning)) {
	case reasoningOff:
		return "off (faster replies; may reduce tool-use accuracy)"
	case reasoningOn:
		return "on (always; slower but best at multi-step work)"
	default:
		return "auto (off for chat, on for tool-driven turns)"
	}
}

// defaultMaxToolRounds is the tool-call round budget for one message when
// nothing is configured.
//
// Raised from 20 once identical-call detection landed: that catches a stuck
// model directly, so this no longer has to double as the runaway guard and
// can be generous enough for work that legitimately needs many steps.
const defaultMaxToolRounds = 40

// unlimitedToolRounds is what resolveMaxToolRounds returns when the cap is
// switched off. The repeated-call detector and Esc remain as backstops.
const unlimitedToolRounds = 0

// resolveMaxToolRounds returns the per-message tool-call budget, or
// unlimitedToolRounds when the user has turned the cap off.
func resolveMaxToolRounds(cfg Config) int {
	switch {
	case cfg.MaxToolRounds < 0:
		return unlimitedToolRounds
	case cfg.MaxToolRounds == 0:
		return defaultMaxToolRounds
	default:
		return cfg.MaxToolRounds
	}
}

// maxToolRoundsDisplay renders the setting for `/set` and `/config`.
func maxToolRoundsDisplay(cfg Config) string {
	n := resolveMaxToolRounds(cfg)
	if n == unlimitedToolRounds {
		return "off (no cap)"
	}
	if cfg.MaxToolRounds == 0 {
		return fmt.Sprintf("%d (default)", n)
	}
	return fmt.Sprintf("%d", n)
}

// defaultCtxSize is the context window used when nothing is configured.
// Modest on purpose: KV cache memory scales linearly with it, and most
// models here are run on laptops.
const defaultCtxSize = 16384

// maxConfigurableCtx caps what /set will accept even when a model claims a
// larger trained context. Once 131072, from when a KV cache at Qwen's full
// 262144 meant tens of gigabytes of f16; with quantized cache types and
// hybrid-attention models that hold KV in only a fraction of their layers,
// contexts that size are genuinely servable on consumer cards. The offload
// planner still charges the cache against VRAM before every launch, so the
// cap only has to reject the absurd, not police fit.
const maxConfigurableCtx = 262144

// minConfigurableCtx keeps a value from being set so small that the system
// prompt and tool definitions alone won't fit.
const minConfigurableCtx = 2048

// resolveCtxSize returns the context size to start llama-server with,
// clamped to what the current model was trained for when that is known.
func resolveCtxSize(cfg Config) int {
	n := cfg.CtxSize
	if n <= 0 {
		n = defaultCtxSize
	}
	if trained := currentModelTrainedContext(); trained > 0 && n > trained {
		n = trained
	}
	if n > maxConfigurableCtx {
		n = maxConfigurableCtx
	}
	if n < minConfigurableCtx {
		n = minConfigurableCtx
	}
	return n
}

// maxTokensCeiling is the largest reply length that still leaves room for
// the prompt and history. Derived from the context window rather than fixed,
// so raising ctx_size raises this too.
func maxTokensCeiling(cfg Config) int {
	n := effectiveCtxSize(cfg) * 3 / 4
	if n < 512 {
		n = 512
	}
	return n
}

// ctxSizeDisplay renders the setting for `/set`, including the model's
// trained ceiling so the headroom is visible.
func ctxSizeDisplay(cfg Config) string {
	eff := resolveCtxSize(cfg)
	set := "auto"
	if cfg.CtxSize > 0 {
		set = fmt.Sprintf("%d", cfg.CtxSize)
	}
	trained := currentModelTrainedContext()
	if trained == 0 {
		return fmt.Sprintf("%s (using %d)", set, eff)
	}
	return fmt.Sprintf("%s (using %d; this model was trained for %d)", set, eff, trained)
}

// defaultMaxTokens is the reply-length cap applied when the config hasn't
// set one. Sized to fit almost any chat response while leaving plenty of
// the 16K ctx window for conversation history.
const defaultMaxTokens = 4096

func atlasDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(home, ".atlas", "atlas.llm.data")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}
	return dir, nil
}

func modelsDir() (string, error) {
	base, err := atlasDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(base, "models")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}
	return dir, nil
}

func configPath() (string, error) {
	base, err := atlasDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(base, "config.json"), nil
}

// engineDir is the directory where the extracted llama.cpp binaries live.
func engineDir() (string, error) {
	base, err := atlasDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(base, "engine")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}
	return dir, nil
}

// findEngineBinary locates llama-cli[.exe] inside the engine dir. llama.cpp
// archives nest the binary under paths like `build/bin/` depending on the
// asset, so we walk to find it rather than hard-coding a location.
func findEngineBinary() (string, error) {
	return findEngineExecutable("llama-cli")
}

// findEngineServer locates llama-server[.exe] in the engine dir. Used for
// the persistent server mode so we don't re-load the model on every turn.
func findEngineServer() (string, error) {
	return findEngineExecutable("llama-server")
}

func findEngineExecutable(base string) (string, error) {
	dir, err := engineDir()
	if err != nil {
		return "", err
	}
	target := base
	if runtime.GOOS == "windows" {
		target = base + ".exe"
	}
	var found string
	err = filepath.WalkDir(dir, func(p string, d fs.DirEntry, werr error) error {
		if werr != nil {
			return nil
		}
		if !d.IsDir() && d.Name() == target {
			found = p
			return filepath.SkipAll
		}
		return nil
	})
	if err != nil {
		return "", err
	}
	if found == "" {
		return "", fmt.Errorf("%s not found under %s", target, dir)
	}
	return found, nil
}

func modelPath(m Model) (string, error) {
	dir, err := modelsDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, m.Filename), nil
}

func findModel(name string) (Model, bool) {
	for _, m := range availableModels {
		if m.Name == name {
			return m, true
		}
	}
	return Model{}, false
}

func loadConfig() (Config, error) {
	cfg := Config{CurrentModel: defaultModel}
	p, err := configPath()
	if err != nil {
		return cfg, err
	}
	data, err := os.ReadFile(p)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return cfg, err
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, err
	}
	if cfg.CurrentModel == "" {
		cfg.CurrentModel = defaultModel
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = defaultMaxTokens
	}
	return cfg, nil
}

func saveConfig(cfg Config) error {
	p, err := configPath()
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(p, data, 0644)
}

func isModelDownloaded(m Model) bool {
	p, err := modelPath(m)
	if err != nil {
		return false
	}
	info, err := os.Stat(p)
	return err == nil && !info.IsDir() && info.Size() > 0
}

func isEngineDownloaded() bool {
	p, err := findEngineBinary()
	if err != nil {
		return false
	}
	info, err := os.Stat(p)
	return err == nil && !info.IsDir() && info.Size() > 0
}

func currentModel() (Model, error) {
	cfg, err := loadConfig()
	if err != nil {
		return Model{}, err
	}
	m, ok := findModel(cfg.CurrentModel)
	if !ok {
		return Model{}, fmt.Errorf("unknown model in config: %s", cfg.CurrentModel)
	}
	return m, nil
}

// engineVariantDisplay renders the engine_variant setting, showing what
// "auto" resolves to on this platform.
func engineVariantDisplay(cfg Config) string {
	resolved := resolveEngineVariant(cfg.EngineVariant)
	if strings.TrimSpace(cfg.EngineVariant) == "" ||
		strings.EqualFold(cfg.EngineVariant, engineVariantAuto) {
		if runtime.GOOS == "darwin" {
			return "auto (" + resolved + ", Metal built in)"
		}
		return "auto (" + resolved + ")"
	}
	return resolved
}

// gpuHelpRows builds the "Performance" block of the in-app /help, tailored
// to what this platform can actually do. There's no point telling a Mac user
// to install a Vulkan build, or a Windows user that Metal is built in.
func gpuHelpRows() [][2]string {
	rows := [][2]string{
		{"/set gpu_layers", "auto (default) · 0 for CPU-only · N to offload N layers"},
	}
	cfg, _ := loadConfig()
	installed := installedEngineVariant()
	// installedEngineVariant answers "cpu" for a missing marker, which is
	// right for pre-variant installs but reads as a lie before the first
	// /download. Separate the two for display only.
	installedLabel := installed
	if !isEngineDownloaded() {
		installedLabel = "not installed"
	}

	switch {
	case runtime.GOOS == "darwin":
		rows = append(rows, [2]string{"", "Metal is built into the macOS engine — GPU is on by default"})
	case engineVariantIsGPU(installed):
		rows = append(rows, [2]string{"", fmt.Sprintf("engine: %s build installed — GPU offload active", installed)})
	default:
		opts := engineVariantNames()
		switch {
		case len(opts) <= 2:
			rows = append(rows, [2]string{"", "no GPU llama.cpp build published for this platform"})
		default:
			// Naming the detected card turns a generic menu into a specific
			// instruction, which is the difference between the setting being
			// discoverable and it sitting unused.
			if hint := engineUpgradeHint(); hint != "" {
				for i, line := range strings.Split(hint, "\n") {
					label := ""
					if i == 0 {
						label = "GPU detected"
					}
					rows = append(rows, [2]string{label, line})
				}
				break
			}
			rows = append(rows,
				[2]string{"/set engine_variant", "GPU builds here: " + strings.Join(opts[2:], ", ")},
				[2]string{"", "then /download engine to install it (CPU-only until you do)"},
			)
		}
	}
	rows = append(rows, [2]string{"/set", fmt.Sprintf("current: gpu_layers=%s, engine=%s",
		gpuLayersDisplay(cfg), installedLabel)})
	return rows
}

// parseModelSize turns a registry Size string ("~700MB", "~2.5GB") into
// bytes for comparison. Returns 0 when it can't be parsed, which sorts such
// entries last rather than making them look tiny.
func parseModelSize(s string) int64 {
	s = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(s), "~"))
	upper := strings.ToUpper(s)
	var mult float64
	switch {
	case strings.HasSuffix(upper, "GB"):
		mult, upper = 1e9, strings.TrimSuffix(upper, "GB")
	case strings.HasSuffix(upper, "MB"):
		mult, upper = 1e6, strings.TrimSuffix(upper, "MB")
	default:
		return 0
	}
	n, err := strconv.ParseFloat(strings.TrimSpace(upper), 64)
	if err != nil || n <= 0 {
		return 0
	}
	// Round rather than truncate: 8.2 * 1e9 lands just under 8.2e9 in
	// binary floating point, which would report 8199999999 bytes.
	return int64(math.Round(n * mult))
}

// lightestModel returns the smallest model in the registry — the safe one
// to fall back to when a heavy model makes startup unusable. Falls back to
// defaultModel if no size parses.
func lightestModel() Model {
	best, bestSize := Model{}, int64(0)
	for _, m := range availableModels {
		size := parseModelSize(m.Size)
		if size == 0 {
			continue
		}
		if bestSize == 0 || size < bestSize {
			best, bestSize = m, size
		}
	}
	if bestSize == 0 {
		if m, ok := findModel(defaultModel); ok {
			return m
		}
		return availableModels[0]
	}
	return best
}

// defaultServePort matches llama-server's own default, so a user who reaches
// for a port number guesses right.
const defaultServePort = 8080

// normalizeEndpoint canonicalises what a user types into `/set endpoint`.
// Accepts "192.168.1.50", "192.168.1.50:8080", or a full URL, because all
// three are things people reasonably type for a machine on their LAN.
// Returns "" for input that clears the setting.
func normalizeEndpoint(raw string) (string, error) {
	s := strings.TrimSpace(raw)
	if s == "" || strings.EqualFold(s, "local") || strings.EqualFold(s, "off") ||
		strings.EqualFold(s, "none") {
		return "", nil
	}
	if !strings.Contains(s, "://") {
		s = "http://" + s
	}
	u, err := url.Parse(s)
	if err != nil {
		return "", fmt.Errorf("invalid endpoint %q: %v", raw, err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return "", fmt.Errorf("endpoint %q: expected an http:// or https:// address", raw)
	}
	if u.Hostname() == "" {
		return "", fmt.Errorf("endpoint %q: no host in the address", raw)
	}
	if u.Port() == "" {
		u.Host = net.JoinHostPort(u.Hostname(), strconv.Itoa(defaultServePort))
	}
	// Only scheme://host:port is meaningful; the API paths are ours to append.
	return strings.TrimRight((&url.URL{Scheme: u.Scheme, Host: u.Host}).String(), "/"), nil
}

// remoteEndpoint returns the configured endpoint and key, or "" when this
// install runs inference locally.
func remoteEndpoint() (string, string) {
	cfg, err := loadConfig()
	if err != nil {
		return "", ""
	}
	ep, err := normalizeEndpoint(cfg.Endpoint)
	if err != nil {
		return "", ""
	}
	return ep, strings.TrimSpace(cfg.EndpointKey)
}

// isRemoteMode reports whether inference runs on another machine. Used to
// skip the local engine/model requirements and to mark settings that the
// remote decides.
func isRemoteMode() bool {
	ep, _ := remoteEndpoint()
	return ep != ""
}

// endpointDisplay renders the endpoint setting, naming the machine doing the
// work so "local" and "remote" are never ambiguous in /set output.
func endpointDisplay(cfg Config) string {
	ep, err := normalizeEndpoint(cfg.Endpoint)
	if err != nil {
		return "invalid (" + strings.TrimSpace(cfg.Endpoint) + ")"
	}
	if ep == "" {
		return "local (inference runs on this machine)"
	}
	return ep
}

// remoteDecidesSetting reports whether a setting is fixed by the machine
// running the model rather than the one typing. These are all passed to
// llama-server at spawn, so a client cannot change them over HTTP.
func remoteDecidesSetting(key string) bool {
	switch strings.ToLower(strings.TrimSpace(key)) {
	case "ctx_size", "gpu_layers", "engine_variant",
		// The engine-tuning settings all bind at server launch, which a
		// remote did on its own machine. temperature is the exception:
		// it rides on each request, so it applies against a remote too.
		"kv_offload", "flash_attn", "cache_type_k", "cache_type_v",
		"threads", "batch_size", "ubatch_size", "parallel", "cache_reuse",
		"mmap", "mlock", "seed", "override_tensor", "context_shift":
		return true
	}
	return false
}

// effectiveCtxSize is the context the next request will actually get.
//
// In remote mode the server fixed it at spawn, and this install's ctx_size
// says nothing about it — a client at the default 16384 talking to a server
// serving 8192 would compute a max_tokens ceiling larger than the server's
// entire context, accept it, and fail at request time.
func effectiveCtxSize(cfg Config) int {
	if ep, _ := remoteEndpoint(); ep != "" {
		if st := getRemoteStatus(); st.HaveInfo && st.Info.CtxPerSlot > 0 {
			return st.Info.CtxPerSlot
		}
		// Connected to something that doesn't report its context. Nothing
		// better than the local value is available, so the ceiling stays
		// advisory rather than accurate.
	}
	return resolveCtxSize(cfg)
}
