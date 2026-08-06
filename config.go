package main

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
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
// Note the CUDA entries: `cudart-llama-bin-win-cuda-12.4-x64.zip` and
// `llama-b10280-bin-win-cuda-12.4-x64.zip` share a suffix, which is why
// asset matching also requires the `llama-` / `cudart-` prefix rather than
// matching on the tail alone.
var llamacppAssetSuffix = map[string]map[string]engineAsset{
	engineVariantCPU: {
		"windows/amd64": {Suffix: "win-cpu-x64.zip", Size: "~30MB"},
		"windows/arm64": {Suffix: "win-cpu-arm64.zip", Size: "~30MB"},
		"darwin/amd64":  {Suffix: "macos-x64.tar.gz", Size: "~15MB"},
		"darwin/arm64":  {Suffix: "macos-arm64.tar.gz", Size: "~11MB"},
		"linux/amd64":   {Suffix: "ubuntu-x64.tar.gz", Size: "~30MB"},
		"linux/arm64":   {Suffix: "ubuntu-arm64.tar.gz", Size: "~30MB"},
	},
	engineVariantVulkan: {
		"windows/amd64": {Suffix: "win-vulkan-x64.zip", Size: "~35MB"},
		"linux/amd64":   {Suffix: "ubuntu-vulkan-x64.tar.gz", Size: "~35MB"},
		"linux/arm64":   {Suffix: "ubuntu-vulkan-arm64.tar.gz", Size: "~35MB"},
	},
	engineVariantCUDA: {
		"windows/amd64": {
			Suffix:    "win-cuda-12.4-x64.zip",
			Companion: "cudart-llama-bin-win-cuda-12.4-x64.zip",
			Size:      "~640MB (engine + CUDA runtime)",
		},
	},
	engineVariantHIP: {
		"windows/amd64": {Suffix: "win-hip-radeon-x64.zip", Size: "~325MB"},
	},
}

// engineVariantNames lists the selectable variants for this platform, for
// help text and `/set` validation messages.
func engineVariantNames() []string {
	key := runtime.GOOS + "/" + runtime.GOARCH
	out := []string{engineVariantAuto}
	for _, v := range []string{engineVariantCPU, engineVariantVulkan, engineVariantCUDA, engineVariantHIP} {
		if _, ok := llamacppAssetSuffix[v][key]; ok {
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

// resolveEngineVariant turns "auto"/"" into a concrete variant. macOS gets
// Metal from the standard archive, so auto stays on "cpu" there; elsewhere
// auto also means "cpu" because a Vulkan build is useless without a Vulkan
// driver and we can't reliably detect one.
func resolveEngineVariant(v string) string {
	name := strings.ToLower(strings.TrimSpace(v))
	if !engineVariantIsGPU(name) {
		return engineVariantCPU
	}
	// A GPU variant with no build for this platform would fail at download
	// time; fall back rather than persisting an unusable selection.
	if _, ok := llamacppAssetSuffix[name][runtime.GOOS+"/"+runtime.GOARCH]; !ok {
		return engineVariantCPU
	}
	return name
}

// engineAssetSuffix returns the release-asset suffix for a variant on this
// platform.
func engineAssetSuffix(variant string) (engineAsset, error) {
	key := runtime.GOOS + "/" + runtime.GOARCH
	byPlatform, ok := llamacppAssetSuffix[variant]
	if !ok {
		return engineAsset{}, fmt.Errorf("unknown engine variant %q (expected %s)",
			variant, strings.Join(engineVariantNames(), ", "))
	}
	asset, ok := byPlatform[key]
	if !ok {
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
// Windows/Linux only a GPU-enabled engine variant can use it.
func autoGPULayers(variant string) int {
	// macOS archives always carry Metal, so offloading is free there.
	// Elsewhere it needs a GPU-enabled engine build.
	if runtime.GOOS == "darwin" || engineVariantIsGPU(resolveEngineVariant(variant)) {
		return maxGPULayers
	}
	return 0
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

type Config struct {
	CurrentModel string `json:"current_model"`
	MaxTokens    int    `json:"max_tokens,omitempty"`
	ToolsEnabled bool   `json:"tools_enabled,omitempty"`

	// GPULayers is the -ngl value handed to llama-server. nil means "auto"
	// — a pointer rather than an int so an explicit 0 (force CPU) is
	// distinguishable from an absent setting.
	GPULayers *int `json:"gpu_layers,omitempty"`

	// EngineVariant selects which llama.cpp release archive to download:
	// "cpu" (default) or "vulkan". Empty means auto.
	EngineVariant string `json:"engine_variant,omitempty"`

	// MaxToolRounds caps tool-call rounds per message. 0 means the default;
	// a negative value means no cap.
	MaxToolRounds int `json:"max_tool_rounds,omitempty"`

	// CtxSize is the context window llama-server is started with (-c).
	// 0 means the default. The ceiling is whatever the model was trained
	// for, which is read from its GGUF metadata.
	CtxSize int `json:"ctx_size,omitempty"`
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
// larger trained context. Qwen3.5 advertises 262144, whose KV cache would
// be tens of gigabytes — allowing it silently would just OOM the machine.
const maxConfigurableCtx = 131072

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
	n := resolveCtxSize(cfg) * 3 / 4
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

	switch {
	case runtime.GOOS == "darwin":
		rows = append(rows, [2]string{"", "Metal is built into the macOS engine — GPU is on by default"})
	case engineVariantIsGPU(installed):
		rows = append(rows, [2]string{"", fmt.Sprintf("engine: %s build installed — GPU offload active", installed)})
	default:
		opts := engineVariantNames()
		if len(opts) > 2 {
			rows = append(rows,
				[2]string{"/set engine_variant", "GPU builds here: " + strings.Join(opts[2:], ", ")},
				[2]string{"", "then /download engine to install it (CPU-only until you do)"},
			)
		} else {
			rows = append(rows, [2]string{"", "no GPU llama.cpp build published for this platform"})
		}
	}
	rows = append(rows, [2]string{"/set", fmt.Sprintf("current: gpu_layers=%s, engine=%s",
		gpuLayersDisplay(cfg), installed)})
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

// resetToLightestModel switches the configured model to the smallest one.
//
// atlas.llm warms the model server at startup, so quitting while a large
// model is selected means the next launch blocks loading it again — with no
// chance to reach /model from inside the TUI. This is the way out, from the
// command line, without touching config.json by hand.
func resetToLightestModel() error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	target := lightestModel()
	if cfg.CurrentModel == target.Name {
		fmt.Printf("Already using %s (%s), the lightest model available.\n", target.Name, target.Size)
		return nil
	}
	previous := cfg.CurrentModel
	cfg.CurrentModel = target.Name
	if err := saveConfig(cfg); err != nil {
		return err
	}
	fmt.Printf("Model reset: %s -> %s (%s)\n", previous, target.Name, target.Size)
	if !isModelDownloaded(target) {
		fmt.Printf("Note: %s is not downloaded yet — run /download %s inside chat.\n",
			target.Name, target.Name)
	}
	return nil
}
