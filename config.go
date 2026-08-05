package main

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
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
		Name:     "gemma-4-e2b-it",
		Filename: "gemma-4-E2B-it-Q4_K_M.gguf",
		URL:      "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf",
		Size:     "~2.9GB",
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
)

// llamacppAssetSuffix maps variant -> GOOS/GOARCH -> the suffix of the
// release asset filename we want. Assets are named like
// `llama-b8892-bin-win-cpu-x64.zip`; we match against the tail so the build
// tag can vary.
var llamacppAssetSuffix = map[string]map[string]string{
	engineVariantCPU: {
		"windows/amd64": "win-cpu-x64.zip",
		"windows/arm64": "win-cpu-arm64.zip",
		"darwin/amd64":  "macos-x64.tar.gz",
		"darwin/arm64":  "macos-arm64.tar.gz",
		"linux/amd64":   "ubuntu-x64.tar.gz",
		"linux/arm64":   "ubuntu-arm64.tar.gz",
	},
	engineVariantVulkan: {
		"windows/amd64": "win-vulkan-x64.zip",
		"linux/amd64":   "ubuntu-vulkan-x64.tar.gz",
		"linux/arm64":   "ubuntu-vulkan-arm64.tar.gz",
	},
}

// resolveEngineVariant turns "auto"/"" into a concrete variant. macOS gets
// Metal from the standard archive, so auto stays on "cpu" there; elsewhere
// auto also means "cpu" because a Vulkan build is useless without a Vulkan
// driver and we can't reliably detect one.
func resolveEngineVariant(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case engineVariantVulkan:
		return engineVariantVulkan
	default:
		return engineVariantCPU
	}
}

// engineAssetSuffix returns the release-asset suffix for a variant on this
// platform.
func engineAssetSuffix(variant string) (string, error) {
	key := runtime.GOOS + "/" + runtime.GOARCH
	byPlatform, ok := llamacppAssetSuffix[variant]
	if !ok {
		return "", fmt.Errorf("unknown engine variant %q", variant)
	}
	suffix, ok := byPlatform[key]
	if !ok {
		if variant != engineVariantCPU {
			return "", fmt.Errorf("no %s llama.cpp build available for %s", variant, key)
		}
		return "", fmt.Errorf("no llama.cpp prebuilt available for %s", key)
	}
	return suffix, nil
}

// maxGPULayers is the "offload everything" sentinel. llama.cpp clamps it to
// the model's actual layer count, so it doesn't need to be exact.
const maxGPULayers = 999

// autoGPULayers decides the default -ngl when the user hasn't set one.
// macOS builds always carry Metal, so offloading is free there. On
// Windows/Linux only a GPU-enabled engine variant can use it.
func autoGPULayers(variant string) int {
	if runtime.GOOS == "darwin" || resolveEngineVariant(variant) == engineVariantVulkan {
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
