package main

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
)

type Model struct {
	Name     string `json:"name"`
	Filename string `json:"filename"`
	URL      string `json:"url"`
	Size     string `json:"size"`
}

var availableModels = []Model{
	{
		Name:     "gemma-4-e2b-it",
		Filename: "gemma-4-E2B-it-Q4_K_M.gguf",
		URL:      "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf",
		Size:     "~2.9GB",
	},
}

const defaultModel = "gemma-4-e2b-it"

// llamacppLatestURL is the GitHub API endpoint that always returns the latest
// ggml-org/llama.cpp release. We resolve it at download time to pick the
// correct prebuilt archive for the current OS/arch.
const llamacppLatestURL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

// llamacppAssetSuffix maps GOOS/GOARCH to the suffix of the release asset
// filename we want. Assets are named like `llama-b8892-bin-win-cpu-x64.zip`;
// we match against the tail so the build tag can vary.
var llamacppAssetSuffix = map[string]string{
	"windows/amd64": "win-cpu-x64.zip",
	"windows/arm64": "win-cpu-arm64.zip",
	"darwin/amd64":  "macos-x64.tar.gz",
	"darwin/arm64":  "macos-arm64.tar.gz",
	"linux/amd64":   "ubuntu-x64.tar.gz",
	"linux/arm64":   "ubuntu-arm64.tar.gz",
}

type Config struct {
	CurrentModel string `json:"current_model"`
}

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
	dir, err := engineDir()
	if err != nil {
		return "", err
	}
	target := "llama-cli"
	if runtime.GOOS == "windows" {
		target = "llama-cli.exe"
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
