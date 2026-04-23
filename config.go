package main

import (
	"encoding/json"
	"fmt"
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
		Filename: "gemma-4-e2b-it-Q4_K_M.gguf",
		URL:      "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-e2b-it-Q4_K_M.gguf",
		Size:     "~1.7GB",
	},
}

const (
	llamafileURL = "https://github.com/Mozilla-Ocho/llamafile/releases/download/0.10.0/llamafile-0.10.0"
	defaultModel = "gemma-4-e2b-it"
)

type Config struct {
	CurrentModel string `json:"current_model"`
}

func atlasDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(home, ".atlas", "atlas.ai.data")
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

func enginePath() (string, error) {
	base, err := atlasDir()
	if err != nil {
		return "", err
	}
	name := "llamafile"
	if runtime.GOOS == "windows" {
		name = "llamafile.exe"
	}
	return filepath.Join(base, name), nil
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
	p, err := enginePath()
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
