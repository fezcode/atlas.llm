package engine

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"atlas.llm/internal/config"
)

// engineDir is the directory where the extracted llama.cpp binaries live.
func EngineDir() (string, error) {
	base, err := config.AtlasDir()
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
	dir, err := EngineDir()
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

func IsEngineDownloaded() bool {
	p, err := findEngineBinary()
	if err != nil {
		return false
	}
	info, err := os.Stat(p)
	return err == nil && !info.IsDir() && info.Size() > 0
}

// engineVariantDisplay renders the engine_variant setting, showing what
// "auto" resolves to on this platform.
func EngineVariantDisplay(cfg config.Config) string {
	resolved := ResolveEngineVariant(cfg.EngineVariant)
	if strings.TrimSpace(cfg.EngineVariant) == "" ||
		strings.EqualFold(cfg.EngineVariant, EngineVariantAuto) {
		if runtime.GOOS == "darwin" {
			return "auto (" + resolved + ", Metal built in)"
		}
		return "auto (" + resolved + ")"
	}
	return resolved
}
