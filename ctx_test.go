package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The GGUF header parser must read the trained context length from a real
// model file, since /set ctx_size validates against it.
func TestModelTrainedContextFromRealGGUF(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip(err)
	}
	dir := filepath.Join(home, ".atlas", "atlas.llm.data", "models")
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Skip("no models downloaded")
	}
	found := 0
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".gguf") {
			continue
		}
		n, err := modelTrainedContext(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Errorf("%s: %v", e.Name(), err)
			continue
		}
		if n < 2048 || n > 1<<22 {
			t.Errorf("%s: implausible context length %d", e.Name(), n)
		}
		t.Logf("%-42s trained context = %d", e.Name(), n)
		found++
	}
	if found == 0 {
		t.Skip("no .gguf files to read")
	}
}

func TestModelTrainedContextRejectsNonGGUF(t *testing.T) {
	p := filepath.Join(t.TempDir(), "x.gguf")
	if err := os.WriteFile(p, []byte("not a gguf file at all"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := modelTrainedContext(p); err == nil {
		t.Error("expected an error for a non-GGUF file")
	}
	if _, err := modelTrainedContext(filepath.Join(t.TempDir(), "missing.gguf")); err == nil {
		t.Error("expected an error for a missing file")
	}
}

func TestResolveCtxSize(t *testing.T) {
	withTempHome(t)
	if got := resolveCtxSize(Config{}); got != defaultCtxSize {
		t.Errorf("default ctx = %d, want %d", got, defaultCtxSize)
	}
	if got := resolveCtxSize(Config{CtxSize: 32768}); got != 32768 {
		t.Errorf("explicit ctx = %d, want 32768", got)
	}
	// Above atlas.llm's own ceiling.
	if got := resolveCtxSize(Config{CtxSize: 1 << 20}); got != maxConfigurableCtx {
		t.Errorf("oversized ctx = %d, want clamped to %d", got, maxConfigurableCtx)
	}
	// Below the floor.
	if got := resolveCtxSize(Config{CtxSize: 128}); got != minConfigurableCtx {
		t.Errorf("undersized ctx = %d, want clamped to %d", got, minConfigurableCtx)
	}
}

// 150K-scale contexts are real on 12GB cards now — q4_0 KV plus hybrid
// attention keeps the cache affordable — so the ceiling must not clamp them.
func TestResolveCtxSizeAllows150K(t *testing.T) {
	withTempHome(t)
	if got := resolveCtxSize(Config{CtxSize: 150000}); got != 150000 {
		t.Errorf("150000 ctx = %d, want 150000 (the ceiling clamps a value real hardware can serve)", got)
	}
}

// max_tokens must track ctx_size — a fixed ceiling would either forbid legal
// values on a big window or permit impossible ones on a small window.
func TestMaxTokensCeilingTracksCtx(t *testing.T) {
	withTempHome(t)
	small := maxTokensCeiling(Config{CtxSize: 8192})
	big := maxTokensCeiling(Config{CtxSize: 65536})
	if big <= small {
		t.Errorf("ceiling did not grow with ctx_size: %d vs %d", small, big)
	}
	if small >= 8192 {
		t.Errorf("ceiling %d leaves no room for prompt/history in an 8192 window", small)
	}
	if got := maxTokensCeiling(Config{CtxSize: minConfigurableCtx}); got < 512 {
		t.Errorf("ceiling %d is below the floor", got)
	}
}

func TestCtxSizeRoundTrips(t *testing.T) {
	withTempHome(t)
	if err := saveConfig(Config{CurrentModel: defaultModel, CtxSize: 32768}); err != nil {
		t.Fatal(err)
	}
	got, err := loadConfig()
	if err != nil {
		t.Fatal(err)
	}
	if got.CtxSize != 32768 {
		t.Errorf("ctx_size = %d, want 32768", got.CtxSize)
	}
	// Unset must stay 0 (auto), not be written as an explicit value.
	if err := saveConfig(Config{CurrentModel: defaultModel}); err != nil {
		t.Fatal(err)
	}
	got, _ = loadConfig()
	if got.CtxSize != 0 {
		t.Errorf("unset ctx_size = %d, want 0", got.CtxSize)
	}
}
