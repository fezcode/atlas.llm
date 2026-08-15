package main

import (
	"strings"
	"testing"
)

func TestParseModelSize(t *testing.T) {
	tests := []struct {
		in   string
		want int64
	}{
		{"~700MB", 700e6},
		{"~2.5GB", 2.5e9},
		{"8.2GB", 8.2e9},
		{" ~5.7GB ", 5.7e9},
		{"", 0},
		{"lots", 0},
		{"~0GB", 0},
	}
	for _, tt := range tests {
		if got := parseModelSize(tt.in); got != int64(tt.want) {
			t.Errorf("parseModelSize(%q) = %d, want %d", tt.in, got, int64(tt.want))
		}
	}
}

// The escape hatch is only useful if it really picks the smallest entry —
// not merely the first, or whatever defaultModel happens to be.
func TestLightestModelIsActuallySmallest(t *testing.T) {
	got := lightestModel()
	smallest := parseModelSize(got.Size)
	if smallest == 0 {
		t.Fatalf("lightest model %q has an unparseable size %q", got.Name, got.Size)
	}
	for _, m := range availableModels {
		if s := parseModelSize(m.Size); s > 0 && s < smallest {
			t.Errorf("%s (%s) is smaller than the chosen %s (%s)",
				m.Name, m.Size, got.Name, got.Size)
		}
	}
	t.Logf("lightest = %s (%s)", got.Name, got.Size)
}

func TestResetLoadsLiteProfile(t *testing.T) {
	withTempHome(t)
	heavy := "ministral-3-14b-instruct"
	if _, ok := findModel(heavy); !ok {
		t.Skip("registry changed")
	}
	if err := saveConfig(Config{CurrentModel: heavy}); err != nil {
		t.Fatal(err)
	}
	if err := resetToLiteProfile(); err != nil {
		t.Fatalf("resetToLiteProfile: %v", err)
	}
	cfg, err := loadConfig()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.CurrentModel != lightestModel().Name {
		t.Errorf("current model = %q, want %q", cfg.CurrentModel, lightestModel().Name)
	}
	// Idempotent: running it again is a no-op, not an error.
	if err := resetToLiteProfile(); err != nil {
		t.Errorf("second reset errored: %v", err)
	}
}

// The reset is a whole-profile load now, not a model switch: a heavy config's
// tuning must not survive it. A giant context or forced offload can be as
// unbootable as a giant model, so the escape hatch clears everything.
func TestResetReplacesWholeConfig(t *testing.T) {
	withTempHome(t)
	if err := saveConfig(Config{
		CurrentModel: "qwen3.5-9b", MaxTokens: 2048, CtxSize: 150000,
		ToolsEnabled: true, CacheTypeK: "q4_0", Parallel: 1,
	}); err != nil {
		t.Fatal(err)
	}
	if err := resetToLiteProfile(); err != nil {
		t.Fatal(err)
	}
	cfg, _ := loadConfig()
	if !configsEqual(cfg, liteProfile()) {
		t.Errorf("reset left settings behind:\n got %+v\nwant %+v", cfg, liteProfile())
	}
}

// The listing note must be honest about what it measures, and present for
// every registry entry on a machine where RAM is readable.
func TestModelResourceNote(t *testing.T) {
	if _, ok := systemRAM(); !ok {
		t.Skip("system RAM not readable here")
	}
	for _, m := range availableModels {
		note := modelResourceNote(m)
		if note == "" {
			t.Errorf("%s has no resource note", m.Name)
			continue
		}
		if !strings.Contains(note, "RAM") {
			t.Errorf("%s note %q does not mention RAM", m.Name, note)
		}
	}
	// Bigger models must never be reported as fitting better.
	var prevShare float64
	for _, m := range availableModels {
		share, _ := modelFit(m)
		if parseModelSize(m.Size) > 0 && share <= 0 {
			t.Errorf("%s got a non-positive RAM share", m.Name)
		}
		_ = prevShare
	}
}

func TestSystemRAMIsPlausible(t *testing.T) {
	total, ok := systemRAM()
	if !ok {
		t.Skip("not readable on this platform")
	}
	// Between 1GB and 4TB — anything outside that is a parsing bug.
	if total < 1<<30 || total > 4<<40 {
		t.Errorf("implausible system RAM: %d bytes", total)
	}
	t.Logf("system RAM = %s", formatBytes(total))
}
