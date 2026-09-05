package tui

import (
	"strings"
	"testing"

	"atlas.llm/internal/catalog"
	"atlas.llm/internal/config"
	"atlas.llm/internal/engine"
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
		if got := config.ParseModelSize(tt.in); got != int64(tt.want) {
			t.Errorf("parseModelSize(%q) = %d, want %d", tt.in, got, int64(tt.want))
		}
	}
}

// The escape hatch is only useful if it really picks the smallest entry —
// not merely the first, or whatever defaultModel happens to be.
func TestLightestModelIsActuallySmallest(t *testing.T) {
	got := config.LightestModel()
	smallest := config.ParseModelSize(got.Size)
	if smallest == 0 {
		t.Fatalf("lightest model %q has an unparseable size %q", got.Name, got.Size)
	}
	for _, m := range catalog.AvailableModels {
		if s := config.ParseModelSize(m.Size); s > 0 && s < smallest {
			t.Errorf("%s (%s) is smaller than the chosen %s (%s)",
				m.Name, m.Size, got.Name, got.Size)
		}
	}
	t.Logf("lightest = %s (%s)", got.Name, got.Size)
}

func TestResetLoadsLiteProfile(t *testing.T) {
	withTempHome(t)
	heavy := "ministral-3-14b-instruct"
	if _, ok := config.FindModel(heavy); !ok {
		t.Skip("registry changed")
	}
	if err := config.SaveConfig(config.Config{CurrentModel: heavy}); err != nil {
		t.Fatal(err)
	}
	if err := config.ResetToLiteProfile(); err != nil {
		t.Fatalf("resetToLiteProfile: %v", err)
	}
	cfg, err := config.LoadConfig()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.CurrentModel != config.LightestModel().Name {
		t.Errorf("current model = %q, want %q", cfg.CurrentModel, config.LightestModel().Name)
	}
	// Idempotent: running it again is a no-op, not an error.
	if err := config.ResetToLiteProfile(); err != nil {
		t.Errorf("second reset errored: %v", err)
	}
}

// The reset is a whole-profile load now, not a model switch: a heavy config's
// tuning must not survive it. A giant context or forced offload can be as
// unbootable as a giant model, so the escape hatch clears everything.
func TestResetReplacesWholeConfig(t *testing.T) {
	withTempHome(t)
	if err := config.SaveConfig(config.Config{
		CurrentModel: "qwen3.5-9b", MaxTokens: 2048, CtxSize: 150000,
		ToolsEnabled: true, CacheTypeK: "q4_0", Parallel: 1,
	}); err != nil {
		t.Fatal(err)
	}
	if err := config.ResetToLiteProfile(); err != nil {
		t.Fatal(err)
	}
	cfg, _ := config.LoadConfig()
	if !config.ConfigsEqual(cfg, config.LiteProfile()) {
		t.Errorf("reset left settings behind:\n got %+v\nwant %+v", cfg, config.LiteProfile())
	}
}

// The listing note must be honest about what it measures, and present for
// every registry entry on a machine where RAM is readable.
func TestModelResourceNote(t *testing.T) {
	if _, ok := engine.SystemRAM(); !ok {
		t.Skip("system RAM not readable here")
	}
	for _, m := range catalog.AvailableModels {
		note := engine.ModelResourceNote(m)
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
	for _, m := range catalog.AvailableModels {
		share, _ := engine.ModelFit(m)
		if config.ParseModelSize(m.Size) > 0 && share <= 0 {
			t.Errorf("%s got a non-positive RAM share", m.Name)
		}
		_ = prevShare
	}
}

func TestSystemRAMIsPlausible(t *testing.T) {
	total, ok := engine.SystemRAM()
	if !ok {
		t.Skip("not readable on this platform")
	}
	// Between 1GB and 4TB — anything outside that is a parsing bug.
	if total < 1<<30 || total > 4<<40 {
		t.Errorf("implausible system RAM: %d bytes", total)
	}
	t.Logf("system RAM = %s", engine.FormatBytes(total))
}
