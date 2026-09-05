package tui

import (
	"os"
	"strings"
	"testing"

	"atlas.llm/internal/config"
	"atlas.llm/internal/tools"
)

func TestValidProfileName(t *testing.T) {
	good := []string{"fast", "quality", "lan-server", "ctx_8k", "A1"}
	for _, n := range good {
		if err := config.ValidProfileName(n); err != nil {
			t.Errorf("%q should be valid: %v", n, err)
		}
	}
	bad := []string{"", "  ", "has space", "slash/name", "back\\slash",
		"dot.name", "..", "name!", "tooooooooooooooooooooooooooooooooooooooooolong"}
	for _, n := range bad {
		if err := config.ValidProfileName(n); err == nil {
			t.Errorf("%q should be rejected", n)
		}
	}
}

func TestProfileRoundTrip(t *testing.T) {
	name := "unittest-roundtrip"
	t.Cleanup(func() { _ = config.DeleteProfile(name) })

	temp := 0.7
	want := config.Config{
		CurrentModel: "qwen3.5-9b",
		CtxSize:      8192,
		MaxTokens:    2048,
		Reasoning:    "off",
		KVOffload:    "off",
		Temperature:  &temp,
		ToolsEnabled: true,
		AMAEnabled:   true,
	}
	if err := config.SaveProfile(name, want); err != nil {
		t.Fatalf("saveProfile: %v", err)
	}
	got, err := config.LoadProfile(name)
	if err != nil {
		t.Fatalf("loadProfile: %v", err)
	}
	if !config.ConfigsEqual(got, want) {
		t.Errorf("round-trip mismatch:\n got %+v\nwant %+v", got, want)
	}
}

// Saving a profile must not touch the live config.json — it is a snapshot,
// not a switch.
func TestSaveProfileLeavesActiveConfigAlone(t *testing.T) {
	name := "unittest-noclobber"
	t.Cleanup(func() { _ = config.DeleteProfile(name) })

	before, err := os.ReadFile(mustConfigPath(t))
	activeExisted := err == nil

	if err := config.SaveProfile(name, config.Config{CurrentModel: "gemma-3-1b-it", CtxSize: 4096}); err != nil {
		t.Fatalf("saveProfile: %v", err)
	}
	after, err := os.ReadFile(mustConfigPath(t))
	if activeExisted {
		if err != nil {
			t.Fatalf("config.json vanished after saveProfile: %v", err)
		}
		if string(before) != string(after) {
			t.Error("saveProfile modified the active config.json")
		}
	}
}

func TestListAndDeleteProfiles(t *testing.T) {
	names := []string{"unittest-a", "unittest-b"}
	for _, n := range names {
		if err := config.SaveProfile(n, config.Config{CurrentModel: "gemma-3-1b-it"}); err != nil {
			t.Fatal(err)
		}
	}
	t.Cleanup(func() {
		for _, n := range names {
			_ = config.DeleteProfile(n)
		}
	})

	got, err := config.ListProfiles()
	if err != nil {
		t.Fatalf("listProfiles: %v", err)
	}
	seen := map[string]bool{}
	for _, n := range got {
		seen[n] = true
	}
	for _, n := range names {
		if !seen[n] {
			t.Errorf("listProfiles missing %q; got %v", n, got)
		}
	}

	if err := config.DeleteProfile("unittest-a"); err != nil {
		t.Fatalf("deleteProfile: %v", err)
	}
	after, _ := config.ListProfiles()
	for _, n := range after {
		if n == "unittest-a" {
			t.Error("unittest-a still listed after delete")
		}
	}
	// Deleting a name that was never saved is an error the caller can show.
	if err := config.DeleteProfile("unittest-never-existed"); err == nil {
		t.Error("deleting a missing profile should error")
	}
}

// Profiles are piml files — the Atlas suite's format — not JSON.
func TestProfileFileIsPiml(t *testing.T) {
	name := "unittest-piml"
	t.Cleanup(func() { _ = config.DeleteProfile(name) })
	if err := config.SaveProfile(name, config.Config{CurrentModel: "gemma-3-1b-it", CtxSize: 4096}); err != nil {
		t.Fatal(err)
	}
	p, err := config.ProfilePath(name)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(p, ".piml") {
		t.Errorf("profile path %q is not a .piml file", p)
	}
	data, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "(current_model) gemma-3-1b-it") {
		t.Errorf("profile content is not piml:\n%s", data)
	}
	if strings.Contains(string(data), "{") {
		t.Errorf("profile content looks like JSON:\n%s", data)
	}
}

// --install-profiles ships the built-in presets; installing over an edited
// profile must never clobber it.
func TestInstallProfiles(t *testing.T) {
	withTempHome(t)
	var out strings.Builder
	if err := config.InstallProfiles(&out); err != nil {
		t.Fatalf("installProfiles: %v", err)
	}
	lite, err := config.LoadProfile("lite")
	if err != nil {
		t.Fatalf("lite preset not installed: %v", err)
	}
	if lite.CurrentModel != config.LightestModel().Name {
		t.Errorf("lite profile model = %q, want the lightest (%q)",
			lite.CurrentModel, config.LightestModel().Name)
	}
	tweet, err := config.LoadProfile("tweet150k")
	if err != nil {
		t.Fatalf("tweet150k preset not installed: %v", err)
	}
	if tweet.CtxSize != 150000 || tweet.CacheTypeK != "q4_0" ||
		tweet.CacheTypeV != "q4_0" || tweet.Parallel != 1 {
		t.Errorf("tweet150k preset content wrong: %+v", tweet)
	}
	if !strings.Contains(out.String(), "lite") || !strings.Contains(out.String(), "tweet150k") {
		t.Errorf("install output does not name the presets:\n%s", out.String())
	}

	// A user's edit survives a re-install.
	edited := lite
	edited.CtxSize = 12345
	if err := config.SaveProfile("lite", edited); err != nil {
		t.Fatal(err)
	}
	if err := config.InstallProfiles(&out); err != nil {
		t.Fatal(err)
	}
	after, err := config.LoadProfile("lite")
	if err != nil {
		t.Fatal(err)
	}
	if after.CtxSize != 12345 {
		t.Error("re-install clobbered an edited profile")
	}
}

// The preset catalog: every entry must carry a note and a valid name, point
// at a real registry model, and be shipped by --install-profiles. Presets
// with a machine-specific engine_variant would break portability, so the
// catalog must leave it on auto.
func TestPresetCatalog(t *testing.T) {
	withTempHome(t)
	want := []string{"lite", "tiny", "basic", "fast", "coder", "quality", "heretic", "current", "tweet150k"}
	if got := len(config.PresetProfiles()); got != len(want) {
		t.Fatalf("catalog has %d presets, want %d", got, len(want))
	}
	for _, p := range config.PresetProfiles() {
		if err := config.ValidProfileName(p.Name); err != nil {
			t.Errorf("preset name %q invalid: %v", p.Name, err)
		}
		if p.Note == "" {
			t.Errorf("preset %q has no note", p.Name)
		}
		cfg := p.Cfg()
		if _, ok := config.FindModel(cfg.CurrentModel); !ok {
			t.Errorf("preset %q names unknown model %q", p.Name, cfg.CurrentModel)
		}
		if cfg.EngineVariant != "" {
			t.Errorf("preset %q pins engine_variant=%q — presets must stay on auto", p.Name, cfg.EngineVariant)
		}
	}

	var out strings.Builder
	if err := config.InstallProfiles(&out); err != nil {
		t.Fatalf("installProfiles: %v", err)
	}
	names, err := config.ListProfiles()
	if err != nil {
		t.Fatal(err)
	}
	installed := map[string]bool{}
	for _, n := range names {
		installed[n] = true
	}
	for _, n := range want {
		if !installed[n] {
			t.Errorf("preset %q not installed; got %v", n, names)
		}
	}

	// Spot-check the recovered settings survived embedding.
	q, err := config.LoadProfile("quality")
	if err != nil {
		t.Fatal(err)
	}
	if q.CurrentModel != "qwen3.8-27b" || q.Reasoning != "on" || q.CtxSize != 32768 || !q.ToolsEnabled {
		t.Errorf("quality preset content wrong: %+v", q)
	}
	cur, err := config.LoadProfile("current")
	if err != nil {
		t.Fatal(err)
	}
	if cur.CtxSize != 65536 || cur.KVOffload != "off" || cur.MaxTokens != 16384 {
		t.Errorf("current preset content wrong: %+v", cur)
	}

	// The abliterated preset is quality's twin on the heretic model — same
	// window, reasoning, and tools, differing only in which weights answer.
	h, err := config.LoadProfile("heretic")
	if err != nil {
		t.Fatal(err)
	}
	if h.CurrentModel != "qwen3.8-27b-heretic" || h.Reasoning != "on" ||
		h.CtxSize != 32768 || !h.ToolsEnabled {
		t.Errorf("heretic preset content wrong: %+v", h)
	}
}

// The reset escape hatch stays on a gemma: the point of the lite preset is
// the lightest, most compatible model, and today that is the gemma family.
// If a lighter non-gemma entry ever lands, this forces a conscious choice.
func TestLiteProfileIsGemma(t *testing.T) {
	if got := config.LiteProfile().CurrentModel; !strings.HasPrefix(got, "gemma") {
		t.Errorf("lite profile model = %q, want a gemma", got)
	}
}

func TestConfigsEqual(t *testing.T) {
	a := config.Config{CurrentModel: "x", CtxSize: 8192, Reasoning: "off"}
	b := config.Config{CurrentModel: "x", CtxSize: 8192, Reasoning: "off"}
	if !config.ConfigsEqual(a, b) {
		t.Error("identical configs reported unequal")
	}
	b.CtxSize = 16384
	if config.ConfigsEqual(a, b) {
		t.Error("differing configs reported equal")
	}
	// Pointer fields compare by value, not identity.
	t1, t2 := 0.5, 0.5
	if !config.ConfigsEqual(config.Config{Temperature: &t1}, config.Config{Temperature: &t2}) {
		t.Error("equal pointer values reported unequal")
	}
}

// Loading a profile must overwrite the active config and re-sync the session
// state that mirrors it (tools, ama). This drives the real /config load path.
func TestConfigLoadAppliesProfile(t *testing.T) {
	// This overwrites the live config.json — back it up and restore.
	p := mustConfigPath(t)
	backup, backupErr := os.ReadFile(p)
	t.Cleanup(func() {
		if backupErr == nil {
			_ = os.WriteFile(p, backup, 0644)
		}
	})
	prevAma := tools.AmaOn.Load()
	t.Cleanup(func() { tools.AmaOn.Store(prevAma) })

	prof := "unittest-load"
	t.Cleanup(func() { _ = config.DeleteProfile(prof) })
	if err := config.SaveProfile(prof, config.Config{CurrentModel: "gemma-3-1b-it",
		CtxSize: 4096, MaxTokens: 2048, ToolsEnabled: false, AMAEnabled: false}); err != nil {
		t.Fatal(err)
	}

	m := newChatModel()
	m.agentEnabled = true
	tools.AmaOn.Store(true)
	m.handleConfigLoad(prof)

	got, err := config.LoadConfig()
	if err != nil {
		t.Fatal(err)
	}
	if got.CurrentModel != "gemma-3-1b-it" || got.CtxSize != 4096 {
		t.Errorf("profile not applied to active config: %+v", got)
	}
	if m.agentEnabled {
		t.Error("agentEnabled should be false after loading a tools-off profile")
	}
	if tools.AmaOn.Load() {
		t.Error("amaOn should be false after loading an ama-off profile")
	}
}

// /config show renders a profile's settings without loading it — the values
// must reflect the profile, and the active marker only when it matches.
func TestRenderProfile(t *testing.T) {
	cfg := config.Config{CurrentModel: "qwen3.5-9b", CtxSize: 8192, Reasoning: "off",
		ToolsEnabled: true, AMAEnabled: false}
	out := renderProfile("fast", cfg, false)
	for _, want := range []string{"fast", "qwen3.5-9b", "8192", "ctx_size", "tools"} {
		if !strings.Contains(out, want) {
			t.Errorf("renderProfile output missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "active") {
		t.Error("inactive profile should not be marked active")
	}
	if !strings.Contains(renderProfile("fast", cfg, true), "active") {
		t.Error("active profile should be marked active")
	}
}

func mustConfigPath(t *testing.T) string {
	t.Helper()
	p, err := config.ConfigPath()
	if err != nil {
		t.Fatalf("configPath: %v", err)
	}
	return p
}
