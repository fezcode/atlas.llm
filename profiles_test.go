package main

import (
	"os"
	"strings"
	"testing"
)

func TestValidProfileName(t *testing.T) {
	good := []string{"fast", "quality", "lan-server", "ctx_8k", "A1"}
	for _, n := range good {
		if err := validProfileName(n); err != nil {
			t.Errorf("%q should be valid: %v", n, err)
		}
	}
	bad := []string{"", "  ", "has space", "slash/name", "back\\slash",
		"dot.name", "..", "name!", "tooooooooooooooooooooooooooooooooooooooooolong"}
	for _, n := range bad {
		if err := validProfileName(n); err == nil {
			t.Errorf("%q should be rejected", n)
		}
	}
}

func TestProfileRoundTrip(t *testing.T) {
	name := "unittest-roundtrip"
	t.Cleanup(func() { _ = deleteProfile(name) })

	temp := 0.7
	want := Config{
		CurrentModel: "qwen3.5-9b",
		CtxSize:      8192,
		MaxTokens:    2048,
		Reasoning:    "off",
		KVOffload:    "off",
		Temperature:  &temp,
		ToolsEnabled: true,
		AMAEnabled:   true,
	}
	if err := saveProfile(name, want); err != nil {
		t.Fatalf("saveProfile: %v", err)
	}
	got, err := loadProfile(name)
	if err != nil {
		t.Fatalf("loadProfile: %v", err)
	}
	if !configsEqual(got, want) {
		t.Errorf("round-trip mismatch:\n got %+v\nwant %+v", got, want)
	}
}

// Saving a profile must not touch the live config.json — it is a snapshot,
// not a switch.
func TestSaveProfileLeavesActiveConfigAlone(t *testing.T) {
	name := "unittest-noclobber"
	t.Cleanup(func() { _ = deleteProfile(name) })

	before, err := os.ReadFile(mustConfigPath(t))
	activeExisted := err == nil

	if err := saveProfile(name, Config{CurrentModel: "gemma-3-1b-it", CtxSize: 4096}); err != nil {
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
		if err := saveProfile(n, Config{CurrentModel: "gemma-3-1b-it"}); err != nil {
			t.Fatal(err)
		}
	}
	t.Cleanup(func() {
		for _, n := range names {
			_ = deleteProfile(n)
		}
	})

	got, err := listProfiles()
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

	if err := deleteProfile("unittest-a"); err != nil {
		t.Fatalf("deleteProfile: %v", err)
	}
	after, _ := listProfiles()
	for _, n := range after {
		if n == "unittest-a" {
			t.Error("unittest-a still listed after delete")
		}
	}
	// Deleting a name that was never saved is an error the caller can show.
	if err := deleteProfile("unittest-never-existed"); err == nil {
		t.Error("deleting a missing profile should error")
	}
}

// Profiles are piml files — the Atlas suite's format — not JSON.
func TestProfileFileIsPiml(t *testing.T) {
	name := "unittest-piml"
	t.Cleanup(func() { _ = deleteProfile(name) })
	if err := saveProfile(name, Config{CurrentModel: "gemma-3-1b-it", CtxSize: 4096}); err != nil {
		t.Fatal(err)
	}
	p, err := profilePath(name)
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
	if err := installProfiles(&out); err != nil {
		t.Fatalf("installProfiles: %v", err)
	}
	lite, err := loadProfile("lite")
	if err != nil {
		t.Fatalf("lite preset not installed: %v", err)
	}
	if lite.CurrentModel != lightestModel().Name {
		t.Errorf("lite profile model = %q, want the lightest (%q)",
			lite.CurrentModel, lightestModel().Name)
	}
	tweet, err := loadProfile("tweet150k")
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
	if err := saveProfile("lite", edited); err != nil {
		t.Fatal(err)
	}
	if err := installProfiles(&out); err != nil {
		t.Fatal(err)
	}
	after, err := loadProfile("lite")
	if err != nil {
		t.Fatal(err)
	}
	if after.CtxSize != 12345 {
		t.Error("re-install clobbered an edited profile")
	}
}

// The reset escape hatch stays on a gemma: the point of the lite preset is
// the lightest, most compatible model, and today that is the gemma family.
// If a lighter non-gemma entry ever lands, this forces a conscious choice.
func TestLiteProfileIsGemma(t *testing.T) {
	if got := liteProfile().CurrentModel; !strings.HasPrefix(got, "gemma") {
		t.Errorf("lite profile model = %q, want a gemma", got)
	}
}

func TestConfigsEqual(t *testing.T) {
	a := Config{CurrentModel: "x", CtxSize: 8192, Reasoning: "off"}
	b := Config{CurrentModel: "x", CtxSize: 8192, Reasoning: "off"}
	if !configsEqual(a, b) {
		t.Error("identical configs reported unequal")
	}
	b.CtxSize = 16384
	if configsEqual(a, b) {
		t.Error("differing configs reported equal")
	}
	// Pointer fields compare by value, not identity.
	t1, t2 := 0.5, 0.5
	if !configsEqual(Config{Temperature: &t1}, Config{Temperature: &t2}) {
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
	prevAma := amaOn.Load()
	t.Cleanup(func() { amaOn.Store(prevAma) })

	prof := "unittest-load"
	t.Cleanup(func() { _ = deleteProfile(prof) })
	if err := saveProfile(prof, Config{CurrentModel: "gemma-3-1b-it",
		CtxSize: 4096, MaxTokens: 2048, ToolsEnabled: false, AMAEnabled: false}); err != nil {
		t.Fatal(err)
	}

	m := newChatModel()
	m.agentEnabled = true
	amaOn.Store(true)
	m.handleConfigLoad(prof)

	got, err := loadConfig()
	if err != nil {
		t.Fatal(err)
	}
	if got.CurrentModel != "gemma-3-1b-it" || got.CtxSize != 4096 {
		t.Errorf("profile not applied to active config: %+v", got)
	}
	if m.agentEnabled {
		t.Error("agentEnabled should be false after loading a tools-off profile")
	}
	if amaOn.Load() {
		t.Error("amaOn should be false after loading an ama-off profile")
	}
}

// /config show renders a profile's settings without loading it — the values
// must reflect the profile, and the active marker only when it matches.
func TestRenderProfile(t *testing.T) {
	cfg := Config{CurrentModel: "qwen3.5-9b", CtxSize: 8192, Reasoning: "off",
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
	p, err := configPath()
	if err != nil {
		t.Fatalf("configPath: %v", err)
	}
	return p
}
