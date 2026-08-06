package main

import (
	"os"
	"strings"
	"testing"
)

func osGetpid() int { return os.Getpid() }

// The registry is the single source of truth for /set, /config, and
// /help set — every key must be complete and self-consistent.
func TestSettingsRegistryIsWellFormed(t *testing.T) {
	withTempHome(t)
	cfg, _ := loadConfig()
	seen := map[string]bool{}
	for _, s := range settingsRegistry {
		if s.Key == "" || s.Summary == "" || s.Usage == "" {
			t.Errorf("incomplete setting: %+v", s)
			continue
		}
		if seen[s.Key] {
			t.Errorf("duplicate setting key %q", s.Key)
		}
		seen[s.Key] = true
		if !strings.HasPrefix(s.Usage, "/set "+s.Key) {
			t.Errorf("%s: usage %q should start with /set %s", s.Key, s.Usage, s.Key)
		}
		if s.Value == nil || s.Value(cfg) == "" {
			t.Errorf("%s: Value renders empty", s.Key)
		}
		if s.Detail == nil || len(s.Detail(cfg)) < 40 {
			t.Errorf("%s: Detail is missing or too thin to help", s.Key)
		}
	}
}

// Every /set key must also be documented as a /help set subcommand, or the
// two drift apart.
func TestEverySettingIsDocumentedInHelp(t *testing.T) {
	tp, ok := findHelpTopic("set")
	if !ok {
		t.Fatal("no help topic for /set")
	}
	for _, key := range settingKeys() {
		if _, ok := tp.findSub(key); !ok {
			t.Errorf("setting %q has no /help set %s entry", key, key)
		}
	}
}

func TestFindSetting(t *testing.T) {
	for _, k := range []string{"ctx_size", "CTX_SIZE", " ctx_size "} {
		if _, ok := findSetting(k); !ok {
			t.Errorf("findSetting(%q) failed", k)
		}
	}
	if _, ok := findSetting("nope"); ok {
		t.Error("unknown key resolved")
	}
}

// `/set <key>` must explain, not just echo — that was the whole point.
func TestSettingDetailIncludesGuidanceAndUsage(t *testing.T) {
	withTempHome(t)
	cfg, _ := loadConfig()
	for _, s := range settingsRegistry {
		out := renderSettingDetail(s, cfg)
		if !strings.Contains(out, s.Key+" = ") {
			t.Errorf("%s: detail omits the current value", s.Key)
		}
		if !strings.Contains(out, "usage: "+s.Usage) {
			t.Errorf("%s: detail omits usage", s.Key)
		}
		if !strings.Contains(out, "/help set "+s.Key) {
			t.Errorf("%s: detail omits the pointer to full help", s.Key)
		}
	}
}

func TestRenderConfigCoversEverySection(t *testing.T) {
	withTempHome(t)
	cfg, _ := loadConfig()
	out := renderConfig(cfg, configState{toolsEnabled: true, yesman: true, mcpServers: 1})
	for _, want := range []string{"SETTINGS", "MODEL", "ENGINE", "SESSION", "MEMORY", "FILES"} {
		if !strings.Contains(out, want) {
			t.Errorf("/config is missing the %s section", want)
		}
	}
	for _, key := range settingKeys() {
		if !strings.Contains(out, key) {
			t.Errorf("/config omits setting %q", key)
		}
	}
	// Session state is the reason /config exists — it isn't in config.json.
	if !strings.Contains(out, "yesman") || !strings.Contains(out, "⚠") {
		t.Error("/config should flag yesman while it is armed")
	}
}

// Memory is measured, so it must degrade gracefully with no server running.
func TestMemoryDisplayWithoutServer(t *testing.T) {
	// Another test may have left a server running — the state lives in a
	// package-level global, so make the precondition explicit.
	shutdownServer()
	withTempHome(t)
	cfg, _ := loadConfig()
	out := memoryDisplay(cfg)
	if !strings.Contains(out, "not running") {
		t.Errorf("expected a not-running note, got %q", out)
	}
	if !strings.Contains(out, "ctx_size") {
		t.Errorf("memory line should tie usage to ctx_size, got %q", out)
	}
}

func TestProcessRSSRejectsBadPID(t *testing.T) {
	if _, ok := processRSS(0); ok {
		t.Error("pid 0 should not report memory")
	}
	if _, ok := processRSS(-1); ok {
		t.Error("negative pid should not report memory")
	}
	// This process definitely exists and uses memory.
	if rss, ok := processRSS(osGetpid()); !ok || rss <= 0 {
		t.Errorf("could not read own RSS (got %d, ok=%v)", rss, ok)
	}
}

func TestResolveMaxToolRounds(t *testing.T) {
	if got := resolveMaxToolRounds(Config{}); got != defaultMaxToolRounds {
		t.Errorf("unset = %d, want the default %d", got, defaultMaxToolRounds)
	}
	if got := resolveMaxToolRounds(Config{MaxToolRounds: 100}); got != 100 {
		t.Errorf("explicit = %d, want 100", got)
	}
	// Negative is the "off" sentinel, and must not leak through as a
	// negative comparison target in the agent loop.
	if got := resolveMaxToolRounds(Config{MaxToolRounds: -1}); got != unlimitedToolRounds {
		t.Errorf("off = %d, want %d", got, unlimitedToolRounds)
	}
}

func TestMaxToolRoundsRoundTrips(t *testing.T) {
	withTempHome(t)
	for _, v := range []int{100, -1} {
		if err := saveConfig(Config{CurrentModel: defaultModel, MaxToolRounds: v}); err != nil {
			t.Fatal(err)
		}
		got, err := loadConfig()
		if err != nil {
			t.Fatal(err)
		}
		if got.MaxToolRounds != v {
			t.Errorf("max_tool_rounds = %d, want %d", got.MaxToolRounds, v)
		}
	}
	// Unset must stay 0 so it keeps tracking the default.
	if err := saveConfig(Config{CurrentModel: defaultModel}); err != nil {
		t.Fatal(err)
	}
	got, _ := loadConfig()
	if got.MaxToolRounds != 0 {
		t.Errorf("unset = %d, want 0", got.MaxToolRounds)
	}
}

func TestMaxToolRoundsDisplay(t *testing.T) {
	if s := maxToolRoundsDisplay(Config{}); !strings.Contains(s, "default") {
		t.Errorf("unset display %q should say it is the default", s)
	}
	if s := maxToolRoundsDisplay(Config{MaxToolRounds: -1}); !strings.Contains(s, "off") {
		t.Errorf("off display = %q", s)
	}
	if s := maxToolRoundsDisplay(Config{MaxToolRounds: 99}); s != "99" {
		t.Errorf("explicit display = %q, want 99", s)
	}
}
