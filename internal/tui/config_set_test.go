package tui

import (
	"os"
	"strings"
	"testing"

	"atlas.llm/internal/catalog"
	"atlas.llm/internal/config"
	"atlas.llm/internal/engine"
)

func osGetpid() int { return os.Getpid() }

// The registry is the single source of truth for /set, /config, and
// /help set — every key must be complete and self-consistent.
func TestSettingsRegistryIsWellFormed(t *testing.T) {
	withTempHome(t)
	cfg, _ := config.LoadConfig()
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
	cfg, _ := config.LoadConfig()
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
	cfg, _ := config.LoadConfig()
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
	engine.ShutdownServer()
	withTempHome(t)
	cfg, _ := config.LoadConfig()
	out := engine.MemoryDisplay(cfg)
	if !strings.Contains(out, "not running") {
		t.Errorf("expected a not-running note, got %q", out)
	}
	if !strings.Contains(out, "ctx_size") {
		t.Errorf("memory line should tie usage to ctx_size, got %q", out)
	}
}

func TestProcessRSSRejectsBadPID(t *testing.T) {
	if _, ok := engine.ProcessRSS(0); ok {
		t.Error("pid 0 should not report memory")
	}
	if _, ok := engine.ProcessRSS(-1); ok {
		t.Error("negative pid should not report memory")
	}
	// This process definitely exists and uses memory.
	if rss, ok := engine.ProcessRSS(osGetpid()); !ok || rss <= 0 {
		t.Errorf("could not read own RSS (got %d, ok=%v)", rss, ok)
	}
}

func TestResolveMaxToolRounds(t *testing.T) {
	if got := config.ResolveMaxToolRounds(config.Config{}); got != config.DefaultMaxToolRounds {
		t.Errorf("unset = %d, want the default %d", got, config.DefaultMaxToolRounds)
	}
	if got := config.ResolveMaxToolRounds(config.Config{MaxToolRounds: 100}); got != 100 {
		t.Errorf("explicit = %d, want 100", got)
	}
	// Negative is the "off" sentinel, and must not leak through as a
	// negative comparison target in the agent loop.
	if got := config.ResolveMaxToolRounds(config.Config{MaxToolRounds: -1}); got != config.UnlimitedToolRounds {
		t.Errorf("off = %d, want %d", got, config.UnlimitedToolRounds)
	}
}

func TestMaxToolRoundsRoundTrips(t *testing.T) {
	withTempHome(t)
	for _, v := range []int{100, -1} {
		if err := config.SaveConfig(config.Config{CurrentModel: catalog.DefaultModel, MaxToolRounds: v}); err != nil {
			t.Fatal(err)
		}
		got, err := config.LoadConfig()
		if err != nil {
			t.Fatal(err)
		}
		if got.MaxToolRounds != v {
			t.Errorf("max_tool_rounds = %d, want %d", got.MaxToolRounds, v)
		}
	}
	// Unset must stay 0 so it keeps tracking the default.
	if err := config.SaveConfig(config.Config{CurrentModel: catalog.DefaultModel}); err != nil {
		t.Fatal(err)
	}
	got, _ := config.LoadConfig()
	if got.MaxToolRounds != 0 {
		t.Errorf("unset = %d, want 0", got.MaxToolRounds)
	}
}

func TestMaxToolRoundsDisplay(t *testing.T) {
	if s := config.MaxToolRoundsDisplay(config.Config{}); !strings.Contains(s, "default") {
		t.Errorf("unset display %q should say it is the default", s)
	}
	if s := config.MaxToolRoundsDisplay(config.Config{MaxToolRounds: -1}); !strings.Contains(s, "off") {
		t.Errorf("off display = %q", s)
	}
	if s := config.MaxToolRoundsDisplay(config.Config{MaxToolRounds: 99}); s != "99" {
		t.Errorf("explicit display = %q, want 99", s)
	}
}

// Hybrid models keep a KV cache on only some layers. Assuming all of them
// overestimated Qwen3.5 by about 4x, which is why full_attention_interval
// is read from the GGUF.
func TestKVLayersRespectsHybridInterval(t *testing.T) {
	hybrid := engine.GgufMeta{BlockCount: 32, FullAttentionInterval: 4, HeadCountKV: 4, KeyLength: 256}
	if got := hybrid.KvLayers(); got != 8 {
		t.Errorf("kvLayers = %d, want 8 (32 blocks, interval 4)", got)
	}
	dense := engine.GgufMeta{BlockCount: 32, HeadCountKV: 4, KeyLength: 256}
	if got := dense.KvLayers(); got != 32 {
		t.Errorf("dense kvLayers = %d, want 32", got)
	}
	// The hybrid cache must be proportionally smaller.
	if h, d := hybrid.KvCacheBytes(16384), dense.KvCacheBytes(16384); h*4 != d {
		t.Errorf("hybrid KV %d should be a quarter of dense %d", h, d)
	}
}

// The estimate is checked against a real measurement, so a regression in
// the formula shows up as a number that no longer matches the hardware.
func TestKVEstimateMatchesMeasuredQwen(t *testing.T) {
	m := engine.GgufMeta{BlockCount: 32, FullAttentionInterval: 4, HeadCountKV: 4, KeyLength: 256}
	// llama-server holding Qwen3.5-4B measured 0.91 GB resident at ctx
	// 16384 and 2.36 GB at 65536 — a 1.45 GB delta.
	delta := m.KvCacheBytes(65536) - m.KvCacheBytes(16384)
	const measured = 1.45e9
	if ratio := float64(delta) / measured; ratio < 0.7 || ratio > 1.4 {
		t.Errorf("predicted KV growth %s is %.1fx the measured 1.45 GB — formula drifted",
			engine.FormatBytes(delta), ratio)
	}
}

func TestKVCacheGrowsWithContext(t *testing.T) {
	m := engine.GgufMeta{BlockCount: 32, FullAttentionInterval: 4, HeadCountKV: 4, KeyLength: 256}
	if a, b := m.KvCacheBytes(16384), m.KvCacheBytes(32768); b != a*2 {
		t.Errorf("doubling ctx gave %d vs %d — expected linear growth", b, a)
	}
	if m.KvCacheBytes(0) != 0 {
		t.Error("zero context should cost nothing")
	}
	if (engine.GgufMeta{}).KvCacheBytes(16384) != 0 {
		t.Error("empty metadata should not produce an estimate")
	}
}

// A downloaded model must be sized from the file, not the registry string —
// the strings are approximate (gemma-4-e2b declares ~2.9GB, is 3.11GB).
func TestModelSizeBytesPrefersRealFile(t *testing.T) {
	withTempHome(t)
	m := catalog.Model{Name: "x", Filename: "x.gguf", Size: "~1GB"}
	if got := engine.ModelSizeBytes(m); got != int64(1e9) {
		t.Errorf("undownloaded model = %d, want the declared 1e9", got)
	}
	p, err := config.ModelPath(m)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, make([]byte, 4096), 0644); err != nil {
		t.Fatal(err)
	}
	if got := engine.ModelSizeBytes(m); got != 4096 {
		t.Errorf("downloaded model = %d, want the real 4096", got)
	}
}
