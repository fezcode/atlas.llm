package tui

import (
	"fmt"
	"strings"
	"testing"

	"atlas.llm/internal/config"
	"atlas.llm/internal/engine"
)

// argsValue returns the value following a flag in an args slice, and whether
// the flag is present at all.
func argsValue(args []string, flag string) (string, bool) {
	for i, a := range args {
		if a == flag {
			if i+1 < len(args) {
				return args[i+1], true
			}
			return "", true
		}
	}
	return "", false
}

// applyTuning is a test helper: look up a setting by key and apply a value
// through the registry, the same path handleSet takes.
func applyTuning(t *testing.T, cfg *config.Config, key, val string) error {
	t.Helper()
	s, ok := findSetting(key)
	if !ok {
		t.Fatalf("setting %q is not registered", key)
	}
	if s.Apply == nil {
		t.Fatalf("setting %q has no Apply — it can't be set via the generic path", key)
	}
	return s.Apply(cfg, val)
}

// Every tuning setting must be registered, applyable, and render a value.
func TestTuningSettingsRegistered(t *testing.T) {
	keys := []string{
		"kv_offload", "flash_attn", "cache_type_k", "cache_type_v",
		"threads", "batch_size", "ubatch_size", "parallel", "cache_reuse",
		"mmap", "mlock", "seed", "temperature", "override_tensor",
		"context_shift",
	}
	for _, k := range keys {
		s, ok := findSetting(k)
		if !ok {
			t.Errorf("%s is not in the settings registry", k)
			continue
		}
		if s.Apply == nil {
			t.Errorf("%s has no Apply func", k)
		}
		if s.Usage == "" || s.Summary == "" {
			t.Errorf("%s is missing usage or summary", k)
		}
		if got := s.Value(config.Config{}); got == "" {
			t.Errorf("%s renders an empty value on a zero config", k)
		}
		if s.Detail == nil || s.Detail(config.Config{}) == "" {
			t.Errorf("%s has no detail text", k)
		}
	}
}

func TestApplyTuningSettings(t *testing.T) {
	cases := []struct {
		key, val string
		wantErr  bool
		check    func(config.Config) bool
	}{
		{"kv_offload", "off", false, func(c config.Config) bool { return c.KVOffload == "off" }},
		{"kv_offload", "on", false, func(c config.Config) bool { return c.KVOffload == "" }},
		{"kv_offload", "maybe", true, nil},
		{"flash_attn", "off", false, func(c config.Config) bool { return c.FlashAttn == "off" }},
		{"flash_attn", "auto", false, func(c config.Config) bool { return c.FlashAttn == "" }},
		{"flash_attn", "fast", true, nil},
		{"cache_type_k", "q4_0", false, func(c config.Config) bool { return c.CacheTypeK == "q4_0" }},
		{"cache_type_k", "auto", false, func(c config.Config) bool { return c.CacheTypeK == "" }},
		{"cache_type_k", "q9_9", true, nil},
		{"cache_type_v", "f16", false, func(c config.Config) bool { return c.CacheTypeV == "f16" }},
		{"threads", "4", false, func(c config.Config) bool { return c.Threads == 4 }},
		{"threads", "auto", false, func(c config.Config) bool { return c.Threads == 0 }},
		{"threads", "0", true, nil},
		{"threads", "-2", true, nil},
		{"batch_size", "1024", false, func(c config.Config) bool { return c.BatchSize == 1024 }},
		{"batch_size", "nope", true, nil},
		{"ubatch_size", "256", false, func(c config.Config) bool { return c.UBatchSize == 256 }},
		{"parallel", "4", false, func(c config.Config) bool { return c.Parallel == 4 }},
		{"parallel", "0", true, nil},
		{"cache_reuse", "off", false, func(c config.Config) bool { return c.CacheReuse != nil && *c.CacheReuse == 0 }},
		{"cache_reuse", "512", false, func(c config.Config) bool { return c.CacheReuse != nil && *c.CacheReuse == 512 }},
		{"cache_reuse", "auto", false, func(c config.Config) bool { return c.CacheReuse == nil }},
		{"mmap", "off", false, func(c config.Config) bool { return c.Mmap == "off" }},
		{"mmap", "on", false, func(c config.Config) bool { return c.Mmap == "" }},
		{"mlock", "on", false, func(c config.Config) bool { return c.Mlock == "on" }},
		{"mlock", "off", false, func(c config.Config) bool { return c.Mlock == "" }},
		{"seed", "42", false, func(c config.Config) bool { return c.Seed != nil && *c.Seed == 42 }},
		{"seed", "auto", false, func(c config.Config) bool { return c.Seed == nil }},
		{"seed", "-3", true, nil},
		{"temperature", "0.7", false, func(c config.Config) bool { return c.Temperature != nil && *c.Temperature == 0.7 }},
		{"temperature", "auto", false, func(c config.Config) bool { return c.Temperature == nil }},
		{"temperature", "3.5", true, nil},
		{"override_tensor", `exps=CPU`, false, func(c config.Config) bool { return c.OverrideTensor == "exps=CPU" }},
		{"override_tensor", "off", false, func(c config.Config) bool { return c.OverrideTensor == "" }},
		{"context_shift", "on", false, func(c config.Config) bool { return c.ContextShift == "on" }},
		{"context_shift", "off", false, func(c config.Config) bool { return c.ContextShift == "" }},
	}
	for _, tc := range cases {
		cfg := config.Config{}
		err := applyTuning(t, &cfg, tc.key, tc.val)
		if tc.wantErr {
			if err == nil {
				t.Errorf("%s=%s: expected an error", tc.key, tc.val)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s=%s: %v", tc.key, tc.val, err)
			continue
		}
		if tc.check != nil && !tc.check(cfg) {
			t.Errorf("%s=%s did not land in the config: %+v", tc.key, tc.val, cfg)
		}
	}
}

// Quantizing V requires flash attention, and llama-server fails at load when
// they disagree — catch it at /set time instead.
func TestFlashAttnGuardsQuantizedV(t *testing.T) {
	cfg := config.Config{FlashAttn: "off"}
	if err := applyTuning(t, &cfg, "cache_type_v", "q8_0"); err == nil {
		t.Error("quantized V was accepted with flash_attn off")
	}
	cfg = config.Config{CacheTypeV: "q4_0"}
	if err := applyTuning(t, &cfg, "flash_attn", "off"); err == nil {
		t.Error("flash_attn off was accepted with a quantized V cache")
	}
	// f16 V is fine without flash attention.
	cfg = config.Config{FlashAttn: "off"}
	if err := applyTuning(t, &cfg, "cache_type_v", "f16"); err != nil {
		t.Errorf("f16 V with flash_attn off: %v", err)
	}
}

func TestCacheTypeElemBytes(t *testing.T) {
	cases := map[string]float64{
		"f32":  4,
		"f16":  2,
		"bf16": 2,
		"q8_0": 34.0 / 32.0,
		"q5_1": 24.0 / 32.0,
		"q5_0": 22.0 / 32.0,
		"q4_1": 20.0 / 32.0,
		"q4_0": 18.0 / 32.0,
	}
	for typ, want := range cases {
		if got := engine.CacheTypeElemBytes(typ); got != want {
			t.Errorf("cacheTypeElemBytes(%s) = %v, want %v", typ, got, want)
		}
	}
	// Unknown types err high, matching the estimator's stance everywhere.
	if got := engine.CacheTypeElemBytes("iq4_nl"); got < 18.0/32.0 || got > 2 {
		t.Errorf("unknown type = %v, want something sane and conservative", got)
	}
}

func TestResolveThreads(t *testing.T) {
	if got := engine.ResolveThreads(config.Config{Threads: 3}); got != 3 {
		t.Errorf("explicit threads = %d, want 3", got)
	}
	auto := engine.ResolveThreads(config.Config{})
	if auto < 1 || auto > 6 {
		t.Errorf("auto threads = %d, want 1..6 (NumCPU-1 capped at 6)", auto)
	}
}

// A zero config must produce exactly the flag set that shipped before tuning
// settings existed — these defaults are measured and documented, and a silent
// drift here changes every user's launch.
func TestBuildServerArgsDefaultsOptimized(t *testing.T) {
	args, slots, serverCtx := engine.BuildServerArgs("model.gguf", engine.ServeOptions{}, 8080,
		engine.OffloadPlan{NGL: engine.MaxGPULayers}, 16384, true, config.Config{})
	if slots != engine.KvCacheSlots {
		t.Errorf("slots = %d, want %d", slots, engine.KvCacheSlots)
	}
	if serverCtx != 16384*engine.KvCacheSlots {
		t.Errorf("serverCtx = %d, want %d", serverCtx, 16384*engine.KvCacheSlots)
	}
	for flag, want := range map[string]string{
		"-fa":            "on",
		"--cache-type-k": "q8_0",
		"--cache-type-v": "q8_0",
		"--parallel":     fmt.Sprintf("%d", engine.KvCacheSlots),
		"--cache-reuse":  "256",
		"-c":             fmt.Sprintf("%d", 16384*engine.KvCacheSlots),
		"-ngl":           fmt.Sprintf("%d", engine.MaxGPULayers),
	} {
		if got, ok := argsValue(args, flag); !ok || got != want {
			t.Errorf("%s = %q (present=%v), want %q", flag, got, ok, want)
		}
	}
	for _, absent := range []string{"--no-kv-offload", "--no-mmap", "--mlock",
		"--seed", "-ot", "--context-shift", "-b", "-ub", "--n-cpu-moe"} {
		if _, ok := argsValue(args, absent); ok {
			t.Errorf("%s present on a zero config", absent)
		}
	}
}

func TestBuildServerArgsDefaultsFallback(t *testing.T) {
	args, slots, serverCtx := engine.BuildServerArgs("model.gguf", engine.ServeOptions{}, 8080,
		engine.OffloadPlan{NGL: 0}, 16384, false, config.Config{})
	if slots != 1 {
		t.Errorf("fallback slots = %d, want 1", slots)
	}
	if serverCtx != 16384 {
		t.Errorf("fallback serverCtx = %d, want 16384", serverCtx)
	}
	for _, absent := range []string{"-fa", "--cache-type-k", "--cache-type-v",
		"--parallel", "--cache-reuse"} {
		if _, ok := argsValue(args, absent); ok {
			t.Errorf("%s present on the fallback flag set", absent)
		}
	}
}

func TestBuildServerArgsTuning(t *testing.T) {
	i512 := 512
	seed := 7
	cases := []struct {
		name string
		cfg  config.Config
		flag string
		want string // "" means: flag present with no asserted value
	}{
		{"kv_offload off", config.Config{KVOffload: "off"}, "--no-kv-offload", ""},
		{"mmap off", config.Config{Mmap: "off"}, "--no-mmap", ""},
		{"mlock on", config.Config{Mlock: "on"}, "--mlock", ""},
		{"context_shift on", config.Config{ContextShift: "on"}, "--context-shift", ""},
		{"seed", config.Config{Seed: &seed}, "--seed", "7"},
		{"threads", config.Config{Threads: 2}, "-t", "2"},
		{"batch", config.Config{BatchSize: 4096}, "-b", "4096"},
		{"ubatch", config.Config{UBatchSize: 128}, "-ub", "128"},
		{"override_tensor", config.Config{OverrideTensor: "exps=CPU"}, "-ot", "exps=CPU"},
		{"cache_type_k", config.Config{CacheTypeK: "q4_0"}, "--cache-type-k", "q4_0"},
		{"cache_type_v", config.Config{CacheTypeV: "f16"}, "--cache-type-v", "f16"},
		{"flash_attn off", config.Config{FlashAttn: "off"}, "-fa", "off"},
		{"cache_reuse", config.Config{CacheReuse: &i512}, "--cache-reuse", "512"},
	}
	for _, tc := range cases {
		args, _, _ := engine.BuildServerArgs("model.gguf", engine.ServeOptions{}, 8080,
			engine.OffloadPlan{NGL: 0}, 16384, true, tc.cfg)
		got, ok := argsValue(args, tc.flag)
		if !ok {
			t.Errorf("%s: %s missing from args %v", tc.name, tc.flag, args)
			continue
		}
		if tc.want != "" && got != tc.want {
			t.Errorf("%s: %s = %q, want %q", tc.name, tc.flag, got, tc.want)
		}
	}

	// parallel reshapes the slot layout: the KV budget stays fixed, so more
	// slots means less context each, never more VRAM.
	args, slots, serverCtx := engine.BuildServerArgs("model.gguf", engine.ServeOptions{}, 8080,
		engine.OffloadPlan{NGL: 0}, 16384, true, config.Config{Parallel: 4})
	if slots != 4 {
		t.Errorf("parallel slots = %d, want 4", slots)
	}
	if serverCtx != 16384*engine.KvCacheSlots {
		t.Errorf("parallel serverCtx = %d, want the fixed budget %d", serverCtx, 16384*engine.KvCacheSlots)
	}
	if got, _ := argsValue(args, "--parallel"); got != "4" {
		t.Errorf("--parallel = %q, want 4", got)
	}

	// Explicit tuning is a user instruction and rides on the fallback flag
	// set too — dropped silently it would make /set a lie; if the old engine
	// rejects it, the launch fails loudly instead.
	args, _, _ = engine.BuildServerArgs("model.gguf", engine.ServeOptions{}, 8080,
		engine.OffloadPlan{NGL: 0}, 16384, false, config.Config{FlashAttn: "on", KVOffload: "off"})
	if got, ok := argsValue(args, "-fa"); !ok || got != "on" {
		t.Errorf("explicit flash_attn dropped on fallback: %v", args)
	}
	if _, ok := argsValue(args, "--no-kv-offload"); !ok {
		t.Error("explicit kv_offload dropped on fallback")
	}
}

// The fingerprint is restart identity: raw settings, not resolved autos, so
// estimate drift never evicts a server but a user change always does.
func TestTuningFingerprint(t *testing.T) {
	base := engine.TuningFingerprint(config.Config{})
	if base == "" {
		t.Fatal("zero-config fingerprint is empty")
	}
	seed := 9
	i128 := 128
	temp := 0.9
	changed := []config.Config{
		{KVOffload: "off"},
		{FlashAttn: "off"},
		{CacheTypeK: "q4_0"},
		{CacheTypeV: "f16"},
		{Threads: 2},
		{BatchSize: 1024},
		{UBatchSize: 64},
		{Parallel: 4},
		{CacheReuse: &i128},
		{Mmap: "off"},
		{Mlock: "on"},
		{Seed: &seed},
		{OverrideTensor: "exps=CPU"},
		{ContextShift: "on"},
	}
	seen := map[string]bool{base: true}
	for _, c := range changed {
		fp := engine.TuningFingerprint(c)
		if seen[fp] {
			t.Errorf("fingerprint collision for %+v", c)
		}
		seen[fp] = true
	}
	// temperature is request-level: changing it must NOT restart the server.
	if got := engine.TuningFingerprint(config.Config{Temperature: &temp}); got != base {
		t.Error("temperature changed the fingerprint — it would restart the server for nothing")
	}
	// cache_reuse explicit-0 must differ from auto (the GPULayers lesson).
	zero := 0
	if engine.TuningFingerprint(config.Config{CacheReuse: &zero}) == base {
		t.Error("cache_reuse off is indistinguishable from auto")
	}
}

// Server-side tuning is decided by the remote when one is attached; the
// notice in handleSet keys off this list.
func TestRemoteDecidesTuning(t *testing.T) {
	for _, k := range []string{"kv_offload", "flash_attn", "cache_type_k",
		"cache_type_v", "threads", "batch_size", "ubatch_size", "parallel",
		"cache_reuse", "mmap", "mlock", "seed", "override_tensor",
		"context_shift"} {
		if !config.RemoteDecidesSetting(k) {
			t.Errorf("%s should be flagged as remote-decided", k)
		}
	}
	if config.RemoteDecidesSetting("temperature") {
		t.Error("temperature is sent per request — the local value applies even against a remote")
	}
}

func TestResolveTemperature(t *testing.T) {
	if got := engine.ResolveTemperature(config.Config{}); got != 0.2 {
		t.Errorf("default temperature = %v, want 0.2", got)
	}
	temp := 0.8
	if got := engine.ResolveTemperature(config.Config{Temperature: &temp}); got != 0.8 {
		t.Errorf("explicit temperature = %v, want 0.8", got)
	}
}

// With kv_offload off the KV cache leaves VRAM, so the fit math must stop
// charging it there — that is the whole point of the setting: more weight
// layers fit.
func TestVRAMKVCharge(t *testing.T) {
	kv := int64(2 << 30)
	if got := engine.VramKVCharge(config.Config{}, kv); got != kv {
		t.Errorf("default charge = %d, want %d", got, kv)
	}
	if got := engine.VramKVCharge(config.Config{KVOffload: "off"}, kv); got != 0 {
		t.Errorf("kv_offload off charge = %d, want 0", got)
	}
}

// The typed KV estimate must reduce with smaller cache types and reproduce
// the documented q8_0 figure on defaults.
func TestKVEstimateFollowsCacheType(t *testing.T) {
	meta := engine.GgufMeta{BlockCount: 32, HeadCount: 32, HeadCountKV: 8,
		KeyLength: 128, EmbeddingLength: 4096}
	def := meta.KvCacheBytes(16384)
	f16 := int64(float64(def) / engine.KvCacheElemBytes * 2)
	got := meta.KvCacheBytesTyped(16384, 2.0)
	if got != f16 {
		t.Errorf("f16 estimate = %d, want %d", got, f16)
	}
	q4 := meta.KvCacheBytesTyped(16384, 18.0/32.0)
	if q4 >= def {
		t.Errorf("q4_0 estimate %d is not below the q8_0 default %d", q4, def)
	}
}

// Sanity: the tuning args never leak into the fingerprint of what /set
// prints — every registered tuning key renders a non-empty display both at
// defaults and when explicitly set.
func TestTuningValueRendering(t *testing.T) {
	seed := 5
	temp := 0.6
	i64 := 64
	cfg := config.Config{KVOffload: "off", FlashAttn: "off", CacheTypeK: "q4_0",
		CacheTypeV: "f16", Threads: 2, BatchSize: 512, UBatchSize: 64,
		Parallel: 3, CacheReuse: &i64, Mmap: "off", Mlock: "on", Seed: &seed,
		Temperature: &temp, OverrideTensor: "exps=CPU", ContextShift: "on"}
	for _, k := range []string{"kv_offload", "flash_attn", "cache_type_k",
		"cache_type_v", "threads", "batch_size", "ubatch_size", "parallel",
		"cache_reuse", "mmap", "mlock", "seed", "temperature",
		"override_tensor", "context_shift"} {
		s, ok := findSetting(k)
		if !ok {
			t.Fatalf("%s missing", k)
		}
		v := s.Value(cfg)
		if v == "" || strings.Contains(v, "%!") {
			t.Errorf("%s renders %q on an explicit config", k, v)
		}
	}
}
