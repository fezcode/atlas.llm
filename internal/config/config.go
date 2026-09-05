package config

import (
	"encoding/json"
	"fmt"
	"math"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"atlas.llm/internal/catalog"
)

// Config carries both tag sets on every field: the active config.json is
// JSON, while named profiles are piml (the Atlas suite's format). A field
// with one tag missing would silently vanish from one of the two files.
type Config struct {
	CurrentModel string `json:"current_model" piml:"current_model"`
	MaxTokens    int    `json:"max_tokens,omitempty" piml:"max_tokens,omitempty"`
	ToolsEnabled bool   `json:"tools_enabled,omitempty" piml:"tools_enabled,omitempty"`

	// GPULayers is the -ngl value handed to llama-server. nil means "auto"
	// — a pointer rather than an int so an explicit 0 (force CPU) is
	// distinguishable from an absent setting.
	GPULayers *int `json:"gpu_layers,omitempty" piml:"gpu_layers,omitempty"`

	// EngineVariant selects which llama.cpp release archive to download:
	// "cpu" (default) or "vulkan". Empty means auto.
	EngineVariant string `json:"engine_variant,omitempty" piml:"engine_variant,omitempty"`

	// Reasoning controls the model's internal <think> block: "on", "off",
	// or "" / "auto" for the default. Only affects models that have one.
	Reasoning string `json:"reasoning,omitempty" piml:"reasoning,omitempty"`

	// ShowThinking streams the think block's text into the transcript,
	// dimmed, instead of the byte counter. Display-only: the think text
	// never joins the history sent back to the model.
	ShowThinking bool `json:"show_thinking,omitempty" piml:"show_thinking,omitempty"`

	// MaxToolRounds caps tool-call rounds per message. 0 means the default;
	// a negative value means no cap.
	MaxToolRounds int `json:"max_tool_rounds,omitempty" piml:"max_tool_rounds,omitempty"`

	// CtxSize is the context window llama-server is started with (-c).
	// 0 means the default. The ceiling is whatever the model was trained
	// for, which is read from its GGUF metadata.
	CtxSize int `json:"ctx_size,omitempty" piml:"ctx_size,omitempty"`

	// Endpoint points inference at a llama-server someone else is running,
	// typically another atlas.llm in --serve mode. When set, this install
	// needs no engine and no model file: it becomes a client. Empty means
	// run inference locally.
	Endpoint string `json:"endpoint,omitempty" piml:"endpoint,omitempty"`

	// EndpointKey is the bearer token for Endpoint, for a server started
	// with --api-key. Empty is the normal case on a trusted LAN.
	EndpointKey string `json:"endpoint_key,omitempty" piml:"endpoint_key,omitempty"`

	// AMAEnabled toggles /ama: whether the agent may ask the user questions
	// through the interactive ask_user picker instead of deciding alone.
	AMAEnabled bool `json:"ama_enabled,omitempty" piml:"ama_enabled,omitempty"`

	// Engine tuning. Each field maps onto one llama-server flag; the zero
	// value always means "auto" — launch exactly as before the setting
	// existed. Pointer types are for settings where an explicit zero is a
	// real choice distinct from auto (the GPULayers lesson).
	KVOffload      string   `json:"kv_offload,omitempty" piml:"kv_offload,omitempty"`           // "off" → --no-kv-offload
	FlashAttn      string   `json:"flash_attn,omitempty" piml:"flash_attn,omitempty"`           // "on"/"off" → -fa; "" = auto
	CacheTypeK     string   `json:"cache_type_k,omitempty" piml:"cache_type_k,omitempty"`       // --cache-type-k; "" = auto
	CacheTypeV     string   `json:"cache_type_v,omitempty" piml:"cache_type_v,omitempty"`       // --cache-type-v; "" = auto
	Threads        int      `json:"threads,omitempty" piml:"threads,omitempty"`                 // -t; 0 = auto
	BatchSize      int      `json:"batch_size,omitempty" piml:"batch_size,omitempty"`           // -b; 0 = auto
	UBatchSize     int      `json:"ubatch_size,omitempty" piml:"ubatch_size,omitempty"`         // -ub; 0 = auto
	Parallel       int      `json:"parallel,omitempty" piml:"parallel,omitempty"`               // --parallel; 0 = auto
	CacheReuse     *int     `json:"cache_reuse,omitempty" piml:"cache_reuse,omitempty"`         // --cache-reuse; nil = auto, 0 = off
	Mmap           string   `json:"mmap,omitempty" piml:"mmap,omitempty"`                       // "off" → --no-mmap
	Mlock          string   `json:"mlock,omitempty" piml:"mlock,omitempty"`                     // "on" → --mlock
	Seed           *int     `json:"seed,omitempty" piml:"seed,omitempty"`                       // --seed; nil = auto
	Temperature    *float64 `json:"temperature,omitempty" piml:"temperature,omitempty"`         // per-request; nil = 0.2
	OverrideTensor string   `json:"override_tensor,omitempty" piml:"override_tensor,omitempty"` // -ot
	ContextShift   string   `json:"context_shift,omitempty" piml:"context_shift,omitempty"`     // "on" → --context-shift
}

// Reasoning settings.
const (
	ReasoningAuto = "auto"
	ReasoningOn   = "on"
	ReasoningOff  = "off"
)

// reasoningEnabledFor reports whether to let the model think, given whether
// this is an agentic turn.
//
// auto splits the two, because the cost differs enormously by task. Measured
// on qwen3.5-4b with "in one sentence, what is a KV cache?": thinking gave a
// first token at 62.1s and finished at 63.9s; without it, 0.3s and 3.2s. A
// twentyfold difference for a question that needed none of it.
//
// So auto turns thinking off for plain chat and leaves it on for tool-driven
// turns, where it measurably improves which tool gets called. `on` and `off`
// override both. One-shot commands never think either way.
func ReasoningEnabledFor(cfg Config, agentic bool) bool {
	switch strings.ToLower(strings.TrimSpace(cfg.Reasoning)) {
	case ReasoningOff:
		return false
	case ReasoningOn:
		return true
	default:
		return agentic
	}
}

// reasoningEnabled is the conversational-turn case.
func ReasoningEnabled(cfg Config) bool { return ReasoningEnabledFor(cfg, false) }

// reasoningDisplay renders the setting for /set and /config.
func ReasoningDisplay(cfg Config) string {
	switch strings.ToLower(strings.TrimSpace(cfg.Reasoning)) {
	case ReasoningOff:
		return "off (faster replies; may reduce tool-use accuracy)"
	case ReasoningOn:
		return "on (always; slower but best at multi-step work)"
	default:
		return "auto (off for chat, on for tool-driven turns)"
	}
}

// defaultMaxToolRounds is the tool-call round budget for one message when
// nothing is configured.
//
// Raised from 20 once identical-call detection landed: that catches a stuck
// model directly, so this no longer has to double as the runaway guard and
// can be generous enough for work that legitimately needs many steps.
const DefaultMaxToolRounds = 40

// unlimitedToolRounds is what resolveMaxToolRounds returns when the cap is
// switched off. The repeated-call detector and Esc remain as backstops.
const UnlimitedToolRounds = 0

// resolveMaxToolRounds returns the per-message tool-call budget, or
// unlimitedToolRounds when the user has turned the cap off.
func ResolveMaxToolRounds(cfg Config) int {
	switch {
	case cfg.MaxToolRounds < 0:
		return UnlimitedToolRounds
	case cfg.MaxToolRounds == 0:
		return DefaultMaxToolRounds
	default:
		return cfg.MaxToolRounds
	}
}

// maxToolRoundsDisplay renders the setting for `/set` and `/config`.
func MaxToolRoundsDisplay(cfg Config) string {
	n := ResolveMaxToolRounds(cfg)
	if n == UnlimitedToolRounds {
		return "off (no cap)"
	}
	if cfg.MaxToolRounds == 0 {
		return fmt.Sprintf("%d (default)", n)
	}
	return fmt.Sprintf("%d", n)
}

// defaultCtxSize is the context window used when nothing is configured.
// Modest on purpose: KV cache memory scales linearly with it, and most
// models here are run on laptops.
const DefaultCtxSize = 16384

// maxConfigurableCtx caps what /set will accept even when a model claims a
// larger trained context. Once 131072, from when a KV cache at Qwen's full
// 262144 meant tens of gigabytes of f16; with quantized cache types and
// hybrid-attention models that hold KV in only a fraction of their layers,
// contexts that size are genuinely servable on consumer cards. The offload
// planner still charges the cache against VRAM before every launch, so the
// cap only has to reject the absurd, not police fit.
const MaxConfigurableCtx = 262144

// minConfigurableCtx keeps a value from being set so small that the system
// prompt and tool definitions alone won't fit.
const MinConfigurableCtx = 2048

// defaultMaxTokens is the reply-length cap applied when the config hasn't
// set one. Sized to fit almost any chat response while leaving plenty of
// the 16K ctx window for conversation history.
const DefaultMaxTokens = 4096

func AtlasDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(home, ".atlas", "atlas.llm.data")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}
	return dir, nil
}

func ModelsDir() (string, error) {
	base, err := AtlasDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(base, "models")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}
	return dir, nil
}

func ConfigPath() (string, error) {
	base, err := AtlasDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(base, "config.json"), nil
}

func ModelPath(m catalog.Model) (string, error) {
	dir, err := ModelsDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, m.Filename), nil
}

func FindModel(name string) (catalog.Model, bool) {
	for _, m := range catalog.AvailableModels {
		if m.Name == name {
			return m, true
		}
	}
	return catalog.Model{}, false
}

func LoadConfig() (Config, error) {
	cfg := Config{CurrentModel: catalog.DefaultModel}
	p, err := ConfigPath()
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
		cfg.CurrentModel = catalog.DefaultModel
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = DefaultMaxTokens
	}
	return cfg, nil
}

func SaveConfig(cfg Config) error {
	p, err := ConfigPath()
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(p, data, 0644)
}

func IsModelDownloaded(m catalog.Model) bool {
	p, err := ModelPath(m)
	if err != nil {
		return false
	}
	info, err := os.Stat(p)
	return err == nil && !info.IsDir() && info.Size() > 0
}

func CurrentModel() (catalog.Model, error) {
	cfg, err := LoadConfig()
	if err != nil {
		return catalog.Model{}, err
	}
	m, ok := FindModel(cfg.CurrentModel)
	if !ok {
		return catalog.Model{}, fmt.Errorf("unknown model in config: %s", cfg.CurrentModel)
	}
	return m, nil
}

// parseModelSize turns a registry Size string ("~700MB", "~2.5GB") into
// bytes for comparison. Returns 0 when it can't be parsed, which sorts such
// entries last rather than making them look tiny.
func ParseModelSize(s string) int64 {
	s = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(s), "~"))
	upper := strings.ToUpper(s)
	var mult float64
	switch {
	case strings.HasSuffix(upper, "GB"):
		mult, upper = 1e9, strings.TrimSuffix(upper, "GB")
	case strings.HasSuffix(upper, "MB"):
		mult, upper = 1e6, strings.TrimSuffix(upper, "MB")
	default:
		return 0
	}
	n, err := strconv.ParseFloat(strings.TrimSpace(upper), 64)
	if err != nil || n <= 0 {
		return 0
	}
	// Round rather than truncate: 8.2 * 1e9 lands just under 8.2e9 in
	// binary floating point, which would report 8199999999 bytes.
	return int64(math.Round(n * mult))
}

// lightestModel returns the smallest model in the registry — the safe one
// to fall back to when a heavy model makes startup unusable. Falls back to
// defaultModel if no size parses.
func LightestModel() catalog.Model {
	best, bestSize := catalog.Model{}, int64(0)
	for _, m := range catalog.AvailableModels {
		size := ParseModelSize(m.Size)
		if size == 0 {
			continue
		}
		if bestSize == 0 || size < bestSize {
			best, bestSize = m, size
		}
	}
	if bestSize == 0 {
		if m, ok := FindModel(catalog.DefaultModel); ok {
			return m
		}
		return catalog.AvailableModels[0]
	}
	return best
}

// defaultServePort matches llama-server's own default, so a user who reaches
// for a port number guesses right.
const DefaultServePort = 8080

// normalizeEndpoint canonicalises what a user types into `/set endpoint`.
// Accepts "192.168.1.50", "192.168.1.50:8080", or a full URL, because all
// three are things people reasonably type for a machine on their LAN.
// Returns "" for input that clears the setting.
func NormalizeEndpoint(raw string) (string, error) {
	s := strings.TrimSpace(raw)
	if s == "" || strings.EqualFold(s, "local") || strings.EqualFold(s, "off") ||
		strings.EqualFold(s, "none") {
		return "", nil
	}
	if !strings.Contains(s, "://") {
		s = "http://" + s
	}
	u, err := url.Parse(s)
	if err != nil {
		return "", fmt.Errorf("invalid endpoint %q: %v", raw, err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return "", fmt.Errorf("endpoint %q: expected an http:// or https:// address", raw)
	}
	if u.Hostname() == "" {
		return "", fmt.Errorf("endpoint %q: no host in the address", raw)
	}
	if u.Port() == "" {
		u.Host = net.JoinHostPort(u.Hostname(), strconv.Itoa(DefaultServePort))
	}
	// Only scheme://host:port is meaningful; the API paths are ours to append.
	return strings.TrimRight((&url.URL{Scheme: u.Scheme, Host: u.Host}).String(), "/"), nil
}

// remoteEndpoint returns the configured endpoint and key, or "" when this
// install runs inference locally.
func RemoteEndpoint() (string, string) {
	cfg, err := LoadConfig()
	if err != nil {
		return "", ""
	}
	ep, err := NormalizeEndpoint(cfg.Endpoint)
	if err != nil {
		return "", ""
	}
	return ep, strings.TrimSpace(cfg.EndpointKey)
}

// isRemoteMode reports whether inference runs on another machine. Used to
// skip the local engine/model requirements and to mark settings that the
// remote decides.
func IsRemoteMode() bool {
	ep, _ := RemoteEndpoint()
	return ep != ""
}

// remoteDecidesSetting reports whether a setting is fixed by the machine
// running the model rather than the one typing. These are all passed to
// llama-server at spawn, so a client cannot change them over HTTP.
func RemoteDecidesSetting(key string) bool {
	switch strings.ToLower(strings.TrimSpace(key)) {
	case "ctx_size", "gpu_layers", "engine_variant",
		// The engine-tuning settings all bind at server launch, which a
		// remote did on its own machine. temperature is the exception:
		// it rides on each request, so it applies against a remote too.
		"kv_offload", "flash_attn", "cache_type_k", "cache_type_v",
		"threads", "batch_size", "ubatch_size", "parallel", "cache_reuse",
		"mmap", "mlock", "seed", "override_tensor", "context_shift":
		return true
	}
	return false
}
