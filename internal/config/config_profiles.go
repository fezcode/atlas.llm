package config

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	piml "github.com/fezcode/go-piml"

	"atlas.llm/internal/catalog"
)

// Named configs ("profiles"): full snapshots of the active config saved
// under a name, so a user can flip between whole setups — a fast one (small
// model, small context, no reasoning) and a quality one (big model, big
// context) — with a single /config load. Nothing about how settings are
// read changes: config.json remains the one active config, and load/save
// copy the whole thing.
//
// Profiles are piml files — the Atlas suite's format, same as recipe.piml —
// while the active config.json stays JSON. The Config struct carries both
// tag sets to serve the two encodings.

// profileNameRE is the allowed shape of a profile name. It doubles as a
// filename, so no separators, dots, or spaces — only these, 1..32 chars.
var profileNameRE = regexp.MustCompile(`^[A-Za-z0-9_-]{1,32}$`)

func ValidProfileName(name string) error {
	if !profileNameRE.MatchString(name) {
		return fmt.Errorf("invalid profile name %q — use letters, digits, dash, or underscore (max 32)", name)
	}
	return nil
}

// profilesDir is where named configs live, created on demand.
func profilesDir() (string, error) {
	base, err := AtlasDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(base, "profiles")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}
	return dir, nil
}

func ProfilePath(name string) (string, error) {
	if err := ValidProfileName(name); err != nil {
		return "", err
	}
	dir, err := profilesDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, name+".piml"), nil
}

// saveProfile writes cfg to profiles/<name>.piml. It never touches the active
// config.json — a save is a snapshot, not a switch.
func SaveProfile(name string, cfg Config) error {
	p, err := ProfilePath(name)
	if err != nil {
		return err
	}
	data, err := piml.Marshal(cfg)
	if err != nil {
		return err
	}
	return os.WriteFile(p, data, 0644)
}

// loadProfile reads a saved profile. The returned Config is normalised the
// same way loadConfig normalises the active file, so a hand-edited or older
// profile still comes back with sane defaults.
func LoadProfile(name string) (Config, error) {
	cfg := Config{CurrentModel: catalog.DefaultModel}
	p, err := ProfilePath(name)
	if err != nil {
		return cfg, err
	}
	data, err := os.ReadFile(p)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, fmt.Errorf("no profile named %q — /config list shows the saved ones", name)
		}
		return cfg, err
	}
	if err := piml.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("profile %q is corrupt: %w", name, err)
	}
	if cfg.CurrentModel == "" {
		cfg.CurrentModel = catalog.DefaultModel
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = DefaultMaxTokens
	}
	return cfg, nil
}

func DeleteProfile(name string) error {
	p, err := ProfilePath(name)
	if err != nil {
		return err
	}
	if err := os.Remove(p); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("no profile named %q", name)
		}
		return err
	}
	return nil
}

// listProfiles returns the saved profile names, sorted.
func ListProfiles() ([]string, error) {
	dir, err := profilesDir()
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var names []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		n := e.Name()
		if !strings.HasSuffix(n, ".piml") {
			continue
		}
		names = append(names, strings.TrimSuffix(n, ".piml"))
	}
	sort.Strings(names)
	return names, nil
}

// configsEqual reports whether two configs hold the same settings. Compared
// through their JSON encoding so pointer fields (temperature, seed, …) match
// by value rather than identity, and field order is irrelevant.
func ConfigsEqual(a, b Config) bool {
	ab, err1 := json.Marshal(a)
	bb, err2 := json.Marshal(b)
	if err1 != nil || err2 != nil {
		return false
	}
	return string(ab) == string(bb)
}

// --- Built-in presets ----------------------------------------------------

// presetProfile is one profile shipped inside the binary, written to the
// profiles directory by --install-profiles. Cfg is a func so presets that
// derive from the registry (lite) can never go stale in a static table.
type presetProfile struct {
	Name string
	Note string
	Cfg  func() Config
}

// The catalog, lightest first. Most entries were captured from a working
// 16GB-NVIDIA setup; they embed everything from that capture except
// engine_variant, which stays auto so the catalog is portable — auto
// resolves to CUDA on a machine that has it and to CPU elsewhere, while a
// pinned "cuda" would wedge a Mac.
func PresetProfiles() []presetProfile {
	return []presetProfile{
		{"lite", "lightest model, everything else auto — what --reset-model loads", LiteProfile},
		{"tiny", "gemma-3-1b-it, 4K context, tools off — minimum-footprint chat", tinyProfile},
		{"basic", "gemma-3-4b-it, 8K context, tools off — plain chat with a bit more brain", basicProfile},
		{"fast", "qwen3.5-9b, 16K context, reasoning off — quick tool-capable daily driver", fastProfile},
		{"coder", "qwen3-coder-30b-a3b, 16K context — MoE code model, 30B knowledge at 4B speed", coderProfile},
		{"quality", "qwen3.8-27b, 32K context, reasoning on — the strongest tool-caller", qualityProfile},
		{"heretic", "qwen3.8-27b-heretic, 32K context — quality's abliterated (uncensored) twin", hereticProfile},
		{"current", "qwen3.8-27b, 64K context with CPU-side KV — big-window agentic work on 16GB", currentProfile},
		{"tweet150k", "qwen3.8-27b-iq2 at 150K context via q4_0 KV, one slot", tweet150kProfile},
	}
}

func tinyProfile() Config {
	return Config{CurrentModel: "gemma-3-1b-it", MaxTokens: 2048, CtxSize: 4096}
}

func basicProfile() Config {
	return Config{CurrentModel: "gemma-3-4b-it", MaxTokens: 2048, CtxSize: 8192}
}

func fastProfile() Config {
	return Config{CurrentModel: "qwen3.5-9b", MaxTokens: 4096, CtxSize: 16384,
		Reasoning: ReasoningOff, ToolsEnabled: true, AMAEnabled: true}
}

func coderProfile() Config {
	return Config{CurrentModel: "qwen3-coder-30b-a3b", MaxTokens: 8192, CtxSize: 16384,
		Reasoning: ReasoningOff, ToolsEnabled: true, AMAEnabled: true}
}

func qualityProfile() Config {
	return Config{CurrentModel: "qwen3.8-27b", MaxTokens: 8192, CtxSize: 32768,
		Reasoning: ReasoningOn, ToolsEnabled: true, AMAEnabled: true}
}

// hereticProfile is qualityProfile pointed at the abliterated cut of the
// same 27B — the Heretic ARA build barely diverges from the base model, so
// every quality setting carries over unchanged. Kept as a separate preset
// rather than a /model switch because loading it is a statement of intent
// the profile name makes visible.
func hereticProfile() Config {
	cfg := qualityProfile()
	cfg.CurrentModel = "qwen3.8-27b-heretic"
	return cfg
}

// currentProfile keeps its capture-time name: it was the live setup the
// catalog was snapshotted around — the dense 27B given a 64K window by
// parking the KV cache in system RAM (kv_offload off) so every weight layer
// stays on a 16GB GPU.
func currentProfile() Config {
	return Config{CurrentModel: "qwen3.8-27b", MaxTokens: 16384, CtxSize: 65536,
		KVOffload: "off", ToolsEnabled: true, AMAEnabled: true}
}

// liteProfile is the escape-hatch preset: the lightest model in the registry
// and every other setting on auto. MaxTokens is set explicitly because
// loadConfig would normalise 0 to the default anyway — writing it keeps a
// loaded lite config comparable to this value with configsEqual.
func LiteProfile() Config {
	return Config{CurrentModel: LightestModel().Name, MaxTokens: DefaultMaxTokens}
}

// tweet150kProfile is the long-context setup: the 2-bit qwen3.8 with a 150K
// window in q4_0 KV on a single slot, sampling at the model's recommended
// temperature. One slot is load-bearing — -c is split across slots, so the
// default two would halve the window per conversation.
func tweet150kProfile() Config {
	temp := 1.0
	return Config{
		CurrentModel: "qwen3.8-27b-iq2",
		MaxTokens:    DefaultMaxTokens,
		Reasoning:    ReasoningOn,
		CtxSize:      150000,
		FlashAttn:    "on",
		CacheTypeK:   "q4_0",
		CacheTypeV:   "q4_0",
		Parallel:     1,
		Temperature:  &temp,
	}
}

// installProfiles writes the built-in presets into the profiles directory,
// reporting one line per preset to w. Existing files are never overwritten:
// a preset name the user has edited is their profile now, and a re-install
// must not undo that.
func InstallProfiles(w io.Writer) error {
	for _, p := range PresetProfiles() {
		path, err := ProfilePath(p.Name)
		if err != nil {
			return err
		}
		if _, err := os.Stat(path); err == nil {
			fmt.Fprintf(w, "%-11s skipped (already exists)\n", p.Name)
			continue
		}
		if err := SaveProfile(p.Name, p.Cfg()); err != nil {
			return err
		}
		fmt.Fprintf(w, "%-11s installed — %s\n", p.Name, p.Note)
	}
	fmt.Fprintln(w, "Load one inside chat with /config load <name>.")
	return nil
}

// resetToLiteProfile replaces the active config with the lite preset.
//
// atlas.llm warms the model server at startup, so quitting while a heavy
// setup is active means the next launch blocks loading it — with no chance
// to reach /model or /set from inside the TUI. A giant context or forced
// offload can wedge a launch as thoroughly as a giant model, which is why
// this loads a whole known-good profile rather than switching the model and
// keeping the rest. It uses the embedded preset, not the on-disk profile,
// so the escape hatch works even when the profiles directory is missing or
// broken.
func ResetToLiteProfile() error {
	cfg, err := LoadConfig()
	if err != nil {
		return err
	}
	lite := LiteProfile()
	if ConfigsEqual(cfg, lite) {
		fmt.Printf("Already on the lite profile (%s, %s).\n",
			lite.CurrentModel, LightestModel().Size)
		return nil
	}
	previous := cfg.CurrentModel
	if err := SaveConfig(lite); err != nil {
		return err
	}
	fmt.Printf("Config reset to the lite profile: %s -> %s (%s)\n",
		previous, lite.CurrentModel, LightestModel().Size)
	if m, ok := FindModel(lite.CurrentModel); ok && !IsModelDownloaded(m) {
		fmt.Printf("Note: %s is not downloaded yet — run /download %s inside chat.\n",
			m.Name, m.Name)
	}
	return nil
}

// matchingProfile returns the name of the saved profile whose settings equal
// cfg exactly, or "" if none does. Lets /config show which profile is active
// without tracking a name that could go stale after a manual /set.
func MatchingProfile(cfg Config) string {
	names, err := ListProfiles()
	if err != nil {
		return ""
	}
	for _, n := range names {
		if p, err := LoadProfile(n); err == nil && ConfigsEqual(p, cfg) {
			return n
		}
	}
	return ""
}
