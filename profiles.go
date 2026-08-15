package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// Named configs ("profiles"): full snapshots of config.json saved under a
// name, so a user can flip between whole setups — a fast one (small model,
// small context, no reasoning) and a quality one (big model, big context) —
// with a single /config load. A profile is just a copy of the active config
// file, so nothing about how settings are read changes: config.json remains
// the one active config, and load/save copy the whole thing.

// profileNameRE is the allowed shape of a profile name. It doubles as a
// filename, so no separators, dots, or spaces — only these, 1..32 chars.
var profileNameRE = regexp.MustCompile(`^[A-Za-z0-9_-]{1,32}$`)

func validProfileName(name string) error {
	if !profileNameRE.MatchString(name) {
		return fmt.Errorf("invalid profile name %q — use letters, digits, dash, or underscore (max 32)", name)
	}
	return nil
}

// profilesDir is where named configs live, created on demand.
func profilesDir() (string, error) {
	base, err := atlasDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(base, "profiles")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}
	return dir, nil
}

func profilePath(name string) (string, error) {
	if err := validProfileName(name); err != nil {
		return "", err
	}
	dir, err := profilesDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, name+".json"), nil
}

// saveProfile writes cfg to profiles/<name>.json. It never touches the active
// config.json — a save is a snapshot, not a switch.
func saveProfile(name string, cfg Config) error {
	p, err := profilePath(name)
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(p, data, 0644)
}

// loadProfile reads a saved profile. The returned Config is normalised the
// same way loadConfig normalises the active file, so a hand-edited or older
// profile still comes back with sane defaults.
func loadProfile(name string) (Config, error) {
	cfg := Config{CurrentModel: defaultModel}
	p, err := profilePath(name)
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
	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("profile %q is corrupt: %w", name, err)
	}
	if cfg.CurrentModel == "" {
		cfg.CurrentModel = defaultModel
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = defaultMaxTokens
	}
	return cfg, nil
}

func deleteProfile(name string) error {
	p, err := profilePath(name)
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
func listProfiles() ([]string, error) {
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
		if !strings.HasSuffix(n, ".json") {
			continue
		}
		names = append(names, strings.TrimSuffix(n, ".json"))
	}
	sort.Strings(names)
	return names, nil
}

// configsEqual reports whether two configs hold the same settings. Compared
// through their JSON encoding so pointer fields (temperature, seed, …) match
// by value rather than identity, and field order is irrelevant.
func configsEqual(a, b Config) bool {
	ab, err1 := json.Marshal(a)
	bb, err2 := json.Marshal(b)
	if err1 != nil || err2 != nil {
		return false
	}
	return string(ab) == string(bb)
}

// matchingProfile returns the name of the saved profile whose settings equal
// cfg exactly, or "" if none does. Lets /config show which profile is active
// without tracking a name that could go stale after a manual /set.
func matchingProfile(cfg Config) string {
	names, err := listProfiles()
	if err != nil {
		return ""
	}
	for _, n := range names {
		if p, err := loadProfile(n); err == nil && configsEqual(p, cfg) {
			return n
		}
	}
	return ""
}
