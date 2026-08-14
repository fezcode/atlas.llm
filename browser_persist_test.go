package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBrowserProfileModeParsing(t *testing.T) {
	cases := map[string]struct {
		mode    profileMode
		wantErr bool
	}{
		"":        {profileFresh, false}, // empty defaults to fresh
		"fresh":   {profileFresh, false},
		"default": {profileDefault, false},
		"persist": {profilePersist, false},
		"PERSIST": {profilePersist, false}, // case-insensitive
		"mine":    {profileFresh, true},
	}
	for in, want := range cases {
		got, err := browserProfileMode(in)
		if want.wantErr {
			if err == nil {
				t.Errorf("browserProfileMode(%q): expected an error", in)
			}
			continue
		}
		if err != nil {
			t.Errorf("browserProfileMode(%q): %v", in, err)
			continue
		}
		if got != want.mode {
			t.Errorf("browserProfileMode(%q) = %v, want %v", in, got, want.mode)
		}
	}
}

func TestPersistentProfileDirIsStableAndUnderData(t *testing.T) {
	base, err := atlasDir()
	if err != nil {
		t.Fatalf("atlasDir: %v", err)
	}
	dir, err := persistentBrowserProfileDir("unittest")
	if err != nil {
		t.Fatalf("persistentBrowserProfileDir: %v", err)
	}
	t.Cleanup(func() { os.RemoveAll(filepath.Join(base, "browser-profiles", "unittest")) })

	want := filepath.Join(base, "browser-profiles", "unittest")
	if dir != want {
		t.Errorf("persistent dir = %q, want %q", dir, want)
	}
	// It must be created, and stable across calls — that stability is the
	// whole feature: the cf_clearance cookie has to land in the same place
	// next launch.
	if fi, err := os.Stat(dir); err != nil || !fi.IsDir() {
		t.Errorf("persistent dir not created: %v", err)
	}
	again, _ := persistentBrowserProfileDir("unittest")
	if again != dir {
		t.Errorf("persistent dir not stable: %q then %q", dir, again)
	}
}

func TestResolveBrowserProfile(t *testing.T) {
	// fresh: a throwaway temp dir, a different one each call, marked ephemeral.
	d1, persist1, err := resolveBrowserProfile(profileFresh, "atlas-test-", "chrome")
	if err != nil {
		t.Fatalf("resolve fresh: %v", err)
	}
	defer os.RemoveAll(d1)
	if persist1 {
		t.Error("fresh profile reported as persistent")
	}
	if !strings.HasPrefix(filepath.Base(d1), "atlas-test-") {
		t.Errorf("fresh dir %q does not use the temp prefix", d1)
	}
	d2, _, _ := resolveBrowserProfile(profileFresh, "atlas-test-", "chrome")
	defer os.RemoveAll(d2)
	if d1 == d2 {
		t.Error("two fresh profiles share a directory")
	}

	// persist: the stable per-family dir, marked persistent.
	dp, persistP, err := resolveBrowserProfile(profilePersist, "atlas-test-", "chrome")
	if err != nil {
		t.Fatalf("resolve persist: %v", err)
	}
	if !persistP {
		t.Error("persist profile not reported as persistent")
	}
	want, _ := persistentBrowserProfileDir("chrome")
	if dp != want {
		t.Errorf("persist dir = %q, want %q", dp, want)
	}
}

func TestKillAndCleanupRespectsRemoveFlag(t *testing.T) {
	// remove=false leaves the directory — this is what keeps a persistent
	// profile alive across sessions.
	keep := t.TempDir()
	marker := filepath.Join(keep, "cf_clearance")
	if err := os.WriteFile(marker, []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}
	killAndCleanup(nil, nil, keep, false)
	if _, err := os.Stat(marker); err != nil {
		t.Errorf("remove=false deleted the profile: %v", err)
	}

	// remove=true still cleans up an ephemeral profile.
	gone := t.TempDir()
	killAndCleanup(nil, nil, gone, true)
	if _, err := os.Stat(gone); !os.IsNotExist(err) {
		t.Errorf("remove=true left the profile behind: %v", err)
	}
}

// A persistent profile keeps the browser's own port/lock files from the last
// run. Left in place they poison the next launch: waitForBrowserFile would
// read the *stale* DevToolsActivePort and connect to a dead port, and a
// lingering SingletonLock can make Chrome forward to nothing. prep clears
// exactly those.
func TestPrepPersistentProfileClearsStaleState(t *testing.T) {
	dir := t.TempDir()
	stale := map[string]bool{ // filename -> should be removed
		"DevToolsActivePort":     true,
		"SingletonLock":          true,
		"SingletonCookie":        true,
		"SingletonSocket":        true,
		"WebDriverBiDiServer.json": true,
		"lock":                   true,
		".parentlock":            true,
		"Cookies":                false, // real profile data must survive
		"Local State":            false,
	}
	for name := range stale {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0644); err != nil {
			t.Fatal(err)
		}
	}
	prepPersistentProfile(dir)
	for name, removed := range stale {
		_, err := os.Stat(filepath.Join(dir, name))
		if removed && !os.IsNotExist(err) {
			t.Errorf("%s was not cleared", name)
		}
		if !removed && err != nil {
			t.Errorf("%s (real profile data) was wrongly removed: %v", name, err)
		}
	}
}

// The browser_open tool must advertise persist as a choice, or the model
// never knows it exists.
func TestBrowserOpenAdvertisesPersist(t *testing.T) {
	tool, ok := toolRegistry["browser_open"]
	if !ok {
		t.Fatal("browser_open not registered")
	}
	props := tool.Parameters["properties"].(map[string]any)
	prof := props["profile"].(map[string]any)
	enum := prof["enum"].([]string)
	found := false
	for _, e := range enum {
		if e == "persist" {
			found = true
		}
	}
	if !found {
		t.Errorf("profile enum %v does not include persist", enum)
	}
}
