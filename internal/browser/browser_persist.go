package browser

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Everything that makes the persistent profile safe to reuse. A fresh or
// default profile is a brand-new temp dir every launch, so none of this
// applies to them: they cannot collide, they cannot accumulate, and they are
// deleted on close. The persistent one is a single fixed directory that
// outlives the process, which is what makes it useful and what makes it need
// a lock, a crash-flag reset, and a way to prune and clear it.

// profileLockName is the file atlas.llm drops in a persistent profile to claim
// it. Chrome's SingletonLock and Firefox's .parentlock already do this job for
// the browsers themselves, but prepPersistentProfile has to clear those — a
// stale one wedges the next launch — so the claim has to be one we own and can
// reason about.
const profileLockName = "atlas.lock"

// claimPersistentProfile takes ownership of a persistent profile directory. It
// fails when another live process already holds it: two browsers on one
// profile write to the same cookie, history and login stores, which SQLite is
// not going to survive. A lock whose pid is gone — the last run crashed, or
// was killed — is taken over, so the feature can never wedge permanently.
func claimPersistentProfile(dir string) error {
	lock := filepath.Join(dir, profileLockName)
	if b, err := os.ReadFile(lock); err == nil {
		if pid, err := strconv.Atoi(strings.TrimSpace(string(b))); err == nil && pidAlive(pid) {
			return fmt.Errorf("the persistent browser profile is already open in another "+
				"atlas.llm session (pid %d) — close that window, or open this one with "+
				`profile="fresh"`, pid)
		}
	}
	return os.WriteFile(lock, []byte(strconv.Itoa(os.Getpid())), 0600)
}

// releasePersistentProfile drops our claim on a persistent profile.
func releasePersistentProfile(dir string) {
	_ = os.Remove(filepath.Join(dir, profileLockName))
}

// profileInUse reports whether a live process holds the profile at dir.
func profileInUse(dir string) bool {
	b, err := os.ReadFile(filepath.Join(dir, profileLockName))
	if err != nil {
		return false
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(b)))
	return err == nil && pidAlive(pid)
}

// clearChromeCrashFlag records the last session as having exited normally.
// Close kills the browser once the graceful attempt times out, and Ctrl+C
// skips that attempt entirely, so Chrome reads the profile as crashed and
// greets the next launch with "Chrome didn't shut down correctly". Nobody ever
// sees that on a throwaway profile; on a persistent one it is every time. Only
// the two exit keys are touched, and a Preferences file we cannot parse is
// left exactly as it was.
func clearChromeCrashFlag(dir string) {
	path := filepath.Join(dir, "Default", "Preferences")
	raw, err := os.ReadFile(path)
	if err != nil {
		return
	}
	var prefs map[string]any
	if err := json.Unmarshal(raw, &prefs); err != nil {
		return
	}
	profile, ok := prefs["profile"].(map[string]any)
	if !ok {
		profile = map[string]any{}
	}
	profile["exit_type"] = "Normal"
	profile["exited_cleanly"] = true
	prefs["profile"] = profile
	out, err := json.Marshal(prefs)
	if err != nil {
		return
	}
	_ = os.WriteFile(path, out, 0600)
}

// pruneProfileCaches deletes the regenerable parts of a persistent profile.
// The skip lists already name every directory that is a cache — a throwaway
// profile simply never copies them — but the persistent one is never deleted,
// so without this it grows without bound inside the data dir. Cookies, logins,
// history and site storage are untouched: that data is the feature.
func pruneProfileCaches(dir string) {
	prune := func(base string) {
		entries, err := os.ReadDir(base)
		if err != nil {
			return
		}
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			if skipChromeProfileEntry(e.Name(), true) || skipFirefoxProfileEntry(e.Name(), true) {
				_ = os.RemoveAll(filepath.Join(base, e.Name()))
			}
		}
	}
	prune(dir)                           // Firefox's profile root, Chrome's user-data root
	prune(filepath.Join(dir, "Default")) // Chrome keeps the real profile a level down
}

// persistNeedsSeed reports whether a persistent profile is still empty and so
// can be seeded from the user's real one. Only the first launch seeds: copying
// over a profile that has accumulated sessions would throw away the very thing
// it was kept for. Our own lock file does not count as content.
func persistNeedsSeed(dir string) bool {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, e := range entries {
		if e.Name() != profileLockName {
			return false
		}
	}
	return true
}

// ProfileInfo describes one persistent browser profile on disk, for /browser.
type ProfileInfo struct {
	Family string    // "chrome" or "firefox"
	Dir    string    // where it lives
	Exists bool      // it has been launched at least once
	Size   int64     // bytes on disk
	Used   time.Time // newest modification anywhere inside it
	InUse  bool      // a live process holds the lock right now
}

// persistentProfileFamilies is every family that can own a stable profile.
// The two profile formats are incompatible, so they never share a directory.
var persistentProfileFamilies = []string{"chrome", "firefox"}

// PersistentProfiles reports what is on disk for each family. Unlike the
// launch path it never creates anything: a profile that has never been used
// comes back with Exists false.
func PersistentProfiles() []ProfileInfo {
	out := make([]ProfileInfo, 0, len(persistentProfileFamilies))
	for _, family := range persistentProfileFamilies {
		dir, err := persistentBrowserProfilePath(family)
		if err != nil {
			continue
		}
		info := ProfileInfo{Family: family, Dir: dir}
		if fi, err := os.Stat(dir); err == nil && fi.IsDir() {
			info.Exists = true
			info.Size, info.Used = profileDiskUsage(dir)
			info.InUse = profileInUse(dir)
		}
		out = append(out, info)
	}
	return out
}

// ClearPersistentProfile deletes a family's persistent profile, signing it out
// of everything. It refuses while a browser is running on it, since deleting
// the files out from under a live process corrupts what is left.
func ClearPersistentProfile(family string) error {
	known := false
	for _, f := range persistentProfileFamilies {
		known = known || f == family
	}
	if !known {
		return fmt.Errorf("unknown browser profile %q (expected %s)",
			family, strings.Join(persistentProfileFamilies, " or "))
	}
	dir, err := persistentBrowserProfilePath(family)
	if err != nil {
		return err
	}
	if profileInUse(dir) {
		return fmt.Errorf("the %s profile is open right now — close the browser first", family)
	}
	if err := os.RemoveAll(dir); err != nil {
		return fmt.Errorf("remove %s: %w", dir, err)
	}
	return nil
}

// profileDiskUsage totals a profile's files and finds the newest one, which is
// a good enough stand-in for "last used".
func profileDiskUsage(dir string) (size int64, newest time.Time) {
	_ = filepath.WalkDir(dir, func(_ string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil //nolint:nilerr // an unreadable corner should not abort the tally
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		size += info.Size()
		if info.ModTime().After(newest) {
			newest = info.ModTime()
		}
		return nil
	})
	return size, newest
}
