package browser

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// A persistent profile is the one profile two sessions can collide on: fresh
// and default each get their own temp dir, but persist is a single fixed
// directory. Chrome and Firefox guard it with their own lock files, and
// prepPersistentProfile deletes those — so the guard has to be ours.

func TestClaimPersistentProfileRefusesALiveHolder(t *testing.T) {
	dir := t.TempDir()
	if err := claimPersistentProfile(dir); err != nil {
		t.Fatalf("first claim: %v", err)
	}
	if err := claimPersistentProfile(dir); err == nil {
		releasePersistentProfile(dir)
		t.Fatal("a second claim succeeded while the profile was still held — " +
			"two browsers would write to one set of cookie and history stores")
	}
	releasePersistentProfile(dir)
	if err := claimPersistentProfile(dir); err != nil {
		t.Errorf("claim after release: %v", err)
	}
	releasePersistentProfile(dir)
}

// A crash leaves the lock behind. The next run must be able to take it over,
// or the feature wedges permanently with no way out but deleting a file the
// user does not know about.
func TestClaimPersistentProfileTakesOverAStaleLock(t *testing.T) {
	dir := t.TempDir()

	// A pid that has certainly exited: run the test binary with a filter that
	// matches nothing, then reap it.
	cmd := exec.Command(os.Args[0], "-test.run=NoSuchTestPleaseMatchNothing")
	if err := cmd.Start(); err != nil {
		t.Skipf("could not spawn a throwaway child: %v", err)
	}
	dead := cmd.Process.Pid
	_ = cmd.Wait()

	lock := filepath.Join(dir, profileLockName)
	if err := os.WriteFile(lock, []byte(strconv.Itoa(dead)), 0600); err != nil {
		t.Fatal(err)
	}
	if err := claimPersistentProfile(dir); err != nil {
		t.Errorf("a lock left by a dead process should be taken over: %v", err)
	}
	releasePersistentProfile(dir)
}

func TestClaimPersistentProfileIgnoresAGarbageLock(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, profileLockName), []byte("not a pid"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := claimPersistentProfile(dir); err != nil {
		t.Errorf("an unreadable lock should not wedge the profile: %v", err)
	}
	releasePersistentProfile(dir)
}

func TestPidAliveKnowsThisProcess(t *testing.T) {
	if !pidAlive(os.Getpid()) {
		t.Error("the running test process reported as dead")
	}
	if pidAlive(0) {
		t.Error("pid 0 reported alive")
	}
}

// We always kill the browser after a graceful attempt, and Ctrl+C skips the
// graceful path entirely. Chrome records that as a crash and greets the next
// launch with "Chrome didn't shut down correctly" — on a throwaway profile no
// one ever sees it, on a persistent one it is every single time.
func TestClearChromeCrashFlagRewritesExitTypeOnly(t *testing.T) {
	dir := t.TempDir()
	def := filepath.Join(dir, "Default")
	if err := os.MkdirAll(def, 0700); err != nil {
		t.Fatal(err)
	}
	prefs := `{"profile":{"exit_type":"Crashed","exited_cleanly":false,"name":"Person 1"},"session":{"restore_on_startup":4}}`
	if err := os.WriteFile(filepath.Join(def, "Preferences"), []byte(prefs), 0600); err != nil {
		t.Fatal(err)
	}

	clearChromeCrashFlag(dir)

	b, err := os.ReadFile(filepath.Join(def, "Preferences"))
	if err != nil {
		t.Fatalf("Preferences gone after the rewrite: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("Preferences is no longer valid JSON: %v", err)
	}
	prof, _ := got["profile"].(map[string]any)
	if prof["exit_type"] != "Normal" {
		t.Errorf("exit_type = %v, want Normal", prof["exit_type"])
	}
	if prof["exited_cleanly"] != true {
		t.Errorf("exited_cleanly = %v, want true", prof["exited_cleanly"])
	}
	if prof["name"] != "Person 1" {
		t.Errorf("rewrote an unrelated profile key: name = %v", prof["name"])
	}
	if _, ok := got["session"]; !ok {
		t.Error("dropped an unrelated top-level key")
	}
}

func TestClearChromeCrashFlagLeavesWhatItCannotParse(t *testing.T) {
	dir := t.TempDir()

	// No Preferences at all (a Firefox profile, or a first launch).
	clearChromeCrashFlag(dir)
	if _, err := os.Stat(filepath.Join(dir, "Default", "Preferences")); !os.IsNotExist(err) {
		t.Error("invented a Preferences file where there was none")
	}

	def := filepath.Join(dir, "Default")
	if err := os.MkdirAll(def, 0700); err != nil {
		t.Fatal(err)
	}
	bad := []byte("{ this is not json")
	if err := os.WriteFile(filepath.Join(def, "Preferences"), bad, 0600); err != nil {
		t.Fatal(err)
	}
	clearChromeCrashFlag(dir)
	got, err := os.ReadFile(filepath.Join(def, "Preferences"))
	if err != nil || string(got) != string(bad) {
		t.Errorf("mangled a Preferences file it could not parse: %q (%v)", got, err)
	}
}

// The persistent profile is the only one that is never deleted, so it is the
// only one whose caches accumulate forever. The skip lists already say which
// directories are disposable; prune reuses them on the live profile.
func TestPruneProfileCachesDropsCachesAndKeepsData(t *testing.T) {
	dir := t.TempDir()
	mk := func(parts ...string) string {
		p := filepath.Join(append([]string{dir}, parts...)...)
		if err := os.MkdirAll(p, 0700); err != nil {
			t.Fatal(err)
		}
		return p
	}
	put := func(p, name string) {
		if err := os.WriteFile(filepath.Join(p, name), []byte("x"), 0600); err != nil {
			t.Fatal(err)
		}
	}

	put(mk("Default", "Cache"), "data_0")           // Chrome, under Default/
	put(mk("Default", "Code Cache", "js"), "index") //
	put(mk("Default", "Service Worker"), "x")       //
	put(mk("GrShaderCache"), "x")                   // Chrome, at the root
	put(mk("cache2", "entries"), "x")               // Firefox, at the root
	put(mk("startupCache"), "x")                    //
	put(mk("Default"), "Cookies")                   // data that must survive
	put(mk("Default"), "Login Data")                //
	put(mk("Default", "IndexedDB"), "x")            //
	put(mk(), "Local State")                        //
	put(mk("storage", "default"), "x")              // Firefox site storage

	pruneProfileCaches(dir)

	for _, gone := range []string{
		"Default/Cache", "Default/Code Cache", "Default/Service Worker",
		"GrShaderCache", "cache2", "startupCache",
	} {
		if _, err := os.Stat(filepath.Join(dir, filepath.FromSlash(gone))); !os.IsNotExist(err) {
			t.Errorf("%s survived the prune", gone)
		}
	}
	for _, keep := range []string{
		"Default/Cookies", "Default/Login Data", "Default/IndexedDB",
		"Local State", "storage/default",
	} {
		if _, err := os.Stat(filepath.Join(dir, filepath.FromSlash(keep))); err != nil {
			t.Errorf("%s was pruned, and that is the user's data: %v", keep, err)
		}
	}
}

// persist starts empty, so "use my logins and remember them" is unreachable.
// Seeding happens on the first launch only — re-seeding later would overwrite
// the sessions the profile exists to accumulate.
func TestPersistNeedsSeedOnlyWhileTheProfileIsEmpty(t *testing.T) {
	dir := t.TempDir()
	if !persistNeedsSeed(dir) {
		t.Error("a brand-new persistent profile should be seeded")
	}

	if err := os.WriteFile(filepath.Join(dir, profileLockName), []byte("1"), 0600); err != nil {
		t.Fatal(err)
	}
	if !persistNeedsSeed(dir) {
		t.Error("our own lock file should not count as profile content")
	}

	if err := os.MkdirAll(filepath.Join(dir, "Default"), 0700); err != nil {
		t.Fatal(err)
	}
	if persistNeedsSeed(dir) {
		t.Error("re-seeding a profile that already has content would wipe the saved sessions")
	}
}

// browser_close always claimed it discarded the profile. For persist that is
// false, and it is the one moment the user wants to hear that it was kept.
func TestCloseMessageTellsTheTruthAboutTheProfile(t *testing.T) {
	ephemeral := closeMessage("chrome", "fresh")
	if !strings.Contains(ephemeral, "discarded") {
		t.Errorf("fresh close = %q, want it to say the profile was discarded", ephemeral)
	}
	persistent := closeMessage("chrome", "persist")
	if strings.Contains(persistent, "discarded") {
		t.Errorf("persist close = %q, but the profile is deliberately kept", persistent)
	}
	if !strings.Contains(persistent, "kept") {
		t.Errorf("persist close = %q, want it to say the profile was kept", persistent)
	}
}

// withTempHome points os.UserHomeDir at a scratch directory, so a test that
// writes or deletes a persistent profile can never touch the real one.
func withTempHome(t *testing.T) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home) // os.UserHomeDir on Windows
}

func TestPersistentProfilesReportsWhatIsOnDisk(t *testing.T) {
	withTempHome(t)
	dir, err := persistentBrowserProfileDir("chrome")
	if err != nil {
		t.Fatalf("persistentBrowserProfileDir: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	if err := os.WriteFile(filepath.Join(dir, "Cookies"), make([]byte, 4096), 0600); err != nil {
		t.Fatal(err)
	}

	var found *ProfileInfo
	for _, p := range PersistentProfiles() {
		if p.Family == "chrome" {
			cp := p
			found = &cp
		}
	}
	if found == nil {
		t.Fatal("chrome profile missing from PersistentProfiles")
	}
	if found.Dir != dir {
		t.Errorf("Dir = %q, want %q", found.Dir, dir)
	}
	if found.Size < 4096 {
		t.Errorf("Size = %d, want at least the 4096 bytes written", found.Size)
	}
	if found.InUse {
		t.Error("reported in use with no browser running")
	}
}

func TestClearPersistentProfileRefusesWhileInUse(t *testing.T) {
	withTempHome(t)
	dir, err := persistentBrowserProfileDir("chrome")
	if err != nil {
		t.Fatalf("persistentBrowserProfileDir: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	marker := filepath.Join(dir, "Cookies")
	if err := os.WriteFile(marker, []byte("session"), 0600); err != nil {
		t.Fatal(err)
	}

	if err := claimPersistentProfile(dir); err != nil {
		t.Fatalf("claim: %v", err)
	}
	if err := ClearPersistentProfile("chrome"); err == nil {
		releasePersistentProfile(dir)
		t.Fatal("cleared a profile that a live browser is running on")
	}
	releasePersistentProfile(dir)

	if err := ClearPersistentProfile("chrome"); err != nil {
		t.Fatalf("clear once released: %v", err)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Error("clear left the profile data behind")
	}
}

func TestClearPersistentProfileRejectsAnUnknownFamily(t *testing.T) {
	if err := ClearPersistentProfile("safari"); err == nil {
		t.Error("expected an error for a family we never create")
	}
}
