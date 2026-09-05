package browser

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// This file seeds a throwaway profile with a copy of the user's real browser
// profile, for browser_open profile="default". We always run on a copy, never
// the live profile: Chrome refuses remote debugging on its real data dir
// (since Chrome 136), Firefox locks a profile that's already open, and a copy
// means a bug or crash can never corrupt the user's actual profile. Caches
// and lock files are skipped so the copy is small and can't clash with a
// running instance.

// seedChromeDefaultProfile copies the user's real Chrome/Chromium/Edge/Brave
// profile into dst (a throwaway --user-data-dir). It brings "Local State"
// (needed to decrypt cookies) and the "Default" profile directory, minus
// caches and singleton locks.
func seedChromeDefaultProfile(exe, dst string) error {
	src, err := chromeUserDataDir(exe)
	if err != nil {
		return err
	}
	defaultDir := filepath.Join(src, "Default")
	if _, err := os.Stat(defaultDir); err != nil {
		return fmt.Errorf("could not find your %s profile at %s — open the browser normally once, "+
			"or use the fresh profile instead", filepath.Base(exe), defaultDir)
	}
	// Local State holds the profile's encryption key wrapper; without it,
	// stored cookies and passwords can't be read.
	if _, err := os.Stat(filepath.Join(src, "Local State")); err == nil {
		if err := copyFile(filepath.Join(src, "Local State"), filepath.Join(dst, "Local State")); err != nil {
			return fmt.Errorf("copy Local State: %w", err)
		}
	}
	if err := copyTree(defaultDir, filepath.Join(dst, "Default"), skipChromeProfileEntry); err != nil {
		return fmt.Errorf("copy Chrome profile: %w", err)
	}
	return nil
}

// seedFirefoxDefaultProfile copies the user's default Firefox profile into
// dst (a throwaway --profile dir), minus caches and lock files.
func seedFirefoxDefaultProfile(dst string) error {
	src, err := firefoxDefaultProfileDir()
	if err != nil {
		return err
	}
	if err := copyTree(src, dst, skipFirefoxProfileEntry); err != nil {
		return fmt.Errorf("copy Firefox profile: %w", err)
	}
	return nil
}

// prepPersistentProfile clears the transient files a previous run left in a
// persistent profile dir, so a relaunch starts clean without touching real
// profile data (cookies, Local State, history). Two classes matter:
//
//   - The browser's own port announcement — Chrome's DevToolsActivePort,
//     Firefox's WebDriverBiDiServer.json. Left in place, waitForBrowserFile
//     would read the *previous* port and connect to a dead endpoint.
//   - Singleton/lock files. After an unclean exit these linger and can make
//     the browser forward to nothing or refuse to open the profile.
//
// Everything else is deliberately preserved — that data is the feature.
// Best-effort: a file that can't be removed is left for the browser to sort
// out, which it usually does.
//
// Two more things only a reused profile needs: the caches it has accumulated
// since last time are dropped (nothing else ever prunes them, so they would
// grow without bound), and Chrome's crash flag is reset, since we always kill
// the browser and would otherwise show a "didn't shut down correctly" bubble
// on every single launch.
func prepPersistentProfile(dir string) {
	transient := []string{
		"DevToolsActivePort",                                  // Chrome debug port
		"WebDriverBiDiServer.json",                            // Firefox remote agent
		"SingletonLock", "SingletonCookie", "SingletonSocket", // Chrome
		"lock", ".parentlock", "parent.lock", // Firefox
	}
	for _, name := range transient {
		_ = os.Remove(filepath.Join(dir, name))
	}
	pruneProfileCaches(dir)
	clearChromeCrashFlag(dir)
}

// chromeUserDataDir returns the user-data directory for the Chromium-family
// browser at exe, inferring the flavour from the executable path.
func chromeUserDataDir(exe string) (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	low := strings.ToLower(exe)
	brave := strings.Contains(low, "brave")
	edge := strings.Contains(low, "edge") || strings.Contains(low, "msedge")
	chromium := strings.Contains(low, "chromium")

	switch runtime.GOOS {
	case "darwin":
		base := filepath.Join(home, "Library", "Application Support")
		switch {
		case brave:
			return filepath.Join(base, "BraveSoftware", "Brave-Browser"), nil
		case edge:
			return filepath.Join(base, "Microsoft Edge"), nil
		case chromium:
			return filepath.Join(base, "Chromium"), nil
		default:
			return filepath.Join(base, "Google", "Chrome"), nil
		}
	case "windows":
		lad := os.Getenv("LocalAppData")
		switch {
		case brave:
			return filepath.Join(lad, "BraveSoftware", "Brave-Browser", "User Data"), nil
		case edge:
			return filepath.Join(lad, "Microsoft", "Edge", "User Data"), nil
		case chromium:
			return filepath.Join(lad, "Chromium", "User Data"), nil
		default:
			return filepath.Join(lad, "Google", "Chrome", "User Data"), nil
		}
	default: // linux and friends
		cfg := filepath.Join(home, ".config")
		switch {
		case brave:
			return filepath.Join(cfg, "BraveSoftware", "Brave-Browser"), nil
		case edge:
			return filepath.Join(cfg, "microsoft-edge"), nil
		case chromium:
			return filepath.Join(cfg, "chromium"), nil
		default:
			return filepath.Join(cfg, "google-chrome"), nil
		}
	}
}

// firefoxRootDir is the directory holding profiles.ini and the Profiles/
// subtree.
func firefoxRootDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	switch runtime.GOOS {
	case "darwin":
		return filepath.Join(home, "Library", "Application Support", "Firefox"), nil
	case "windows":
		return filepath.Join(os.Getenv("AppData"), "Mozilla", "Firefox"), nil
	default:
		return filepath.Join(home, ".mozilla", "firefox"), nil
	}
}

// firefoxDefaultProfileDir resolves the absolute path of the user's default
// Firefox profile from profiles.ini.
func firefoxDefaultProfileDir() (string, error) {
	root, err := firefoxRootDir()
	if err != nil {
		return "", err
	}
	ini, err := os.ReadFile(filepath.Join(root, "profiles.ini"))
	if err != nil {
		return "", fmt.Errorf("could not read your Firefox profiles at %s — open Firefox normally once, "+
			"or use the fresh profile instead", filepath.Join(root, "profiles.ini"))
	}
	rel, absolute := parseFirefoxDefaultProfile(ini)
	if rel == "" {
		return "", fmt.Errorf("no default Firefox profile found in profiles.ini — use the fresh profile instead")
	}
	dir := rel
	if !absolute {
		dir = filepath.Join(root, filepath.FromSlash(rel))
	}
	if _, err := os.Stat(dir); err != nil {
		return "", fmt.Errorf("default Firefox profile %s does not exist — use the fresh profile instead", dir)
	}
	return dir, nil
}

// parseFirefoxDefaultProfile reads profiles.ini and returns the default
// profile's Path plus whether it is absolute (IsRelative=0). It prefers the
// [Install*] section's Default (the profile Firefox actually launches), then
// falls back to the [Profile*] entry flagged Default=1.
func parseFirefoxDefaultProfile(ini []byte) (path string, absolute bool) {
	type section struct {
		name   string
		values map[string]string
	}
	var sections []section
	var cur *section
	sc := bufio.NewScanner(bytes.NewReader(ini))
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, ";") {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			sections = append(sections, section{name: line[1 : len(line)-1], values: map[string]string{}})
			cur = &sections[len(sections)-1]
			continue
		}
		if cur == nil {
			continue
		}
		if k, v, ok := strings.Cut(line, "="); ok {
			cur.values[strings.TrimSpace(k)] = strings.TrimSpace(v)
		}
	}

	// An [Install*] section names the profile the current install opens.
	for _, s := range sections {
		if strings.HasPrefix(s.name, "Install") {
			if d := s.values["Default"]; d != "" {
				return d, filepath.IsAbs(filepath.FromSlash(d))
			}
		}
	}
	// Otherwise the [Profile*] entry flagged as default.
	for _, s := range sections {
		if strings.HasPrefix(s.name, "Profile") && s.values["Default"] == "1" {
			return s.values["Path"], s.values["IsRelative"] == "0"
		}
	}
	return "", false
}

// skipChromeProfileEntry drops caches and singleton locks when copying a
// Chrome profile: they bloat the copy and, in the case of Singleton* / locks,
// would fight a running instance.
func skipChromeProfileEntry(name string, isDir bool) bool {
	if isDir {
		switch name {
		case "Cache", "Code Cache", "GPUCache", "DawnCache", "DawnGraphiteCache",
			"DawnWebGPUCache", "GraphiteDawnCache", "GrShaderCache", "ShaderCache",
			"Service Worker", "Application Cache", "component_crx_cache",
			"extensions_crx_cache", "Crashpad", "blob_storage", "Download Service":
			return true
		}
		return false
	}
	switch name {
	case "SingletonLock", "SingletonCookie", "SingletonSocket", "lockfile", "LOCK":
		return true
	}
	return false
}

// skipFirefoxProfileEntry drops caches and the profile lock when copying a
// Firefox profile.
func skipFirefoxProfileEntry(name string, isDir bool) bool {
	if isDir {
		switch name {
		case "cache2", "startupCache", "shader-cache", "OfflineCache", "thumbnails":
			return true
		}
		return false
	}
	switch name {
	case "lock", ".parentlock", "parent.lock":
		return true
	}
	return false
}

// copyTree recursively copies src into dst (created if absent). skip decides,
// per entry name, whether to drop it; a skipped directory is not descended.
// Individual files that can't be read (e.g. momentarily locked by a running
// browser) are skipped rather than failing the whole copy — a best-effort
// profile clone is more useful than none.
func copyTree(src, dst string, skip func(name string, isDir bool) bool) error {
	info, err := os.Stat(src)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dst, info.Mode().Perm()|0700); err != nil {
		return err
	}
	entries, err := os.ReadDir(src)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if skip != nil && skip(e.Name(), e.IsDir()) {
			continue
		}
		srcPath := filepath.Join(src, e.Name())
		dstPath := filepath.Join(dst, e.Name())
		if e.IsDir() {
			if err := copyTree(srcPath, dstPath, skip); err != nil {
				return err
			}
			continue
		}
		// Symlinks and other non-regular files are skipped: a profile clone
		// needs the data, not the plumbing.
		if !e.Type().IsRegular() {
			continue
		}
		if err := copyFile(srcPath, dstPath); err != nil {
			continue
		}
	}
	return nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	info, err := in.Stat()
	if err != nil {
		return err
	}
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, info.Mode().Perm())
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}
