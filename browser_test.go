package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBrowserToolsRegistered(t *testing.T) {
	wantDestructive := map[string]bool{
		"browser_open":     true,
		"browser_navigate": false,
		"browser_read":     false,
		"browser_act":      false,
		"browser_close":    false,
	}
	for name, destructive := range wantDestructive {
		tool, ok := toolRegistry[name]
		if !ok {
			t.Fatalf("%s missing from toolRegistry", name)
		}
		if tool.Destructive != destructive {
			t.Errorf("%s: Destructive = %v, want %v", name, tool.Destructive, destructive)
		}
		if tool.Run == nil {
			t.Errorf("%s has no Run func", name)
		}
	}
}

func TestNormalizeURL(t *testing.T) {
	cases := map[string]string{
		"example.com":          "https://example.com",
		"https://example.com":  "https://example.com",
		"http://localhost:80":  "http://localhost:80",
		"about:blank":          "about:blank",
		"  example.com/a?b=c ": "https://example.com/a?b=c",
		"":                     "",
	}
	for in, want := range cases {
		if got := normalizeURL(in); got != want {
			t.Errorf("normalizeURL(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestJSStr(t *testing.T) {
	got := jsStr(`he said "hi" \ 'there'` + "\n")
	want := `"he said \"hi\" \\ 'there'\n"`
	if got != want {
		t.Errorf("jsStr = %s, want %s", got, want)
	}
}

func TestParseBiDiServerFile(t *testing.T) {
	host, port, err := parseBiDiServerFile([]byte(`{"ws_host":"127.0.0.1","ws_port":54321}`))
	if err != nil || host != "127.0.0.1" || port != 54321 {
		t.Fatalf("got %q %d err=%v", host, port, err)
	}
	// Missing host falls back to loopback.
	host, port, err = parseBiDiServerFile([]byte(`{"ws_port":9222}`))
	if err != nil || host != "127.0.0.1" || port != 9222 {
		t.Fatalf("fallback: got %q %d err=%v", host, port, err)
	}
	if _, _, err := parseBiDiServerFile([]byte(`{}`)); err == nil {
		t.Fatal("missing port should error")
	}
	if _, _, err := parseBiDiServerFile([]byte(`not json`)); err == nil {
		t.Fatal("garbage should error")
	}
}

func TestRawJSValueToString(t *testing.T) {
	if got := rawJSValueToString("string", []byte(`"hello"`), ""); got != "hello" {
		t.Errorf("string: got %q", got)
	}
	if got := rawJSValueToString("number", []byte(`42`), ""); got != "42" {
		t.Errorf("number: got %q", got)
	}
	if got := rawJSValueToString("object", []byte(`{"a":1}`), ""); got != `{"a":1}` {
		t.Errorf("object: got %q", got)
	}
	if got := rawJSValueToString("undefined", nil, ""); got != "(undefined)" {
		t.Errorf("undefined: got %q", got)
	}
	if got := rawJSValueToString("function", nil, "function f()"); got != "function f()" {
		t.Errorf("description: got %q", got)
	}
}

func TestBrowserToolsWithoutSession(t *testing.T) {
	browserMu.Lock()
	activeBrowser = nil
	browserMu.Unlock()

	for _, name := range []string{"browser_navigate", "browser_read", "browser_act"} {
		args := map[string]any{"url": "https://example.com", "action": "eval", "js": "1"}
		_, err := toolRegistry[name].Run(args)
		if err == nil || !strings.Contains(err.Error(), "browser_open") {
			t.Errorf("%s without a session should point at browser_open, got %v", name, err)
		}
	}
	// browser_close without a session is a no-op, not an error.
	out, err := toolRegistry["browser_close"].Run(map[string]any{})
	if err != nil || !strings.Contains(out, "No browser") {
		t.Errorf("browser_close: got %q err=%v", out, err)
	}
}

func TestBrowserActValidation(t *testing.T) {
	// These fail on argument validation before any browser is needed.
	cases := []map[string]any{
		{"action": "click"},                    // no text or selector
		{"action": "type", "text": "q"},        // missing value
		{"action": "type", "value": "hi"},      // missing target
		{"action": "select", "selector": "#s"}, // missing value
		{"action": "hover"},                    // no target
		{"action": "get"},                      // no target
		{"action": "clear"},                    // no target
		{"action": "wait"},                     // nothing to wait for
		{"action": "eval"},                     // no js
		{"action": "teleport"},                 // unknown action
	}
	for _, args := range cases {
		if _, err := toolBrowserAct(args); err == nil {
			t.Errorf("toolBrowserAct(%v) should error", args)
		}
	}
}

func TestDecodeActResult(t *testing.T) {
	out, err := decodeActResult(`{"ok":true,"msg":"clicked <a>"}`)
	if err != nil || !strings.Contains(out, "clicked") {
		t.Errorf("ok envelope: got %q err=%v", out, err)
	}
	// A miss (ok:false) must surface as a Go error carrying the guidance.
	if _, err = decodeActResult(`{"ok":false,"msg":"no clickable element found"}`); err == nil ||
		!strings.Contains(err.Error(), "no clickable element") {
		t.Errorf("miss should be an error, got %v", err)
	}
	// Non-envelope output is passed through rather than swallowed.
	if out, err = decodeActResult("raw string"); err != nil || out != "raw string" {
		t.Errorf("raw passthrough: got %q err=%v", out, err)
	}
}

// The action JS builders must embed the target and, on a miss, the directive
// hint — that hint is what stops the model repeating a failed call.
func TestActJSBuildersEmbedGuidance(t *testing.T) {
	if js := clickJS("", "Sign in"); !strings.Contains(js, "Sign in") ||
		!strings.Contains(js, "do not repeat this exact call") {
		t.Error("clickJS should embed the text and the do-not-repeat hint")
	}
	if js := typeJS("#q", "", "hello"); !strings.Contains(js, "hello") || !strings.Contains(js, "__findField") {
		t.Error("typeJS should embed the value and use the field finder")
	}
	if js := scrollJS("", "bottom"); !strings.Contains(js, "scrollHeight") {
		t.Error("scrollJS should handle the bottom direction")
	}
	if js := selectJS("", "Country", "France"); !strings.Contains(js, "France") {
		t.Error("selectJS should embed the option")
	}
}

func TestLaunchBrowserUnknownKind(t *testing.T) {
	if _, err := launchBrowser("safari", profileFresh); err == nil || !strings.Contains(err.Error(), "safari") {
		t.Fatalf("unknown kind should name itself in the error, got %v", err)
	}
}

func TestFindBrowserBinaryEnvOverride(t *testing.T) {
	t.Setenv("ATLAS_CHROME", "/definitely/not/a/real/path")
	if _, err := chromeExecutable(); err == nil || !strings.Contains(err.Error(), "ATLAS_CHROME") {
		t.Fatalf("bad env override should error mentioning ATLAS_CHROME, got %v", err)
	}
}

// fakeSession lets the session-plumbing tests run without a real browser.
type fakeSession struct {
	evalResult string
	evalErr    error
	closed     bool

	shot      []byte
	gotClip   *screenshotClip
	tabs      []browserTab
	switched  string
	openedURL string
	closedTab string
	files     []string
}

func (f *fakeSession) Kind() string                { return "chrome" }
func (f *fakeSession) Navigate(string) error       { return nil }
func (f *fakeSession) Eval(string) (string, error) { return f.evalResult, f.evalErr }
func (f *fakeSession) Close()                      { f.closed = true }
func (f *fakeSession) Screenshot(clip *screenshotClip) ([]byte, error) {
	f.gotClip = clip
	return f.shot, nil
}
func (f *fakeSession) Tabs() ([]browserTab, error) { return f.tabs, nil }
func (f *fakeSession) SwitchTab(id string) error   { f.switched = id; return nil }
func (f *fakeSession) NewTab(url string) error     { f.openedURL = url; return nil }
func (f *fakeSession) CloseTab(id string) error    { f.closedTab = id; return nil }
func (f *fakeSession) SetFiles(paths []string) error {
	f.files = paths
	return nil
}

// setFakeBrowser installs a fake session as the active browser and returns a
// cleanup that removes it again.
func setFakeBrowser(t *testing.T, f *fakeSession) {
	t.Helper()
	browserMu.Lock()
	activeBrowser = f
	browserMu.Unlock()
	t.Cleanup(func() {
		browserMu.Lock()
		activeBrowser = nil
		browserMu.Unlock()
	})
}

func TestWithBrowserClearsDeadSession(t *testing.T) {
	fake := &fakeSession{evalErr: fmt.Errorf("read: %w", errBrowserGone)}
	browserMu.Lock()
	activeBrowser = fake
	browserMu.Unlock()
	defer func() {
		browserMu.Lock()
		activeBrowser = nil
		browserMu.Unlock()
	}()

	_, err := withBrowser(func(s browserSession) (string, error) { return s.Eval("1") })
	if err == nil || !strings.Contains(err.Error(), "browser_open") {
		t.Fatalf("dead session should ask for a relaunch, got %v", err)
	}
	if !fake.closed {
		t.Fatal("dead session should be closed")
	}
	browserMu.Lock()
	gone := activeBrowser == nil
	browserMu.Unlock()
	if !gone {
		t.Fatal("dead session should be cleared")
	}
}

// liveBrowserPage is a self-contained page for the live tests below, so
// they don't depend on the network. It carries a labelled field, a
// button, a dropdown, and a link so every action has something to hit.
const liveBrowserPage = `data:text/html,<title>atlas live test</title>` +
	`<form><label>Search <input id="q" name="q"></label>` +
	`<select id="country"><option>France</option><option>Spain</option></select>` +
	`<button type="button">Sign in</button></form>` +
	`<a id="l" href="%23x">a link</a><p id="p">hello world</p>` +
	`<input type="file" id="up">`

// testLiveSession runs the full drive cycle — navigate, read, type, click —
// against a real browser window. Gated behind ATLAS_BROWSER_LIVE=1 because
// it opens a visible window on the developer's machine.
func testLiveSession(t *testing.T, launch func() (browserSession, error)) {
	if os.Getenv("ATLAS_BROWSER_LIVE") == "" {
		t.Skip("set ATLAS_BROWSER_LIVE=1 to run live browser tests")
	}
	s, err := launch()
	if err != nil {
		t.Fatalf("launch: %v", err)
	}
	defer s.Close()

	if err := s.Navigate(liveBrowserPage); err != nil {
		t.Fatalf("navigate: %v", err)
	}
	out, err := readPage(s, "text")
	if err != nil || !strings.Contains(out, "atlas live test") {
		t.Fatalf("readPage: got %q err=%v", out, err)
	}
	// Each action is exercised through the same envelope the tool decodes.
	okEval := func(label, js string) {
		t.Helper()
		out, err := s.Eval(js)
		if err != nil {
			t.Fatalf("%s: eval error: %v", label, err)
		}
		if res, derr := decodeActResult(out); derr != nil {
			t.Fatalf("%s: not ok: %v (raw %q)", label, derr, out)
		} else if res == "" {
			t.Fatalf("%s: empty ok message", label)
		}
	}

	// Type into a field located by its label text, not a selector.
	okEval("type by label", typeJS("", "Search", "hello"))
	if out, err = s.Eval(`document.querySelector("#q").value`); err != nil || out != "hello" {
		t.Fatalf("typed value: got %q err=%v", out, err)
	}
	okEval("clear", clearJS("", "Search"))
	if out, err = s.Eval(`document.querySelector("#q").value`); err != nil || out != "" {
		t.Fatalf("clear left value: got %q err=%v", out, err)
	}
	okEval("select option by text", selectJS("#country", "", "Spain"))
	if out, err = s.Eval(`document.querySelector("#country").value`); err != nil || out != "Spain" {
		t.Fatalf("select value: got %q err=%v", out, err)
	}
	okEval("hover button", hoverJS("", "Sign in"))
	okEval("get paragraph", getJS("#p", ""))
	okEval("scroll to bottom", scrollJS("", "bottom"))
	okEval("click by text", clickJS("", "a link"))

	// A miss must be ok:false so the tool layer turns it into an error.
	if out, err = s.Eval(clickJS("", "no such button")); err != nil || !strings.Contains(out, `"ok":false`) {
		t.Fatalf("click miss should be ok:false: got %q err=%v", out, err)
	}
	// wait finds text that is already present.
	if out, err = waitForPage(s, "", "hello world"); err != nil || !strings.Contains(out, "found") {
		t.Fatalf("wait for present text: got %q err=%v", out, err)
	}
	if out, err = readPage(s, "links"); err != nil || !strings.Contains(out, "a link") {
		t.Fatalf("links: got %q err=%v", out, err)
	}

	// The shim must capture console output and neuter alert — an unshimmed
	// alert would hang this Eval until its timeout.
	if _, err = s.Eval(`console.error("boom-live"); alert("hi there")`); err != nil {
		t.Fatalf("console/alert eval: %v", err)
	}
	if out, err = readPage(s, "console"); err != nil ||
		!strings.Contains(out, "boom-live") || !strings.Contains(out, "alert") {
		t.Fatalf("console read: got %q err=%v", out, err)
	}

	// Screenshots: full viewport, then one element via its measured rect.
	shot, err := s.Screenshot(nil)
	if err != nil {
		t.Fatalf("screenshot: %v", err)
	}
	if w, h, err := pngDimensions(shot); err != nil || w <= 0 || h <= 0 {
		t.Fatalf("screenshot dims: %dx%d err=%v", w, h, err)
	}
	out, err = s.Eval(elementRectJS("#p", ""))
	if err != nil {
		t.Fatalf("element rect: %v", err)
	}
	msg, derr := decodeActResult(out)
	if derr != nil {
		t.Fatalf("element rect envelope: %v", derr)
	}
	var rect struct{ X, Y, W, H float64 }
	if err := json.Unmarshal([]byte(msg), &rect); err != nil || rect.W <= 0 {
		t.Fatalf("element rect payload: %q err=%v", msg, err)
	}
	shot, err = s.Screenshot(&screenshotClip{X: rect.X, Y: rect.Y, Width: rect.W, Height: rect.H})
	if err != nil {
		t.Fatalf("element screenshot: %v", err)
	}
	if w, h, err := pngDimensions(shot); err != nil || w <= 0 || h <= 0 {
		t.Fatalf("element screenshot dims: %dx%d err=%v", w, h, err)
	}

	// Upload: mark the file input, attach a real temp file, and confirm the
	// page sees it.
	up := filepath.Join(t.TempDir(), "up.txt")
	if err := os.WriteFile(up, []byte("hello"), 0644); err != nil {
		t.Fatal(err)
	}
	okEval("mark upload target", markUploadTargetJS("#up", ""))
	if err := s.SetFiles([]string{up}); err != nil {
		t.Fatalf("SetFiles: %v", err)
	}
	if out, err = s.Eval(`String(document.querySelector("#up").files.length)`); err != nil || out != "1" {
		t.Fatalf("attached file count: got %q err=%v", out, err)
	}

	// Tabs: open a second one, switch back, close it.
	if err := s.NewTab(liveBrowserPage); err != nil {
		t.Fatalf("NewTab: %v", err)
	}
	tabs, err := s.Tabs()
	if err != nil || len(tabs) < 2 {
		t.Fatalf("tabs after open: %+v err=%v", tabs, err)
	}
	var active, other string
	for _, tb := range tabs {
		if tb.Active {
			active = tb.ID
		} else {
			other = tb.ID
		}
	}
	if active == "" || other == "" {
		t.Fatalf("expected one active and one background tab: %+v", tabs)
	}
	if err := s.SwitchTab(other); err != nil {
		t.Fatalf("SwitchTab: %v", err)
	}
	if err := s.CloseTab(active); err != nil {
		t.Fatalf("CloseTab: %v", err)
	}
	if tabs, err = s.Tabs(); err != nil || len(tabs) != 1 || !tabs[0].Active {
		t.Fatalf("tabs after close: %+v err=%v", tabs, err)
	}
}

func TestLiveChrome(t *testing.T) {
	testLiveSession(t, func() (browserSession, error) { return launchChrome(profileFresh) })
}

func TestLiveFirefox(t *testing.T) {
	testLiveSession(t, func() (browserSession, error) { return launchFirefox(profileFresh) })
}

// Live default-profile tests: gated by a second env var so they only run
// when someone explicitly wants a real profile copied. They just verify the
// session comes up on the seeded copy; they don't assert on any login.
func TestLiveChromeDefaultProfile(t *testing.T) {
	if os.Getenv("ATLAS_BROWSER_LIVE_DEFAULT") == "" {
		t.Skip("set ATLAS_BROWSER_LIVE_DEFAULT=1 to run default-profile live tests")
	}
	testLiveSession(t, func() (browserSession, error) { return launchChrome(profileDefault) })
}

func TestLiveFirefoxDefaultProfile(t *testing.T) {
	if os.Getenv("ATLAS_BROWSER_LIVE_DEFAULT") == "" {
		t.Skip("set ATLAS_BROWSER_LIVE_DEFAULT=1 to run default-profile live tests")
	}
	testLiveSession(t, func() (browserSession, error) { return launchFirefox(profileDefault) })
}

// Persist live test: the two integration facts unit tests can't cover — the
// profile dir survives Close (so state can accumulate), and a second launch
// on the leftover dir comes up cleanly (so prepPersistentProfile really does
// clear the stale port and lock files the first run left behind).
func TestLivePersistProfile(t *testing.T) {
	if os.Getenv("ATLAS_BROWSER_LIVE") == "" {
		t.Skip("set ATLAS_BROWSER_LIVE=1 to run live browser tests")
	}
	dir, err := persistentBrowserProfileDir("chrome")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("persistent profile at %s", dir)

	s1, err := launchChrome(profilePersist)
	if err != nil {
		t.Fatalf("first launch: %v", err)
	}
	s1.Close()
	if _, err := os.Stat(dir); err != nil {
		t.Fatalf("persistent profile was deleted on Close: %v", err)
	}

	s2, err := launchChrome(profilePersist)
	if err != nil {
		t.Fatalf("relaunch on the persisted dir failed — stale state not cleared? %v", err)
	}
	s2.Close()
}

func TestBrowserOpenRejectsUnknownProfile(t *testing.T) {
	browserMu.Lock()
	activeBrowser = nil
	browserMu.Unlock()
	if _, err := toolBrowserOpen(map[string]any{"profile": "mine"}); err == nil || !strings.Contains(err.Error(), "profile") {
		t.Fatalf("unknown profile should error naming the arg, got %v", err)
	}
}

func TestChromeUserDataDir(t *testing.T) {
	// Path-based flavour detection, checked by a lowercased token that appears
	// in that flavour's data dir on every OS (chrome/chromium/edge/brave) but
	// not in the others'.
	cases := map[string]string{
		"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome":   "chrome",
		"/opt/google/chrome/chrome":                                      "chrome",
		"/usr/bin/chromium-browser":                                      "chromium",
		"/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge": "edge",
		"/Applications/Brave Browser.app/Contents/MacOS/Brave Browser":   "brave",
	}
	for exe, want := range cases {
		dir, err := chromeUserDataDir(exe)
		if err != nil {
			t.Fatalf("chromeUserDataDir(%q): %v", exe, err)
		}
		if !strings.Contains(strings.ToLower(dir), want) {
			t.Errorf("chromeUserDataDir(%q) = %q, want it to contain %q", exe, dir, want)
		}
	}
}

func TestParseFirefoxDefaultProfile(t *testing.T) {
	// Install section wins over Profile flags.
	ini := []byte(`[Profile1]
Name=default
IsRelative=1
Path=Profiles/aaaa.default
Default=1

[Profile0]
Name=default-release
IsRelative=1
Path=Profiles/bbbb.default-release

[Install123]
Default=Profiles/bbbb.default-release
Locked=1
`)
	path, absolute := parseFirefoxDefaultProfile(ini)
	if path != "Profiles/bbbb.default-release" || absolute {
		t.Fatalf("install default: got %q absolute=%v", path, absolute)
	}

	// No Install section → fall back to the Default=1 profile.
	ini = []byte("[Profile0]\nIsRelative=1\nPath=Profiles/only.default\nDefault=1\n")
	if path, absolute = parseFirefoxDefaultProfile(ini); path != "Profiles/only.default" || absolute {
		t.Fatalf("profile default: got %q absolute=%v", path, absolute)
	}

	// Absolute path is reported as such.
	ini = []byte("[Profile0]\nIsRelative=0\nPath=/custom/place\nDefault=1\n")
	if path, absolute = parseFirefoxDefaultProfile(ini); path != "/custom/place" || !absolute {
		t.Fatalf("absolute: got %q absolute=%v", path, absolute)
	}

	// Nothing marked default.
	if path, _ = parseFirefoxDefaultProfile([]byte("[General]\nVersion=2\n")); path != "" {
		t.Fatalf("no default should be empty, got %q", path)
	}
}

func TestProfileSkipPredicates(t *testing.T) {
	if !skipChromeProfileEntry("Cache", true) || !skipChromeProfileEntry("SingletonLock", false) {
		t.Error("chrome skip should drop caches and singleton locks")
	}
	if skipChromeProfileEntry("Cookies", false) || skipChromeProfileEntry("Login Data", false) {
		t.Error("chrome skip must keep cookies and login data")
	}
	if !skipFirefoxProfileEntry("cache2", true) || !skipFirefoxProfileEntry(".parentlock", false) {
		t.Error("firefox skip should drop cache and lock")
	}
	if skipFirefoxProfileEntry("cookies.sqlite", false) || skipFirefoxProfileEntry("logins.json", false) {
		t.Error("firefox skip must keep cookies and logins")
	}
}

func TestCopyTreeFiltersAndCopies(t *testing.T) {
	src := t.TempDir()
	dst := t.TempDir()
	mustWriteTree(t, filepath.Join(src, "Cookies"), "keepme")
	mustWriteTree(t, filepath.Join(src, "Cache", "big.bin"), "dropme")
	mustWriteTree(t, filepath.Join(src, "sub", "Login Data"), "nested")

	if err := copyTree(src, dst, skipChromeProfileEntry); err != nil {
		t.Fatalf("copyTree: %v", err)
	}
	if b, err := os.ReadFile(filepath.Join(dst, "Cookies")); err != nil || string(b) != "keepme" {
		t.Errorf("Cookies not copied: %q err=%v", b, err)
	}
	if b, err := os.ReadFile(filepath.Join(dst, "sub", "Login Data")); err != nil || string(b) != "nested" {
		t.Errorf("nested file not copied: %q err=%v", b, err)
	}
	if _, err := os.Stat(filepath.Join(dst, "Cache")); !os.IsNotExist(err) {
		t.Error("Cache directory should have been skipped")
	}
}

func mustWriteTree(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
}

func TestReadPageUsesEval(t *testing.T) {
	fake := &fakeSession{evalResult: "URL: https://example.com\ntitle: Example\n\nSome text"}
	out, err := readPage(fake, "text")
	if err != nil || !strings.Contains(out, "Example") {
		t.Fatalf("readPage: got %q err=%v", out, err)
	}
}

// --- new capabilities: screenshot, tabs, upload, console, overlay ------------

func TestNewBrowserToolsRegistered(t *testing.T) {
	wantDestructive := map[string]bool{
		"browser_screenshot": false,
		"browser_tabs":       false,
		"browser_upload":     true, // sends a local file to a page — outbound, like web_fetch
	}
	for name, destructive := range wantDestructive {
		tool, ok := toolRegistry[name]
		if !ok {
			t.Fatalf("%s missing from toolRegistry", name)
		}
		if tool.Destructive != destructive {
			t.Errorf("%s: Destructive = %v, want %v", name, tool.Destructive, destructive)
		}
		if tool.Run == nil {
			t.Errorf("%s has no Run func", name)
		}
	}
}

func TestBrowserUploadValidation(t *testing.T) {
	if _, err := toolBrowserUpload(map[string]any{"file": "x.txt"}); err == nil {
		t.Error("upload without a target should error")
	}
	if _, err := toolBrowserUpload(map[string]any{"selector": "#f"}); err == nil {
		t.Error("upload without file should error")
	}
	// A path that doesn't exist is rejected by name, before touching the browser.
	missing := filepath.Join(t.TempDir(), "missing.bin")
	if _, err := toolBrowserUpload(map[string]any{"selector": "#f", "file": missing}); err == nil ||
		!strings.Contains(err.Error(), "missing.bin") {
		t.Errorf("nonexistent file should error naming it, got %v", err)
	}
}

func TestBrowserUploadCallsSetFiles(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "up-*.txt")
	if err != nil {
		t.Fatal(err)
	}
	f.Close()
	fake := &fakeSession{evalResult: `{"ok":true,"msg":"marked <input>"}`}
	setFakeBrowser(t, fake)
	out, err := toolBrowserUpload(map[string]any{"selector": "#f", "file": f.Name()})
	if err != nil {
		t.Fatalf("upload: %v", err)
	}
	if len(fake.files) != 1 || fake.files[0] != f.Name() {
		t.Fatalf("SetFiles got %v, want [%s]", fake.files, f.Name())
	}
	if !strings.Contains(out, "1 file") {
		t.Errorf("result should mention the file count, got %q", out)
	}
}

func TestBrowserTabsValidation(t *testing.T) {
	for _, args := range []map[string]any{
		{},                       // no action
		{"action": "levitate"},   // unknown action
		{"action": "switch"},     // switch needs a tab number
	} {
		if _, err := toolBrowserTabs(args); err == nil {
			t.Errorf("toolBrowserTabs(%v) should error", args)
		}
	}
}

func TestBrowserTabsListSwitchClose(t *testing.T) {
	fake := &fakeSession{tabs: []browserTab{
		{ID: "A", Title: "One", URL: "https://one.test", Active: true},
		{ID: "B", Title: "Two", URL: "https://two.test"},
	}}
	setFakeBrowser(t, fake)

	out, err := toolBrowserTabs(map[string]any{"action": "list"})
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if !strings.Contains(out, "1.") || !strings.Contains(out, "2.") ||
		!strings.Contains(out, "One") || !strings.Contains(out, "active") {
		t.Errorf("list should number tabs and mark the active one, got %q", out)
	}

	if _, err := toolBrowserTabs(map[string]any{"action": "switch", "tab": float64(2)}); err != nil {
		t.Fatalf("switch: %v", err)
	}
	if fake.switched != "B" {
		t.Errorf("switch tab=2 should target B, got %q", fake.switched)
	}
	if _, err := toolBrowserTabs(map[string]any{"action": "switch", "tab": float64(9)}); err == nil {
		t.Error("out-of-range tab should error")
	}

	// Closing the only remaining tab must refuse and point at browser_close.
	fake.tabs = fake.tabs[:1]
	if _, err := toolBrowserTabs(map[string]any{"action": "close"}); err == nil ||
		!strings.Contains(err.Error(), "browser_close") {
		t.Errorf("closing the last tab should point at browser_close, got %v", err)
	}
}

func TestBrowserReadConsole(t *testing.T) {
	if _, err := toolBrowserRead(map[string]any{"what": "bogus"}); err == nil ||
		!strings.Contains(err.Error(), "console") {
		t.Errorf("invalid what should list console as an option, got %v", err)
	}
	fake := &fakeSession{evalResult: "URL: x\ntitle: y\n\n[error] boom"}
	setFakeBrowser(t, fake)
	out, err := toolBrowserRead(map[string]any{"what": "console"})
	if err != nil || !strings.Contains(out, "[error] boom") {
		t.Fatalf("console read: got %q err=%v", out, err)
	}
}

func TestPNGDimensions(t *testing.T) {
	shot := fakePNG(640, 480)
	w, h, err := pngDimensions(shot)
	if err != nil || w != 640 || h != 480 {
		t.Fatalf("got %dx%d err=%v", w, h, err)
	}
	if _, _, err := pngDimensions([]byte("not a png")); err == nil {
		t.Error("garbage should error")
	}
}

// fakePNG builds just enough of a PNG header for pngDimensions.
func fakePNG(w, h int) []byte {
	b := make([]byte, 24)
	copy(b, "\x89PNG\r\n\x1a\n")
	binary.BigEndian.PutUint32(b[16:20], uint32(w))
	binary.BigEndian.PutUint32(b[20:24], uint32(h))
	return b
}

func TestScreenshotPath(t *testing.T) {
	p, err := screenshotPath("")
	if err != nil || !strings.Contains(p, "atlas-shot-") || !strings.HasSuffix(p, ".png") {
		t.Fatalf("default path: got %q err=%v", p, err)
	}
	// An explicit path that already exists is refused, never overwritten.
	existing := filepath.Join(t.TempDir(), "have.png")
	if err := os.WriteFile(existing, []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := screenshotPath(existing); err == nil {
		t.Error("existing file should be refused")
	}
	// A missing .png extension is added.
	if p, err = screenshotPath(filepath.Join(t.TempDir(), "shot")); err != nil || !strings.HasSuffix(p, ".png") {
		t.Errorf("extension: got %q err=%v", p, err)
	}
}

func TestToolBrowserScreenshotWritesFile(t *testing.T) {
	fake := &fakeSession{shot: fakePNG(4, 5)}
	setFakeBrowser(t, fake)
	dest := filepath.Join(t.TempDir(), "out.png")
	out, err := toolBrowserScreenshot(map[string]any{"file": dest})
	if err != nil {
		t.Fatalf("screenshot: %v", err)
	}
	if b, err := os.ReadFile(dest); err != nil || len(b) != 24 {
		t.Fatalf("file not written: %v", err)
	}
	if !strings.Contains(out, "4x5") || !strings.Contains(out, "out.png") {
		t.Errorf("result should mention dimensions and path, got %q", out)
	}
	if fake.gotClip != nil {
		t.Error("full-page shot should pass a nil clip")
	}
}

func TestToolBrowserScreenshotElementClip(t *testing.T) {
	fake := &fakeSession{
		shot:       fakePNG(10, 20),
		evalResult: `{"ok":true,"msg":"{\"x\":1,\"y\":2,\"w\":10,\"h\":20}"}`,
	}
	setFakeBrowser(t, fake)
	dest := filepath.Join(t.TempDir(), "el.png")
	if _, err := toolBrowserScreenshot(map[string]any{"file": dest, "text": "Sign in"}); err != nil {
		t.Fatalf("element screenshot: %v", err)
	}
	if fake.gotClip == nil || fake.gotClip.X != 1 || fake.gotClip.Y != 2 ||
		fake.gotClip.Width != 10 || fake.gotClip.Height != 20 {
		t.Fatalf("clip = %+v, want {1 2 10 20}", fake.gotClip)
	}
}

func TestParseCDPTargets(t *testing.T) {
	list := []byte(`[
		{"id":"T1","type":"page","title":"One","url":"https://one.test","webSocketDebuggerUrl":"ws://x/1"},
		{"id":"BG","type":"background_page","title":"ext","url":"chrome-extension://x","webSocketDebuggerUrl":"ws://x/bg"},
		{"id":"T2","type":"page","title":"Two","url":"https://two.test","webSocketDebuggerUrl":"ws://x/2"}
	]`)
	targets, err := parseCDPTargets(list)
	if err != nil {
		t.Fatal(err)
	}
	if len(targets) != 2 || targets[0].ID != "T1" || targets[1].Title != "Two" {
		t.Fatalf("got %+v", targets)
	}
	if targets[0].wsURL != "ws://x/1" {
		t.Errorf("wsURL not kept: %+v", targets[0])
	}
	if _, err := parseCDPTargets([]byte("nope")); err == nil {
		t.Error("garbage should error")
	}
}

// Element-targeted actions glide the fake cursor to the target and highlight
// it before acting; press has no element target so it must not animate.
func TestActJSBuildersAnimate(t *testing.T) {
	// The helpers (and thus the __animateTo definition) ride along in every
	// script, so what distinguishes an animating action is the await call.
	for name, js := range map[string]string{
		"click":  clickJS("", "Sign in"),
		"type":   typeJS("", "Search", "hi"),
		"hover":  hoverJS("", "Menu"),
		"select": selectJS("", "Country", "France"),
		"clear":  clearJS("", "Search"),
	} {
		if !strings.Contains(js, "await __animateTo") || !strings.Contains(js, "async") {
			t.Errorf("%sJS should animate the cursor to its target", name)
		}
	}
	if js := pressJS("Enter"); strings.Contains(js, "await __animateTo") {
		t.Error("pressJS has no element target and should not animate")
	}
}

func TestElementRectJS(t *testing.T) {
	js := elementRectJS("", "Sign in")
	if !strings.Contains(js, "Sign in") || !strings.Contains(js, "getBoundingClientRect") ||
		!strings.Contains(js, "__err") {
		t.Error("elementRectJS should embed the target, measure it, and use the envelope")
	}
}

func TestMarkUploadTargetJS(t *testing.T) {
	js := markUploadTargetJS("", "Resume")
	if !strings.Contains(js, "data-atlas-upload") || !strings.Contains(js, "file") ||
		!strings.Contains(js, "Resume") {
		t.Error("markUploadTargetJS should tag a file input by its visible text")
	}
}

// The shim is injected into every page: it must capture console output and
// errors, and neuter blocking dialogs so an alert() can never hang Eval.
func TestBrowserShim(t *testing.T) {
	for _, want := range []string{"__atlasLog", "window.alert", "window.confirm", "window.prompt", "unhandledrejection"} {
		if !strings.Contains(browserShimBody, want) {
			t.Errorf("browserShimBody should contain %q", want)
		}
	}
	if !strings.Contains(consoleLogJS(), "__atlasLog") {
		t.Error("consoleLogJS should read the shim's buffer")
	}
}
