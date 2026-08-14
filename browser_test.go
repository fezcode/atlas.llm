package main

import (
	"fmt"
	"os"
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
	cases := []map[string]any{
		{"action": "click"},                  // no selector
		{"action": "type"},                   // no selector
		{"action": "eval"},                   // no js
		{"action": "hover", "selector": "a"}, // unknown action
	}
	for _, args := range cases {
		if _, err := toolBrowserAct(args); err == nil {
			t.Errorf("toolBrowserAct(%v) should error", args)
		}
	}
}

func TestLaunchBrowserUnknownKind(t *testing.T) {
	if _, err := launchBrowser("safari"); err == nil || !strings.Contains(err.Error(), "safari") {
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
}

func (f *fakeSession) Kind() string                { return "chrome" }
func (f *fakeSession) Navigate(string) error       { return nil }
func (f *fakeSession) Eval(string) (string, error) { return f.evalResult, f.evalErr }
func (f *fakeSession) Close()                      { f.closed = true }

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
// they don't depend on the network.
const liveBrowserPage = `data:text/html,<title>atlas live test</title><form><input id="q" name="q"></form><a id="l" href="%23x">a link</a>`

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
	if out, err = s.Eval(typeJS("#q", "hello")); err != nil || !strings.Contains(out, "typed into") {
		t.Fatalf("type: got %q err=%v", out, err)
	}
	if out, err = s.Eval(`document.querySelector("#q").value`); err != nil || out != "hello" {
		t.Fatalf("typed value: got %q err=%v", out, err)
	}
	if out, err = s.Eval(clickJS("#l")); err != nil || !strings.Contains(out, "clicked") {
		t.Fatalf("click: got %q err=%v", out, err)
	}
	if out, err = readPage(s, "links"); err != nil || !strings.Contains(out, "a link") {
		t.Fatalf("links: got %q err=%v", out, err)
	}
}

func TestLiveChrome(t *testing.T) {
	testLiveSession(t, func() (browserSession, error) { return launchChrome() })
}

func TestLiveFirefox(t *testing.T) {
	testLiveSession(t, func() (browserSession, error) { return launchFirefox() })
}

func TestReadPageUsesEval(t *testing.T) {
	fake := &fakeSession{evalResult: "URL: https://example.com\ntitle: Example\n\nSome text"}
	out, err := readPage(fake, "text")
	if err != nil || !strings.Contains(out, "Example") {
		t.Fatalf("readPage: got %q err=%v", out, err)
	}
}
