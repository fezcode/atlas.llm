package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// Browser control: the model can launch a real, visible Chrome or Firefox
// window and drive it (navigate, read the page, click, type). The window is
// deliberately headed so the user watches every step, and it runs on a
// throwaway profile so the model never touches the user's own cookies,
// sessions, or history.
//
// Chrome/Chromium/Edge speak the DevTools protocol (browser_cdp.go); Firefox
// speaks WebDriver BiDi (browser_bidi.go). Everything beyond "navigate" and
// "run this JS" — reading text, clicking, typing — is built on top of those
// two primitives, so each protocol backend stays small.

// browserSession is one live, driveable browser window.
type browserSession interface {
	// Kind is "chrome" or "firefox" (Chromium and Edge count as chrome).
	Kind() string
	// Navigate loads a URL in the active tab and waits for the page to load.
	Navigate(url string) error
	// Eval runs a JavaScript expression in the active tab and returns its
	// result rendered as a string.
	Eval(js string) (string, error)
	// Close shuts the browser down and removes its throwaway profile.
	Close()
}

// errBrowserGone marks transport failures — the user closed the window, the
// browser crashed. Tools translate it into "relaunch with browser_open".
var errBrowserGone = fmt.Errorf("browser is no longer running")

var (
	browserMu     sync.Mutex
	activeBrowser browserSession
)

// closeActiveBrowser tears down the launched browser (if any). Called on
// process exit alongside shutdownServer so a Ctrl+C never orphans a
// browser we started.
func closeActiveBrowser() {
	browserMu.Lock()
	defer browserMu.Unlock()
	if activeBrowser != nil {
		activeBrowser.Close()
		activeBrowser = nil
	}
}

// withBrowser runs fn against the active session. A transport failure
// (window closed by hand, crash) clears the session and tells the model how
// to recover instead of letting it retry into a dead socket.
func withBrowser(fn func(browserSession) (string, error)) (string, error) {
	browserMu.Lock()
	defer browserMu.Unlock()
	if activeBrowser == nil {
		return "", fmt.Errorf("no browser is running — call browser_open first")
	}
	out, err := fn(activeBrowser)
	if err != nil && strings.Contains(err.Error(), errBrowserGone.Error()) {
		activeBrowser.Close()
		activeBrowser = nil
		return "", fmt.Errorf("the browser window was closed — call browser_open to relaunch it")
	}
	return out, err
}

func init() {
	toolRegistry["browser_open"] = Tool{
		Name: "browser_open",
		Description: "Launch a visible Chrome or Firefox window that you can then drive with the other browser_* tools. " +
			"The user watches the window; it uses a fresh temporary profile with none of their logins. " +
			"If a browser is already open this just navigates it. Confirmation required.",
		Destructive: true,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"browser": map[string]any{
					"type":        "string",
					"enum":        []string{"chrome", "firefox"},
					"description": "Which browser to launch. Defaults to chrome (also matches Chromium/Edge), falling back to firefox if no Chrome is installed.",
				},
				"url": map[string]any{
					"type":        "string",
					"description": "Page to open right away. Optional; defaults to a blank tab.",
				},
			},
		},
		Run: toolBrowserOpen,
	}
	toolRegistry["browser_navigate"] = Tool{
		Name:        "browser_navigate",
		Description: "Navigate the open browser window to a URL, wait for the page to load, and return its title plus the beginning of its text.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"url": map[string]any{
					"type":        "string",
					"description": "Absolute URL, e.g. https://example.com. A bare domain gets https:// prepended.",
				},
			},
			"required": []string{"url"},
		},
		Run: toolBrowserNavigate,
	}
	toolRegistry["browser_read"] = Tool{
		Name:        "browser_read",
		Description: "Read the page currently shown in the browser: its visible text (default), its links, or its raw HTML.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"what": map[string]any{
					"type":        "string",
					"enum":        []string{"text", "links", "html"},
					"description": "What to return. 'text' is the visible page text, 'links' lists anchor text with hrefs (useful to decide where to click), 'html' is the raw markup. Defaults to text.",
				},
			},
		},
		Run: toolBrowserRead,
	}
	toolRegistry["browser_act"] = Tool{
		Name: "browser_act",
		Description: "Interact with the page in the open browser. action=click clicks the first element matching a CSS selector; " +
			"action=type focuses a matching input and types text into it; action=press sends a key (text holds the key name, e.g. Enter) to the focused element, submitting its form on Enter; " +
			"action=eval runs a JavaScript expression and returns its result. After an action that loads a new page, call browser_read to see it.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"action": map[string]any{
					"type": "string",
					"enum": []string{"click", "type", "press", "eval"},
				},
				"selector": map[string]any{
					"type":        "string",
					"description": "CSS selector for click/type, e.g. 'input[name=q]' or '#submit'. Use browser_read with what=links or what=html to find one.",
				},
				"text": map[string]any{
					"type":        "string",
					"description": "For type: the text to enter. For press: the key name (Enter, Escape, Tab, ArrowDown...); defaults to Enter.",
				},
				"js": map[string]any{
					"type":        "string",
					"description": "For eval: a JavaScript expression evaluated in the page.",
				},
			},
			"required": []string{"action"},
		},
		Run: toolBrowserAct,
	}
	toolRegistry["browser_close"] = Tool{
		Name:        "browser_close",
		Description: "Close the browser window opened with browser_open and discard its temporary profile.",
		Parameters:  map[string]any{"type": "object", "properties": map[string]any{}},
		Run:         toolBrowserClose,
	}
}

func toolBrowserOpen(args map[string]any) (string, error) {
	kind, _ := argString(args, "browser", false)
	url, _ := argString(args, "url", false)
	url = normalizeURL(url)

	browserMu.Lock()
	defer browserMu.Unlock()

	// Already running: reuse the window unless the model asked for the other
	// browser by name.
	if activeBrowser != nil {
		if kind != "" && kind != activeBrowser.Kind() {
			activeBrowser.Close()
			activeBrowser = nil
		} else {
			if url != "" {
				if err := activeBrowser.Navigate(url); err != nil {
					return "", err
				}
			}
			return fmt.Sprintf("Browser (%s) is already open. %s", activeBrowser.Kind(), pageSummary(activeBrowser)), nil
		}
	}

	sess, err := launchBrowser(kind)
	if err != nil {
		return "", err
	}
	if url != "" {
		if err := sess.Navigate(url); err != nil {
			sess.Close()
			return "", err
		}
	}
	activeBrowser = sess
	return fmt.Sprintf("Launched %s in a visible window with a fresh temporary profile. %s", sess.Kind(), pageSummary(sess)), nil
}

// launchBrowser starts the requested browser, or picks one: chrome first,
// firefox as fallback.
func launchBrowser(kind string) (browserSession, error) {
	switch kind {
	case "chrome":
		return launchChrome()
	case "firefox":
		return launchFirefox()
	case "":
		if _, err := chromeExecutable(); err == nil {
			return launchChrome()
		}
		if _, err := firefoxExecutable(); err == nil {
			return launchFirefox()
		}
		return nil, fmt.Errorf("no supported browser found — install Google Chrome, Chromium, Edge, or Firefox, " +
			"or point ATLAS_CHROME/ATLAS_FIREFOX at a browser binary")
	default:
		return nil, fmt.Errorf("unknown browser %q (expected chrome or firefox)", kind)
	}
}

func toolBrowserNavigate(args map[string]any) (string, error) {
	url, err := argString(args, "url", true)
	if err != nil {
		return "", err
	}
	url = normalizeURL(url)
	return withBrowser(func(s browserSession) (string, error) {
		if err := s.Navigate(url); err != nil {
			return "", err
		}
		return readPage(s, "text")
	})
}

func toolBrowserRead(args map[string]any) (string, error) {
	what, _ := argString(args, "what", false)
	if what == "" {
		what = "text"
	}
	if what != "text" && what != "links" && what != "html" {
		return "", fmt.Errorf("unknown what %q (expected text, links, or html)", what)
	}
	return withBrowser(func(s browserSession) (string, error) {
		return readPage(s, what)
	})
}

func toolBrowserAct(args map[string]any) (string, error) {
	action, err := argString(args, "action", true)
	if err != nil {
		return "", err
	}
	selector, _ := argString(args, "selector", false)
	text, _ := argString(args, "text", false)
	jsArg, _ := argString(args, "js", false)

	var js string
	switch action {
	case "click":
		if selector == "" {
			return "", fmt.Errorf("click needs a selector")
		}
		js = clickJS(selector)
	case "type":
		if selector == "" {
			return "", fmt.Errorf("type needs a selector")
		}
		js = typeJS(selector, text)
	case "press":
		if text == "" {
			text = "Enter"
		}
		js = pressJS(text)
	case "eval":
		if jsArg == "" {
			return "", fmt.Errorf("eval needs js")
		}
		js = jsArg
	default:
		return "", fmt.Errorf("unknown action %q (expected click, type, press, or eval)", action)
	}
	return withBrowser(func(s browserSession) (string, error) {
		out, err := s.Eval(js)
		if err != nil {
			return "", err
		}
		return truncateForModel(out), nil
	})
}

func toolBrowserClose(map[string]any) (string, error) {
	browserMu.Lock()
	defer browserMu.Unlock()
	if activeBrowser == nil {
		return "No browser is running.", nil
	}
	kind := activeBrowser.Kind()
	activeBrowser.Close()
	activeBrowser = nil
	return fmt.Sprintf("Closed the %s window and discarded its temporary profile.", kind), nil
}

// normalizeURL turns the bare domains models like to emit ("example.com")
// into fetchable URLs.
func normalizeURL(u string) string {
	u = strings.TrimSpace(u)
	if u == "" || strings.Contains(u, "://") || strings.HasPrefix(u, "about:") {
		return u
	}
	return "https://" + u
}

// jsStr renders s as a JavaScript string literal.
func jsStr(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

// browserPageTextCap bounds how much page content the in-page scripts hand
// back, below toolResultSizeLimit so the label/URL around it survive the
// final truncation.
const browserPageTextCap = 5 * 1024

// readPage returns the current URL and title plus the requested slice of
// the page. Built on Eval so it works identically over CDP and BiDi.
func readPage(s browserSession, what string) (string, error) {
	var body string
	switch what {
	case "links":
		body = `Array.from(document.querySelectorAll("a[href]")).slice(0, 100)
			.map(a => ((a.innerText || "").trim().replace(/\s+/g, " ").slice(0, 80) || "(no text)") + " -> " + a.href)
			.join("\n")`
	case "html":
		body = fmt.Sprintf(`document.documentElement.outerHTML.slice(0, %d)`, browserPageTextCap)
	default:
		body = fmt.Sprintf(`((document.body && document.body.innerText) || "").replace(/\n{3,}/g, "\n\n").slice(0, %d)`, browserPageTextCap)
	}
	js := fmt.Sprintf(`(() => { return "URL: " + location.href + "\ntitle: " + document.title + "\n\n" + (%s); })()`, body)
	out, err := s.Eval(js)
	if err != nil {
		return "", err
	}
	return truncateForModel(out), nil
}

// pageSummary is a best-effort "where are we now" line for launch/open
// results. Errors are swallowed — the launch itself already succeeded.
func pageSummary(s browserSession) string {
	out, err := s.Eval(`"Now at " + location.href + (document.title ? " — " + document.title : "")`)
	if err != nil {
		return "Now at a blank tab."
	}
	return out
}

func clickJS(selector string) string {
	return fmt.Sprintf(`(() => {
		const el = document.querySelector(%s);
		if (!el) return "no element matches that selector";
		el.scrollIntoView({block: "center"});
		el.click();
		return "clicked <" + el.tagName.toLowerCase() + "> " + ((el.innerText || el.value || "").trim().slice(0, 60));
	})()`, jsStr(selector))
}

// typeJS sets the value through the prototype setter so framework-controlled
// inputs (React et al.) see the change, then fires input/change.
func typeJS(selector, text string) string {
	return fmt.Sprintf(`(() => {
		const el = document.querySelector(%s);
		if (!el) return "no element matches that selector";
		el.focus();
		if (el.isContentEditable) {
			el.textContent = %s;
		} else {
			const d = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), "value");
			if (d && d.set) d.set.call(el, %s); else el.value = %s;
		}
		el.dispatchEvent(new Event("input", {bubbles: true}));
		el.dispatchEvent(new Event("change", {bubbles: true}));
		return "typed into <" + el.tagName.toLowerCase() + ">";
	})()`, jsStr(selector), jsStr(text), jsStr(text), jsStr(text))
}

// pressJS dispatches key events to the focused element. Synthetic events are
// untrusted, so browsers won't act on them natively — for the one case that
// matters most (Enter in a search box) we also submit the enclosing form.
func pressJS(key string) string {
	return fmt.Sprintf(`(() => {
		const key = %s;
		const el = document.activeElement || document.body;
		const opts = {key: key, bubbles: true, cancelable: true};
		const proceed = el.dispatchEvent(new KeyboardEvent("keydown", opts));
		el.dispatchEvent(new KeyboardEvent("keyup", opts));
		if (key === "Enter" && proceed && el.form) {
			if (el.form.requestSubmit) el.form.requestSubmit(); else el.form.submit();
			return "pressed Enter and submitted the form";
		}
		return "pressed " + key + " on <" + el.tagName.toLowerCase() + ">";
	})()`, jsStr(key))
}

// --- locating browser binaries ---------------------------------------------

// chromeExecutable finds a Chromium-family binary (they all speak CDP).
// ATLAS_CHROME overrides discovery.
func chromeExecutable() (string, error) {
	return findBrowserBinary("ATLAS_CHROME", chromeCandidates(), "Chrome/Chromium/Edge")
}

// firefoxExecutable finds Firefox. ATLAS_FIREFOX overrides discovery.
func firefoxExecutable() (string, error) {
	return findBrowserBinary("ATLAS_FIREFOX", firefoxCandidates(), "Firefox")
}

func findBrowserBinary(envKey string, candidates []string, label string) (string, error) {
	if p := os.Getenv(envKey); p != "" {
		if _, err := os.Stat(p); err != nil {
			return "", fmt.Errorf("%s=%q does not exist", envKey, p)
		}
		return p, nil
	}
	for _, c := range candidates {
		if filepath.IsAbs(c) {
			if _, err := os.Stat(c); err == nil {
				return c, nil
			}
			continue
		}
		if p, err := exec.LookPath(c); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("could not find %s (set %s to the browser binary to override)", label, envKey)
}

func chromeCandidates() []string {
	switch runtime.GOOS {
	case "darwin":
		home, _ := os.UserHomeDir()
		return []string{
			"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
			filepath.Join(home, "Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
			"/Applications/Chromium.app/Contents/MacOS/Chromium",
			"/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
			"/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
		}
	case "windows":
		var out []string
		for _, root := range []string{os.Getenv("ProgramFiles"), os.Getenv("ProgramFiles(x86)"), os.Getenv("LocalAppData")} {
			if root == "" {
				continue
			}
			out = append(out,
				filepath.Join(root, `Google\Chrome\Application\chrome.exe`),
				filepath.Join(root, `Chromium\Application\chrome.exe`),
				filepath.Join(root, `Microsoft\Edge\Application\msedge.exe`),
			)
		}
		return out
	default:
		return []string{"google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "brave-browser", "microsoft-edge"}
	}
}

func firefoxCandidates() []string {
	switch runtime.GOOS {
	case "darwin":
		home, _ := os.UserHomeDir()
		return []string{
			"/Applications/Firefox.app/Contents/MacOS/firefox",
			filepath.Join(home, "Applications/Firefox.app/Contents/MacOS/firefox"),
		}
	case "windows":
		var out []string
		for _, root := range []string{os.Getenv("ProgramFiles"), os.Getenv("ProgramFiles(x86)"), os.Getenv("LocalAppData")} {
			if root == "" {
				continue
			}
			out = append(out, filepath.Join(root, `Mozilla Firefox\firefox.exe`))
		}
		return out
	default:
		return []string{"firefox", "firefox-esr"}
	}
}

// --- shared launch helpers --------------------------------------------------

// browserTempProfile creates the throwaway profile directory a launched
// browser runs on. Lives under the OS temp dir, removed by Close.
func browserTempProfile(prefix string) (string, error) {
	dir, err := os.MkdirTemp("", prefix)
	if err != nil {
		return "", fmt.Errorf("create temp profile: %w", err)
	}
	return dir, nil
}

// waitForBrowserFile polls for a file the launching browser writes (Chrome's
// DevToolsActivePort, Firefox's WebDriverBiDiServer.json), giving up when
// the browser process dies or the deadline passes. exited is closed by the
// launcher's cmd.Wait goroutine.
func waitForBrowserFile(path string, exited <-chan struct{}, timeout time.Duration) ([]byte, error) {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if b, err := os.ReadFile(path); err == nil && len(b) > 0 {
			return b, nil
		}
		select {
		case <-exited:
			return nil, fmt.Errorf("browser exited during startup")
		case <-time.After(150 * time.Millisecond):
		}
	}
	return nil, fmt.Errorf("browser did not report its debugging port within %s", timeout)
}

// killAndCleanup force-stops the browser process and removes its profile.
// The graceful path (protocol-level close) happens before this in Close.
func killAndCleanup(cmd *exec.Cmd, exited <-chan struct{}, profile string) {
	if cmd != nil && cmd.Process != nil {
		select {
		case <-exited:
		case <-time.After(3 * time.Second):
			_ = cmd.Process.Kill()
			<-exited
		}
	}
	if profile != "" {
		// The browser may still be flushing the profile as it exits; retry
		// briefly rather than leaving the directory behind.
		for i := 0; i < 5; i++ {
			if err := os.RemoveAll(profile); err == nil {
				break
			}
			time.Sleep(200 * time.Millisecond)
		}
	}
}
