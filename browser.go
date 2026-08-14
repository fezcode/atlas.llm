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
	// activeBrowserProfile records how the open window was launched
	// ("fresh" or "default"), so a browser_open asking for the other kind
	// relaunches instead of silently reusing the wrong profile.
	activeBrowserProfile string
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
		activeBrowserProfile = ""
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
		activeBrowserProfile = ""
		return "", fmt.Errorf("the browser window was closed — call browser_open to relaunch it")
	}
	return out, err
}

func init() {
	toolRegistry["browser_open"] = Tool{
		Name: "browser_open",
		Description: "Launch a visible Chrome or Firefox window that you can then drive with the other browser_* tools. " +
			"The user watches the window. By default it uses a fresh temporary profile with none of their logins. " +
			"Set profile=\"default\" only when the user explicitly asks to use their own/logged-in profile: that copies " +
			"their real browser profile so their existing logins are available. " +
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
				"profile": map[string]any{
					"type": "string",
					"enum": []string{"fresh", "default"},
					"description": "Which profile to launch on. 'fresh' (default) is an empty throwaway profile with no logins. " +
						"'default' copies the user's real browser profile so their existing logins and cookies are available — " +
						"use it only when the user explicitly asks to browse as themselves / signed in.",
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
		Description: "Interact with the page in the open browser. Set action to one of:\n" +
			"- click: click a link/button/element — target by visible text (text=\"Sign in\") or a CSS selector.\n" +
			"- type: enter text into a field (value=\"hello\"), locating it by label/placeholder text (text=\"Search\") or a selector.\n" +
			"- press: send a key (key=\"Enter\") to the focused element; Enter submits its form.\n" +
			"- hover: move the mouse over an element (text or selector) — reveals hover menus.\n" +
			"- select: choose an option in a dropdown (text/selector = the <select>, value = the option to choose).\n" +
			"- clear: empty a field (text or selector).\n" +
			"- get: return the text, value, and link of an element (text or selector).\n" +
			"- scroll: scroll to an element (text or selector), or text=\"top\"/\"bottom\"/\"up\"/\"down\".\n" +
			"- wait: wait until text appears on the page, or a selector matches (up to ~10s) — use after a click that loads content.\n" +
			"- back / forward / reload: browser history and refresh.\n" +
			"- eval: run a JavaScript expression and return its result.\n" +
			"Prefer text over CSS selectors — it matches what the user sees. After an action that loads a new page, call browser_read to see it.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"action": map[string]any{
					"type": "string",
					"enum": []string{
						"click", "type", "press", "hover", "select", "clear",
						"get", "scroll", "wait", "back", "forward", "reload", "eval",
					},
				},
				"text": map[string]any{
					"type": "string",
					"description": "The visible text that locates the target element — link/button text for click/hover, a field's label or placeholder for type/clear/select/get, or the text to wait for. " +
						"For scroll it may instead be a direction: top, bottom, up, or down. Matched case-insensitively, exact match preferred over a substring.",
				},
				"value": map[string]any{
					"type":        "string",
					"description": "For type: the text to enter. For select: the option label or value to choose.",
				},
				"selector": map[string]any{
					"type":        "string",
					"description": "Optional CSS selector instead of text, e.g. 'input[name=q]' or '#submit'. Use text when you can; fall back to this for elements with no clear visible text.",
				},
				"key": map[string]any{
					"type":        "string",
					"description": "For press: the key name — Enter, Escape, Tab, ArrowDown, etc. Defaults to Enter.",
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

	profile, _ := argString(args, "profile", false)
	profileSet := profile != ""
	if profile == "" {
		profile = "fresh"
	}
	if profile != "fresh" && profile != "default" {
		return "", fmt.Errorf("unknown profile %q (expected fresh or default)", profile)
	}
	useDefault := profile == "default"

	browserMu.Lock()
	defer browserMu.Unlock()

	// Already running: reuse the window unless the model asked for a
	// different browser, or explicitly asked for the other profile mode.
	if activeBrowser != nil {
		kindMismatch := kind != "" && kind != activeBrowser.Kind()
		profileMismatch := profileSet && profile != activeBrowserProfile
		if kindMismatch || profileMismatch {
			activeBrowser.Close()
			activeBrowser = nil
			activeBrowserProfile = ""
		} else {
			if url != "" {
				if err := activeBrowser.Navigate(url); err != nil {
					return "", err
				}
			}
			return fmt.Sprintf("Browser (%s) is already open. %s", activeBrowser.Kind(), pageSummary(activeBrowser)), nil
		}
	}

	sess, err := launchBrowser(kind, useDefault)
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
	activeBrowserProfile = profile

	profileNote := "a fresh temporary profile"
	if useDefault {
		profileNote = "a copy of your default " + sess.Kind() + " profile — your existing logins are available, " +
			"but changes are not written back to your real profile"
	}
	return fmt.Sprintf("Launched %s in a visible window with %s. %s", sess.Kind(), profileNote, pageSummary(sess)), nil
}

// launchBrowser starts the requested browser, or picks one: chrome first,
// firefox as fallback. useDefault copies the user's real profile instead of
// starting from an empty one.
func launchBrowser(kind string, useDefault bool) (browserSession, error) {
	switch kind {
	case "chrome":
		return launchChrome(useDefault)
	case "firefox":
		return launchFirefox(useDefault)
	case "":
		if _, err := chromeExecutable(); err == nil {
			return launchChrome(useDefault)
		}
		if _, err := firefoxExecutable(); err == nil {
			return launchFirefox(useDefault)
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
	value, _ := argString(args, "value", false)
	key, _ := argString(args, "key", false)
	jsArg, _ := argString(args, "js", false)

	// eval returns its raw result; the other actions return a {ok,msg}
	// envelope so a miss (no matching element) comes back as a tool error
	// the model can't skim past, with guidance on what to do instead.
	if action == "eval" {
		if jsArg == "" {
			return "", fmt.Errorf("eval needs js")
		}
		return withBrowser(func(s browserSession) (string, error) {
			out, err := s.Eval(jsArg)
			if err != nil {
				return "", err
			}
			return truncateForModel(out), nil
		})
	}

	// wait polls the page rather than running once, so give it its own path.
	if action == "wait" {
		if selector == "" && text == "" {
			return "", fmt.Errorf("wait needs text (words to wait for) or a selector")
		}
		return withBrowser(func(s browserSession) (string, error) {
			return waitForPage(s, selector, text)
		})
	}

	var js string
	switch action {
	case "click":
		if selector == "" && text == "" {
			return "", fmt.Errorf("click needs either text (the visible text to click) or a selector")
		}
		js = clickJS(selector, text)
	case "type":
		if selector == "" && text == "" {
			return "", fmt.Errorf("type needs either text (the field's label or placeholder) or a selector, plus value")
		}
		if value == "" {
			return "", fmt.Errorf("type needs value — the text to enter into the field")
		}
		js = typeJS(selector, text, value)
	case "press":
		if key == "" {
			key = "Enter"
		}
		js = pressJS(key)
	case "hover":
		if selector == "" && text == "" {
			return "", fmt.Errorf("hover needs either text or a selector")
		}
		js = hoverJS(selector, text)
	case "select":
		if selector == "" && text == "" {
			return "", fmt.Errorf("select needs the dropdown's text or a selector")
		}
		if value == "" {
			return "", fmt.Errorf("select needs value — the option to choose")
		}
		js = selectJS(selector, text, value)
	case "clear":
		if selector == "" && text == "" {
			return "", fmt.Errorf("clear needs either text or a selector")
		}
		js = clearJS(selector, text)
	case "get":
		if selector == "" && text == "" {
			return "", fmt.Errorf("get needs either text or a selector")
		}
		js = getJS(selector, text)
	case "scroll":
		js = scrollJS(selector, text)
	case "back":
		js = historyJS("history.back()", "went back")
	case "forward":
		js = historyJS("history.forward()", "went forward")
	case "reload":
		js = historyJS("location.reload()", "reloaded the page")
	default:
		return "", fmt.Errorf("unknown action %q", action)
	}
	return withBrowser(func(s browserSession) (string, error) {
		out, err := s.Eval(js)
		if err != nil {
			return "", err
		}
		return decodeActResult(out)
	})
}

// waitForPage polls until text/selector is present and visible, or ~10s
// elapse. Small models forget to wait for SPA content; this gives them a
// tool that blocks until the page catches up instead of a bare retry loop.
func waitForPage(s browserSession, selector, text string) (string, error) {
	deadline := time.Now().Add(10 * time.Second)
	for {
		out, err := s.Eval(presenceJS(selector, text))
		if err != nil {
			return "", err
		}
		if res, derr := decodeActResult(out); derr == nil {
			return res, nil
		}
		if time.Now().After(deadline) {
			target := fmt.Sprintf("text %q", text)
			if selector != "" {
				target = fmt.Sprintf("selector %q", selector)
			}
			return "", fmt.Errorf("waited 10s but %s never appeared. The page may not have loaded it, "+
				"or the wording differs — call browser_read to see what is there", target)
		}
		time.Sleep(400 * time.Millisecond)
	}
}

// decodeActResult turns the {ok,msg} envelope from a click/type/press script
// into either a success message or a tool error. A miss becomes an error so
// it renders red and the model treats it as a failure rather than repeating
// the same call.
func decodeActResult(out string) (string, error) {
	var r struct {
		OK  bool   `json:"ok"`
		Msg string `json:"msg"`
	}
	if err := json.Unmarshal([]byte(out), &r); err != nil {
		// Shouldn't happen — the scripts always stringify the envelope — but
		// if it does, hand the raw output back rather than swallowing it.
		return truncateForModel(out), nil
	}
	if !r.OK {
		return "", fmt.Errorf("%s", r.Msg)
	}
	return truncateForModel(r.Msg), nil
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

// actJSHelpers are shared page-side functions injected before each
// click/type script: ok/err build the {ok,msg} envelope the Go side decodes,
// and the finders locate an element by CSS selector or visible text so a
// model can say "click Sign in" instead of hand-writing a selector.
const actJSHelpers = `
	const __ok = (msg) => JSON.stringify({ok: true, msg: msg});
	const __err = (msg) => JSON.stringify({ok: false, msg: msg});
	const __norm = (s) => (s || "").trim().replace(/\s+/g, " ").toLowerCase();
	const __visible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
	// __findClickable(selector, text): selector wins; otherwise match visible
	// text against clickable elements (exact first, then substring), falling
	// back to any visible element whose own text matches.
	const __findClickable = (selector, text) => {
		if (selector) return document.querySelector(selector);
		const want = __norm(text);
		if (!want) return null;
		const label = (el) => __norm(el.innerText || el.value || el.getAttribute("aria-label") || el.title);
		const clickable = Array.from(document.querySelectorAll(
			"a,button,input[type=button],input[type=submit],input[type=reset],[role=button],[role=link],[role=menuitem],[role=tab],[role=option],summary,label,[onclick]"
		)).filter(__visible);
		return clickable.find(el => label(el) === want)
			|| clickable.find(el => label(el).includes(want))
			|| Array.from(document.querySelectorAll("body *")).filter(__visible)
				.find(el => __norm(el.innerText) === want)
			|| null;
	};
	// __findField(selector, text): selector wins; otherwise match a form field
	// by placeholder / aria-label / name / id, or an associated <label>.
	const __findField = (selector, text) => {
		if (selector) return document.querySelector(selector);
		const want = __norm(text);
		if (!want) return null;
		const fields = Array.from(document.querySelectorAll(
			"input:not([type=hidden]):not([type=submit]):not([type=button]),textarea,select,[contenteditable=''],[contenteditable=true]"
		)).filter(__visible);
		const attr = (el, a) => __norm(el.getAttribute(a));
		let m = fields.find(el => ["placeholder","aria-label","name","id"].some(a => attr(el, a) === want))
			|| fields.find(el => attr(el, "placeholder").includes(want) || attr(el, "aria-label").includes(want));
		if (!m) {
			const lbl = Array.from(document.querySelectorAll("label")).find(l => __norm(l.innerText).includes(want));
			if (lbl) {
				const forId = lbl.getAttribute("for");
				m = (forId && document.getElementById(forId)) || lbl.querySelector("input,textarea,select");
			}
		}
		return m || null;
	};
	// __findAny(selector, text): the loosest finder — any visible element
	// matching the selector or whose text/value/aria-label matches. Used by
	// hover / scroll / get / clear / wait where the target need not be a
	// canonically clickable or fillable element.
	const __findAny = (selector, text) => {
		if (selector) return document.querySelector(selector);
		const want = __norm(text);
		if (!want) return null;
		const label = (el) => __norm(el.innerText || el.value || el.getAttribute("aria-label") || el.title || el.getAttribute("placeholder"));
		const all = Array.from(document.querySelectorAll("body *")).filter(__visible);
		return all.find(el => label(el) === want)
			|| __findClickable(null, text)
			|| __findField(null, text)
			|| all.find(el => label(el).includes(want))
			|| null;
	};`

// clickJS clicks an element found by selector or visible text, returning the
// {ok,msg} envelope. A miss is ok:false with guidance to read the page first.
func clickJS(selector, text string) string {
	return fmt.Sprintf(`(() => {%s
		const el = __findClickable(%s, %s);
		if (!el) return __err(%s);
		el.scrollIntoView({block: "center"});
		el.click();
		return __ok("clicked <" + el.tagName.toLowerCase() + "> " + (__norm(el.innerText || el.value).slice(0, 60)));
	})()`, actJSHelpers, jsStr(selector), jsStr(text), jsStr(notFoundHint("clickable element", selector, text)))
}

// typeJS sets the value through the prototype setter so framework-controlled
// inputs (React et al.) see the change, then fires input/change.
func typeJS(selector, text, value string) string {
	return fmt.Sprintf(`(() => {%s
		const el = __findField(%s, %s);
		if (!el) return __err(%s);
		el.scrollIntoView({block: "center"});
		el.focus();
		if (el.isContentEditable) {
			el.textContent = %s;
		} else {
			const d = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), "value");
			if (d && d.set) d.set.call(el, %s); else el.value = %s;
		}
		el.dispatchEvent(new Event("input", {bubbles: true}));
		el.dispatchEvent(new Event("change", {bubbles: true}));
		return __ok("typed into <" + el.tagName.toLowerCase() + ">");
	})()`, actJSHelpers, jsStr(selector), jsStr(text),
		jsStr(notFoundHint("input field", selector, text)),
		jsStr(value), jsStr(value), jsStr(value))
}

// pressJS dispatches key events to the focused element. Synthetic events are
// untrusted, so browsers won't act on them natively — for the one case that
// matters most (Enter in a search box) we also submit the enclosing form.
func pressJS(key string) string {
	return fmt.Sprintf(`(() => {%s
		const key = %s;
		const el = document.activeElement || document.body;
		const opts = {key: key, bubbles: true, cancelable: true};
		const proceed = el.dispatchEvent(new KeyboardEvent("keydown", opts));
		el.dispatchEvent(new KeyboardEvent("keyup", opts));
		if (key === "Enter" && proceed && el.form) {
			if (el.form.requestSubmit) el.form.requestSubmit(); else el.form.submit();
			return __ok("pressed Enter and submitted the form");
		}
		return __ok("pressed " + key + " on <" + el.tagName.toLowerCase() + ">");
	})()`, actJSHelpers, jsStr(key))
}

// notFoundHint is the error text for a miss: it names what was searched for
// and tells the model to read the page rather than repeat the same call.
func notFoundHint(kind, selector, text string) string {
	target := fmt.Sprintf("text %q", text)
	if selector != "" {
		target = fmt.Sprintf("selector %q", selector)
	}
	return fmt.Sprintf("no %s found for %s. Call browser_read (what=\"links\" or what=\"html\") to see "+
		"what is actually on the page, then try different text or a selector — do not repeat this exact call.",
		kind, target)
}

// hoverJS dispatches the pointer/mouse-enter sequence real hover menus listen
// for; a plain :hover can't be forced, but these events open most menus.
func hoverJS(selector, text string) string {
	return fmt.Sprintf(`(() => {%s
		const el = __findAny(%s, %s);
		if (!el) return __err(%s);
		el.scrollIntoView({block: "center"});
		for (const t of ["pointerover","mouseover","mouseenter","pointermove","mousemove"]) {
			el.dispatchEvent(new MouseEvent(t, {bubbles: true, cancelable: true, view: window}));
		}
		return __ok("hovered <" + el.tagName.toLowerCase() + "> " + (__norm(el.innerText).slice(0, 60)));
	})()`, actJSHelpers, jsStr(selector), jsStr(text), jsStr(notFoundHint("element", selector, text)))
}

// selectJS chooses an option in a <select> by visible label or value, firing
// input/change so listeners react.
func selectJS(selector, text, value string) string {
	return fmt.Sprintf(`(() => {%s
		const el = __findField(%s, %s);
		if (!el || el.tagName.toLowerCase() !== "select") return __err(%s);
		const want = __norm(%s);
		const opt = Array.from(el.options).find(o => __norm(o.label || o.text) === want || __norm(o.value) === want)
			|| Array.from(el.options).find(o => __norm(o.label || o.text).includes(want));
		if (!opt) return __err("the dropdown has no option matching " + %s + ". Options: " +
			Array.from(el.options).map(o => (o.text || "").trim()).join(", "));
		el.value = opt.value;
		el.dispatchEvent(new Event("input", {bubbles: true}));
		el.dispatchEvent(new Event("change", {bubbles: true}));
		return __ok("selected \"" + (opt.text || opt.value) + "\"");
	})()`, actJSHelpers, jsStr(selector), jsStr(text),
		jsStr(notFoundHint("dropdown (<select>)", selector, text)), jsStr(value), jsStr(value))
}

// clearJS empties an input, textarea, or contenteditable, firing input/change.
func clearJS(selector, text string) string {
	return fmt.Sprintf(`(() => {%s
		const el = __findField(%s, %s);
		if (!el) return __err(%s);
		el.focus();
		if (el.isContentEditable) {
			el.textContent = "";
		} else {
			const d = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), "value");
			if (d && d.set) d.set.call(el, ""); else el.value = "";
		}
		el.dispatchEvent(new Event("input", {bubbles: true}));
		el.dispatchEvent(new Event("change", {bubbles: true}));
		return __ok("cleared <" + el.tagName.toLowerCase() + ">");
	})()`, actJSHelpers, jsStr(selector), jsStr(text), jsStr(notFoundHint("input field", selector, text)))
}

// getJS reads back one element's text, value, and href — a targeted
// alternative to browser_read when the model wants a single value.
func getJS(selector, text string) string {
	return fmt.Sprintf(`(() => {%s
		const el = __findAny(%s, %s);
		if (!el) return __err(%s);
		const parts = [];
		const t = (el.innerText || el.textContent || "").trim();
		if (t) parts.push("text: " + t.replace(/\s+/g, " ").slice(0, 300));
		if (el.value) parts.push("value: " + String(el.value).slice(0, 300));
		if (el.href) parts.push("href: " + el.href);
		if (!parts.length) parts.push("(<" + el.tagName.toLowerCase() + "> with no text or value)");
		return __ok(parts.join("\n"));
	})()`, actJSHelpers, jsStr(selector), jsStr(text), jsStr(notFoundHint("element", selector, text)))
}

// scrollJS scrolls to an element, or the page itself for the direction words
// top/bottom/up/down (matched before treating text as an element to find).
func scrollJS(selector, text string) string {
	return fmt.Sprintf(`(() => {%s
		const dir = __norm(%s);
		if (!%s && ["top","bottom","up","down"].includes(dir)) {
			if (dir === "top") window.scrollTo({top: 0});
			else if (dir === "bottom") window.scrollTo({top: document.body.scrollHeight});
			else window.scrollBy({top: (dir === "down" ? 1 : -1) * window.innerHeight * 0.9});
			return __ok("scrolled " + dir);
		}
		const el = __findAny(%s, %s);
		if (!el) return __err(%s);
		el.scrollIntoView({block: "center"});
		return __ok("scrolled to <" + el.tagName.toLowerCase() + "> " + (__norm(el.innerText).slice(0, 60)));
	})()`, actJSHelpers, jsStr(text), jsStr(selector),
		jsStr(selector), jsStr(text), jsStr(notFoundHint("element", selector, text)))
}

// presenceJS reports whether text/selector is currently present and visible,
// as the {ok,msg} envelope waitForPage polls on.
func presenceJS(selector, text string) string {
	return fmt.Sprintf(`(() => {%s
		const el = %s
			? document.querySelector(%s)
			: Array.from(document.querySelectorAll("body *")).filter(__visible)
				.find(e => __norm(e.innerText).includes(__norm(%s)));
		if (!el || !__visible(el)) return __err("not present yet");
		return __ok("found <" + el.tagName.toLowerCase() + ">");
	})()`, actJSHelpers, jsStr(selector), jsStr(selector), jsStr(text))
}

// historyJS runs a history/reload statement and reports it. Navigation
// finishes asynchronously, so the model should browser_read afterward.
func historyJS(stmt, msg string) string {
	return fmt.Sprintf(`(() => {%s
		%s;
		return __ok(%s);
	})()`, actJSHelpers, stmt, jsStr(msg+" — call browser_read to see the page"))
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
