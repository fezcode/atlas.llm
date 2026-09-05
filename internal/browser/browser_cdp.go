package browser

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"
)

// cdpSession drives a Chromium-family browser over the Chrome DevTools
// Protocol. Kept deliberately minimal: launch, one WebSocket to the first
// page target, and two commands (Page.navigate, Runtime.evaluate). Events
// arriving on the socket are skipped, so no domains need enabling.
type cdpSession struct {
	cmd     *exec.Cmd
	exited  chan struct{}
	profile string
	conn    *websocket.Conn
	// port is the DevTools HTTP endpoint, used for tab management; targetID
	// is the tab the WebSocket currently drives.
	port     int
	targetID string
	// persist is true when profile is a persistent dir that must survive
	// Close rather than being deleted with the session.
	persist bool

	mu     sync.Mutex
	nextID int64
}

func (s *cdpSession) Kind() string { return "chrome" }

// launchChrome starts a visible Chrome/Chromium/Edge window with an ephemeral
// DevTools port, and connects to its first tab.
//
// The profile depends on mode: fresh is an empty throwaway; default copies the
// user's real profile into a throwaway so their logins are available (Chrome
// since 136 refuses remote debugging on the actual default data dir, so a copy
// is the only way to drive a logged-in Chrome, and it keeps the real profile
// safe); persist is a stable dir atlas.llm owns and reuses, so cookies survive
// across runs. Either way remote debugging runs against our own directory,
// which the 136 restriction does not cover.
func launchChrome(mode profileMode) (*cdpSession, error) {
	exe, err := chromeExecutable()
	if err != nil {
		return nil, err
	}
	profile, persist, err := resolveBrowserProfile(mode, "atlas-chrome-", "chrome")
	if err != nil {
		return nil, err
	}
	// A persistent profile is seeded on its first launch only: that is what
	// makes "use my logins and remember them" reachable, and re-seeding later
	// would overwrite the sessions it exists to accumulate. Having no real
	// profile to copy is fatal for default (it was the whole request) but not
	// for persist, which just starts empty.
	if mode == profileDefault || (mode == profilePersist && persistNeedsSeed(profile)) {
		if err := seedChromeDefaultProfile(exe, profile); err != nil && mode == profileDefault {
			killAndCleanup(nil, nil, profile, !persist)
			return nil, err
		}
	}
	cmd := exec.Command(exe,
		// Port 0: Chrome picks a free port and writes it to
		// DevToolsActivePort inside the profile dir.
		"--remote-debugging-port=0",
		"--user-data-dir="+profile,
		"--no-first-run",
		"--no-default-browser-check",
		"--disable-search-engine-choice-screen",
		"--new-window",
		"about:blank",
	)
	if err := cmd.Start(); err != nil {
		killAndCleanup(nil, nil, profile, !persist)
		return nil, fmt.Errorf("start %s: %w", filepath.Base(exe), err)
	}
	exited := make(chan struct{})
	go func() { _ = cmd.Wait(); close(exited) }()

	fail := func(err error) (*cdpSession, error) {
		killAndCleanup(cmd, exited, profile, !persist)
		return nil, err
	}

	portFile, err := waitForBrowserFile(filepath.Join(profile, "DevToolsActivePort"), exited, 30*time.Second)
	if err != nil {
		return fail(err)
	}
	port, err := strconv.Atoi(strings.TrimSpace(strings.SplitN(string(portFile), "\n", 2)[0]))
	if err != nil {
		return fail(fmt.Errorf("parse DevToolsActivePort: %w", err))
	}

	target, err := cdpFirstPageTarget(port, 10*time.Second)
	if err != nil {
		return fail(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, target.wsURL, nil)
	if err != nil {
		return fail(fmt.Errorf("connect to DevTools: %w", err))
	}
	// Big pages and chatty targets can exceed the 32KB default.
	conn.SetReadLimit(32 << 20)

	s := &cdpSession{cmd: cmd, exited: exited, profile: profile, persist: persist, conn: conn, port: port, targetID: target.ID}
	s.installShim()
	return s, nil
}

// installShim injects browserShimBody into the current document and registers
// it for every future one. Best-effort: a page that blocks it just loses
// console capture, not the whole session. Preload registration is per-target
// in CDP, so this runs again after every tab switch.
func (s *cdpSession) installShim() {
	// New-document scripts are only injected while the Page domain is
	// enabled; its events arrive on the socket and are discarded by call().
	_, _ = s.call("Page.enable", map[string]any{}, 5*time.Second)
	_, _ = s.call("Page.addScriptToEvaluateOnNewDocument", map[string]any{"source": browserShimJS()}, 5*time.Second)
	_, _ = s.Eval(browserShimJS())
}

// cdpFirstPageTarget asks the DevTools HTTP endpoint for the tab list and
// returns the first page target. Retries while the endpoint warms up.
func cdpFirstPageTarget(port int, timeout time.Duration) (browserTab, error) {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		targets, err := cdpListTargets(port)
		if err == nil {
			for _, t := range targets {
				if t.wsURL != "" {
					return t, nil
				}
			}
			err = fmt.Errorf("no page target yet")
		}
		lastErr = err
		time.Sleep(200 * time.Millisecond)
	}
	return browserTab{}, fmt.Errorf("locate DevTools page target: %w", lastErr)
}

// cdpListTargets fetches and parses the DevTools /json/list tab list.
func cdpListTargets(port int) ([]browserTab, error) {
	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/json/list", port))
	if err != nil {
		return nil, fmt.Errorf("%w (%v)", errBrowserGone, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	return parseCDPTargets(data)
}

// parseCDPTargets decodes a DevTools target list, keeping only real pages —
// extensions, service workers, and devtools windows are not driveable tabs.
func parseCDPTargets(data []byte) ([]browserTab, error) {
	var raw []struct {
		ID    string `json:"id"`
		Type  string `json:"type"`
		Title string `json:"title"`
		URL   string `json:"url"`
		WS    string `json:"webSocketDebuggerUrl"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse DevTools target list: %w", err)
	}
	var out []browserTab
	for _, t := range raw {
		if t.Type != "page" {
			continue
		}
		out = append(out, browserTab{ID: t.ID, URL: t.URL, Title: t.Title, wsURL: t.WS})
	}
	return out, nil
}

// call sends one CDP command and waits for the response with a matching id,
// discarding interleaved protocol events.
func (s *cdpSession) call(method string, params map[string]any, timeout time.Duration) (json.RawMessage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.nextID++
	id := s.nextID

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	req, err := json.Marshal(map[string]any{"id": id, "method": method, "params": params})
	if err != nil {
		return nil, err
	}
	if err := s.conn.Write(ctx, websocket.MessageText, req); err != nil {
		return nil, fmt.Errorf("%w (%v)", errBrowserGone, err)
	}
	for {
		_, data, err := s.conn.Read(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return nil, fmt.Errorf("%s timed out after %s", method, timeout)
			}
			return nil, fmt.Errorf("%w (%v)", errBrowserGone, err)
		}
		var msg struct {
			ID     int64           `json:"id"`
			Result json.RawMessage `json:"result"`
			Error  *struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal(data, &msg); err != nil || msg.ID != id {
			continue // an event, or a reply to something we gave up on
		}
		if msg.Error != nil {
			return nil, fmt.Errorf("%s: %s", method, msg.Error.Message)
		}
		return msg.Result, nil
	}
}

func (s *cdpSession) Navigate(url string) error {
	res, err := s.call("Page.navigate", map[string]any{"url": url}, 30*time.Second)
	if err != nil {
		return err
	}
	var nav struct {
		ErrorText string `json:"errorText"`
	}
	_ = json.Unmarshal(res, &nav)
	if nav.ErrorText != "" {
		return fmt.Errorf("navigation to %s failed: %s", url, nav.ErrorText)
	}
	return s.waitForLoad(20 * time.Second)
}

// waitForLoad polls document.readyState instead of subscribing to Page
// events — one less protocol surface, and it tolerates the execution
// context being torn down mid-navigation.
func (s *cdpSession) waitForLoad(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		state, err := s.Eval("document.readyState")
		if err == nil && (state == "complete" || state == "interactive") {
			return nil
		}
		if err != nil && strings.Contains(err.Error(), errBrowserGone.Error()) {
			return err
		}
		time.Sleep(300 * time.Millisecond)
	}
	// A page that never settles is still usable; let the caller read what's
	// there rather than failing the whole step.
	return nil
}

func (s *cdpSession) Eval(js string) (string, error) {
	res, err := s.call("Runtime.evaluate", map[string]any{
		"expression":    js,
		"returnByValue": true,
		"awaitPromise":  true,
	}, 20*time.Second)
	if err != nil {
		return "", err
	}
	var out struct {
		Result struct {
			Type        string          `json:"type"`
			Value       json.RawMessage `json:"value"`
			Description string          `json:"description"`
		} `json:"result"`
		ExceptionDetails *struct {
			Text      string `json:"text"`
			Exception *struct {
				Description string `json:"description"`
			} `json:"exception"`
		} `json:"exceptionDetails"`
	}
	if err := json.Unmarshal(res, &out); err != nil {
		return "", err
	}
	if ex := out.ExceptionDetails; ex != nil {
		msg := ex.Text
		if ex.Exception != nil && ex.Exception.Description != "" {
			msg = ex.Exception.Description
		}
		return "", fmt.Errorf("JavaScript error: %s", msg)
	}
	return rawJSValueToString(out.Result.Type, out.Result.Value, out.Result.Description), nil
}

// rawJSValueToString renders an evaluate result for the model: strings
// unquoted, other JSON values as-is, valueless results by description/type.
func rawJSValueToString(typ string, value json.RawMessage, description string) string {
	if len(value) > 0 {
		var str string
		if err := json.Unmarshal(value, &str); err == nil {
			return str
		}
		return string(value)
	}
	if description != "" {
		return description
	}
	return "(" + typ + ")"
}

func (s *cdpSession) Screenshot(clip *screenshotClip) ([]byte, error) {
	params := map[string]any{"format": "png"}
	if clip != nil {
		params["clip"] = map[string]any{
			"x": clip.X, "y": clip.Y, "width": clip.Width, "height": clip.Height, "scale": 1,
		}
		// The clip is in document coordinates; without this, regions outside
		// the current viewport come back blank.
		params["captureBeyondViewport"] = true
	}
	res, err := s.call("Page.captureScreenshot", params, 20*time.Second)
	if err != nil {
		return nil, err
	}
	var out struct {
		Data string `json:"data"`
	}
	if err := json.Unmarshal(res, &out); err != nil {
		return nil, err
	}
	return base64.StdEncoding.DecodeString(out.Data)
}

func (s *cdpSession) Tabs() ([]browserTab, error) {
	tabs, err := cdpListTargets(s.port)
	if err != nil {
		return nil, err
	}
	for i := range tabs {
		tabs[i].Active = tabs[i].ID == s.targetID
	}
	return tabs, nil
}

// attach points the session's WebSocket at another tab. Safe without extra
// locking: every entry point already serializes on browserMu.
func (s *cdpSession) attach(t browserTab) error {
	if t.wsURL == "" {
		return fmt.Errorf("tab %q has no DevTools socket — another debugger may be attached to it", t.ID)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, t.wsURL, nil)
	if err != nil {
		return fmt.Errorf("connect to tab: %w", err)
	}
	conn.SetReadLimit(32 << 20)
	_ = s.conn.Close(websocket.StatusNormalClosure, "")
	s.conn = conn
	s.targetID = t.ID
	// Bring the tab forward so the user sees what is being driven, and
	// re-arm the shim — both preload registration and the live document are
	// per-target in CDP.
	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/json/activate/%s", s.port, t.ID))
	if err == nil {
		resp.Body.Close()
	}
	s.installShim()
	return nil
}

func (s *cdpSession) SwitchTab(id string) error {
	tabs, err := cdpListTargets(s.port)
	if err != nil {
		return err
	}
	for _, t := range tabs {
		if t.ID == id {
			return s.attach(t)
		}
	}
	return fmt.Errorf("tab %q no longer exists", id)
}

func (s *cdpSession) NewTab(url string) error {
	// The tab is created blank and navigated over the protocol afterwards:
	// /json/new takes the target URL raw on the request line, which breaks
	// on anything with spaces or quotes. Creation requires PUT since
	// Chrome 111.
	req, err := http.NewRequest(http.MethodPut,
		fmt.Sprintf("http://127.0.0.1:%d/json/new", s.port), nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("%w (%v)", errBrowserGone, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	var t struct {
		ID string `json:"id"`
		WS string `json:"webSocketDebuggerUrl"`
	}
	if err := json.Unmarshal(data, &t); err != nil || t.ID == "" {
		return fmt.Errorf("open new tab: unexpected response %q", strings.TrimSpace(string(data)))
	}
	if err := s.attach(browserTab{ID: t.ID, wsURL: t.WS}); err != nil {
		return err
	}
	if url != "" && url != "about:blank" {
		return s.Navigate(url)
	}
	return nil
}

func (s *cdpSession) CloseTab(id string) error {
	if id == s.targetID {
		return fmt.Errorf("cannot close the tab being driven — switch to another tab first")
	}
	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/json/close/%s", s.port, id))
	if err != nil {
		return fmt.Errorf("%w (%v)", errBrowserGone, err)
	}
	resp.Body.Close()
	return nil
}

func (s *cdpSession) SetFiles(paths []string) error {
	res, err := s.call("DOM.getDocument", map[string]any{"depth": 0}, 10*time.Second)
	if err != nil {
		return err
	}
	var doc struct {
		Root struct {
			NodeID int64 `json:"nodeId"`
		} `json:"root"`
	}
	if err := json.Unmarshal(res, &doc); err != nil {
		return err
	}
	res, err = s.call("DOM.querySelector", map[string]any{
		"nodeId": doc.Root.NodeID, "selector": "[data-atlas-upload]",
	}, 10*time.Second)
	if err != nil {
		return err
	}
	var q struct {
		NodeID int64 `json:"nodeId"`
	}
	if err := json.Unmarshal(res, &q); err != nil {
		return err
	}
	if q.NodeID == 0 {
		return fmt.Errorf("could not locate the marked file input")
	}
	_, err = s.call("DOM.setFileInputFiles", map[string]any{"files": paths, "nodeId": q.NodeID}, 10*time.Second)
	return err
}

func (s *cdpSession) Close() {
	// Graceful first: Browser.close lets Chrome flush and exit cleanly.
	_, _ = s.call("Browser.close", map[string]any{}, 3*time.Second)
	_ = s.conn.Close(websocket.StatusNormalClosure, "")
	killAndCleanup(s.cmd, s.exited, s.profile, !s.persist)
}
