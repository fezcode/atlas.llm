package main

import (
	"context"
	"encoding/json"
	"fmt"
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

	mu     sync.Mutex
	nextID int64
}

func (s *cdpSession) Kind() string { return "chrome" }

// launchChrome starts a visible Chrome/Chromium/Edge window on a throwaway
// profile with an ephemeral DevTools port, and connects to its first tab.
func launchChrome() (*cdpSession, error) {
	exe, err := chromeExecutable()
	if err != nil {
		return nil, err
	}
	profile, err := browserTempProfile("atlas-chrome-")
	if err != nil {
		return nil, err
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
		killAndCleanup(nil, nil, profile)
		return nil, fmt.Errorf("start %s: %w", filepath.Base(exe), err)
	}
	exited := make(chan struct{})
	go func() { _ = cmd.Wait(); close(exited) }()

	fail := func(err error) (*cdpSession, error) {
		killAndCleanup(cmd, exited, profile)
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

	wsURL, err := cdpFirstPageTarget(port, 10*time.Second)
	if err != nil {
		return fail(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, wsURL, nil)
	if err != nil {
		return fail(fmt.Errorf("connect to DevTools: %w", err))
	}
	// Big pages and chatty targets can exceed the 32KB default.
	conn.SetReadLimit(32 << 20)

	return &cdpSession{cmd: cmd, exited: exited, profile: profile, conn: conn}, nil
}

// cdpFirstPageTarget asks the DevTools HTTP endpoint for the tab list and
// returns the WebSocket URL of the first page. Retries while the endpoint
// warms up.
func cdpFirstPageTarget(port int, timeout time.Duration) (string, error) {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/json/list", port))
		if err == nil {
			var targets []struct {
				Type  string `json:"type"`
				WSURL string `json:"webSocketDebuggerUrl"`
			}
			err = json.NewDecoder(resp.Body).Decode(&targets)
			resp.Body.Close()
			if err == nil {
				for _, t := range targets {
					if t.Type == "page" && t.WSURL != "" {
						return t.WSURL, nil
					}
				}
				err = fmt.Errorf("no page target yet")
			}
		}
		lastErr = err
		time.Sleep(200 * time.Millisecond)
	}
	return "", fmt.Errorf("locate DevTools page target: %w", lastErr)
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

func (s *cdpSession) Close() {
	// Graceful first: Browser.close lets Chrome flush and exit cleanly.
	_, _ = s.call("Browser.close", map[string]any{}, 3*time.Second)
	_ = s.conn.Close(websocket.StatusNormalClosure, "")
	killAndCleanup(s.cmd, s.exited, s.profile)
}
