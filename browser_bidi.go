package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"

	"github.com/coder/websocket"
)

// bidiSession drives Firefox over WebDriver BiDi (Firefox dropped CDP in
// 129). Same minimal shape as cdpSession: navigate + evaluate, everything
// else layered on top in browser.go.
type bidiSession struct {
	cmd     *exec.Cmd
	exited  chan struct{}
	profile string
	conn    *websocket.Conn
	context string // top-level browsing context (the tab we drive)

	mu     sync.Mutex
	nextID int64
}

func (s *bidiSession) Kind() string { return "firefox" }

// firefoxPrefs is written as user.js into the throwaway profile so a fresh
// Firefox comes up as a plain blank window instead of onboarding tabs and
// default-browser nags.
const firefoxPrefs = `user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("browser.aboutwelcome.enabled", false);
user_pref("browser.startup.homepage", "about:blank");
user_pref("browser.startup.firstrunSkipsHomepage", true);
user_pref("datareporting.policy.dataSubmissionEnabled", false);
user_pref("app.update.auto", false);
`

// launchFirefox starts a visible Firefox window on a throwaway profile with
// the remote agent on an ephemeral port, and opens a BiDi session to it.
func launchFirefox() (*bidiSession, error) {
	exe, err := firefoxExecutable()
	if err != nil {
		return nil, err
	}
	profile, err := browserTempProfile("atlas-firefox-")
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(filepath.Join(profile, "user.js"), []byte(firefoxPrefs), 0644); err != nil {
		killAndCleanup(nil, nil, profile)
		return nil, err
	}
	cmd := exec.Command(exe,
		// Port 0: the remote agent picks a free port and writes it to
		// WebDriverBiDiServer.json inside the profile dir.
		"--remote-debugging-port", "0",
		"--profile", profile,
		"--no-remote",
		"--new-instance",
		"about:blank",
	)
	if err := cmd.Start(); err != nil {
		killAndCleanup(nil, nil, profile)
		return nil, fmt.Errorf("start firefox: %w", err)
	}
	exited := make(chan struct{})
	go func() { _ = cmd.Wait(); close(exited) }()

	fail := func(err error) (*bidiSession, error) {
		killAndCleanup(cmd, exited, profile)
		return nil, err
	}

	serverFile, err := waitForBrowserFile(filepath.Join(profile, "WebDriverBiDiServer.json"), exited, 45*time.Second)
	if err != nil {
		return fail(err)
	}
	host, port, err := parseBiDiServerFile(serverFile)
	if err != nil {
		return fail(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, fmt.Sprintf("ws://%s:%d/session", host, port), nil)
	if err != nil {
		return fail(fmt.Errorf("connect to Firefox remote agent: %w", err))
	}
	conn.SetReadLimit(32 << 20)

	s := &bidiSession{cmd: cmd, exited: exited, profile: profile, conn: conn}
	if _, err := s.call("session.new", map[string]any{"capabilities": map[string]any{}}, 15*time.Second); err != nil {
		_ = conn.Close(websocket.StatusNormalClosure, "")
		return fail(fmt.Errorf("open BiDi session: %w", err))
	}
	tree, err := s.call("browsingContext.getTree", map[string]any{}, 10*time.Second)
	if err != nil {
		_ = conn.Close(websocket.StatusNormalClosure, "")
		return fail(fmt.Errorf("list browsing contexts: %w", err))
	}
	var contexts struct {
		Contexts []struct {
			Context string `json:"context"`
		} `json:"contexts"`
	}
	if err := json.Unmarshal(tree, &contexts); err != nil || len(contexts.Contexts) == 0 {
		_ = conn.Close(websocket.StatusNormalClosure, "")
		return fail(fmt.Errorf("no browsing context found"))
	}
	s.context = contexts.Contexts[0].Context
	return s, nil
}

// parseBiDiServerFile decodes the {"ws_host","ws_port"} file Firefox's
// remote agent writes into the profile.
func parseBiDiServerFile(data []byte) (string, int, error) {
	var f struct {
		Host string `json:"ws_host"`
		Port int    `json:"ws_port"`
	}
	if err := json.Unmarshal(data, &f); err != nil {
		return "", 0, fmt.Errorf("parse WebDriverBiDiServer.json: %w", err)
	}
	if f.Port == 0 {
		return "", 0, fmt.Errorf("WebDriverBiDiServer.json has no ws_port")
	}
	if f.Host == "" {
		f.Host = "127.0.0.1"
	}
	return f.Host, f.Port, nil
}

// call sends one BiDi command and waits for the success/error envelope with
// a matching id, discarding interleaved events.
func (s *bidiSession) call(method string, params map[string]any, timeout time.Duration) (json.RawMessage, error) {
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
			Type    string          `json:"type"`
			ID      int64           `json:"id"`
			Result  json.RawMessage `json:"result"`
			Error   string          `json:"error"`
			Message string          `json:"message"`
		}
		if err := json.Unmarshal(data, &msg); err != nil || msg.ID != id || msg.Type == "event" {
			continue
		}
		if msg.Type == "error" {
			what := msg.Error
			if msg.Message != "" {
				what += ": " + msg.Message
			}
			return nil, fmt.Errorf("%s: %s", method, what)
		}
		return msg.Result, nil
	}
}

func (s *bidiSession) Navigate(url string) error {
	// wait:"complete" makes the remote agent hold the reply until the load
	// event, so no readyState polling is needed on this path.
	_, err := s.call("browsingContext.navigate", map[string]any{
		"context": s.context,
		"url":     url,
		"wait":    "complete",
	}, 45*time.Second)
	if err != nil {
		return fmt.Errorf("navigation to %s failed: %w", url, err)
	}
	return nil
}

func (s *bidiSession) Eval(js string) (string, error) {
	res, err := s.call("script.evaluate", map[string]any{
		"expression":      js,
		"target":          map[string]any{"context": s.context},
		"awaitPromise":    true,
		"resultOwnership": "none",
	}, 20*time.Second)
	if err != nil {
		return "", err
	}
	var out struct {
		Type   string `json:"type"` // "success" or "exception"
		Result struct {
			Type  string          `json:"type"`
			Value json.RawMessage `json:"value"`
		} `json:"result"`
		ExceptionDetails struct {
			Text string `json:"text"`
		} `json:"exceptionDetails"`
	}
	if err := json.Unmarshal(res, &out); err != nil {
		return "", err
	}
	if out.Type == "exception" {
		return "", fmt.Errorf("JavaScript error: %s", out.ExceptionDetails.Text)
	}
	return rawJSValueToString(out.Result.Type, out.Result.Value, ""), nil
}

func (s *bidiSession) Close() {
	// Graceful first so the profile isn't mid-write when it's removed.
	_, _ = s.call("browser.close", map[string]any{}, 3*time.Second)
	_ = s.conn.Close(websocket.StatusNormalClosure, "")
	killAndCleanup(s.cmd, s.exited, s.profile)
}
