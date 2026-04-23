package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"sync"
	"time"
)

// llamaServer wraps a long-running llama-server subprocess and the HTTP
// client that talks to it. One instance per (model, context-size). The
// model stays loaded in memory so subsequent /completion calls don't pay
// the GGUF mmap + warmup cost on every turn.
type llamaServer struct {
	cmd      *exec.Cmd
	port     int
	model    Model
	ctxN     int
	client   *http.Client
	waitOnce sync.Once
	waitErr  chan error
}

var (
	serverMu     sync.Mutex
	activeServer *llamaServer
)

// ensureServer returns a running llamaServer for the current model, spawning
// one if needed and restarting if the caller switched models. Safe to call
// concurrently.
func ensureServer() (*llamaServer, error) {
	serverMu.Lock()
	defer serverMu.Unlock()

	m, err := currentModel()
	if err != nil {
		return nil, err
	}
	if activeServer != nil && activeServer.model.Name == m.Name {
		return activeServer, nil
	}
	if activeServer != nil {
		activeServer.stopLocked()
		activeServer = nil
	}
	s, err := startLlamaServer(m)
	if err != nil {
		return nil, err
	}
	activeServer = s
	return s, nil
}

// shutdownServer is called from startChat's defer so the subprocess doesn't
// outlive the TUI session.
func shutdownServer() {
	serverMu.Lock()
	defer serverMu.Unlock()
	if activeServer != nil {
		activeServer.stopLocked()
		activeServer = nil
	}
}

func startLlamaServer(m Model) (*llamaServer, error) {
	bin, err := findEngineServer()
	if err != nil {
		return nil, fmt.Errorf("llama-server: %w", err)
	}
	modelPath, err := requireModel(m)
	if err != nil {
		return nil, err
	}
	port, err := pickFreePort()
	if err != nil {
		return nil, fmt.Errorf("pick port: %w", err)
	}

	threads := runtime.NumCPU() - 1
	if threads < 1 {
		threads = 1
	}
	if threads > 6 {
		threads = 6
	}

	args := []string{
		"-m", modelPath,
		"--host", "127.0.0.1",
		"--port", fmt.Sprintf("%d", port),
		"-c", "4096",
		"-t", fmt.Sprintf("%d", threads),
		"-ngl", "0",
		"--log-disable",
	}
	cmd := exec.Command(bin, args...)
	cmd.Stdin = bytes.NewReader(nil)
	cmd.Stdout = newLogWriter("llama-server stdout")
	cmd.Stderr = newLogWriter("llama-server stderr")
	cmd.Env = append(os.Environ(), "OMP_STACKSIZE=64M")
	applyEngineSysProcAttr(cmd)

	log.Printf("starting llama-server on :%d (model=%s threads=%d)", port, m.Name, threads)
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start llama-server: %w", err)
	}

	s := &llamaServer{
		cmd:     cmd,
		port:    port,
		model:   m,
		ctxN:    4096,
		client:  &http.Client{Timeout: 10 * time.Minute},
		waitErr: make(chan error, 1),
	}
	// Single background Wait(); result is broadcast via waitErr so both
	// waitReady and stopLocked can observe exit without double-calling Wait.
	go func() { s.waitErr <- cmd.Wait() }()

	if err := s.waitReady(90 * time.Second); err != nil {
		s.stopLocked()
		return nil, err
	}
	log.Printf("llama-server ready on :%d", port)
	return s, nil
}

func pickFreePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

// waitReady polls GET /health until the server reports ready, or gives up.
// If the subprocess exits early, returns that error instead of timing out.
func (s *llamaServer) waitReady(maxWait time.Duration) error {
	deadline := time.Now().Add(maxWait)
	url := fmt.Sprintf("http://127.0.0.1:%d/health", s.port)

	for time.Now().Before(deadline) {
		select {
		case err := <-s.waitErr:
			// Process exited before becoming ready — put the error back for
			// stopLocked(), then report to caller.
			s.waitErr <- err
			return fmt.Errorf("llama-server exited before ready: %v", err)
		default:
		}
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
		resp, err := s.client.Do(req)
		cancel()
		if err == nil {
			body := make([]byte, 256)
			n, _ := resp.Body.Read(body)
			resp.Body.Close()
			if resp.StatusCode == 200 && bytes.Contains(body[:n], []byte(`"ok"`)) {
				return nil
			}
		}
		time.Sleep(250 * time.Millisecond)
	}
	return fmt.Errorf("llama-server did not become ready in %s", maxWait)
}

func (s *llamaServer) stopLocked() {
	if s.cmd == nil || s.cmd.Process == nil {
		return
	}
	log.Printf("stopping llama-server pid=%d", s.cmd.Process.Pid)
	_ = s.cmd.Process.Kill()
	// Drain the background Wait goroutine so resources are released.
	select {
	case <-s.waitErr:
	case <-time.After(5 * time.Second):
	}
}

// ChatMsg is a single turn passed to /v1/chat/completions. The "role" is
// one of "system", "user", or "assistant".
type ChatMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Messages    []ChatMsg `json:"messages"`
	MaxTokens   int       `json:"max_tokens"`
	Temperature float64   `json:"temperature"`
	Stream      bool      `json:"stream"`
	CachePrompt bool      `json:"cache_prompt"`
}

type chatChoice struct {
	Message      ChatMsg `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

type chatResponse struct {
	Choices []chatChoice `json:"choices"`
	Error   struct {
		Message string `json:"message"`
	} `json:"error"`
}

// ChatComplete sends a chat-style request to the running server. llama-server
// applies the model's own chat template from the GGUF metadata (Gemma-3's
// <start_of_turn>/<end_of_turn> sentinels, ChatML for other families, etc.)
// and returns only the assistant's reply — so the model stops at the turn
// boundary instead of spewing fake "User:/Assistant:" continuations.
func (s *llamaServer) ChatComplete(msgs []ChatMsg, maxTokens int) (string, error) {
	body, _ := json.Marshal(chatRequest{
		Messages:    msgs,
		MaxTokens:   maxTokens,
		Temperature: 0.2,
		Stream:      false,
		CachePrompt: true,
	})
	url := fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", s.port)
	start := time.Now()
	resp, err := s.client.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("POST /v1/chat/completions: %w", err)
	}
	defer resp.Body.Close()
	var cr chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		return "", fmt.Errorf("decode chat completion: %w", err)
	}
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("llama-server HTTP %d: %s", resp.StatusCode, cr.Error.Message)
	}
	if len(cr.Choices) == 0 {
		return "", fmt.Errorf("empty chat completion response")
	}
	content := cr.Choices[0].Message.Content
	log.Printf("chat completion ok in %s (msgs=%d, max=%d, reply=%d bytes, finish=%s)",
		time.Since(start), len(msgs), maxTokens, len(content), cr.Choices[0].FinishReason)
	return content, nil
}

// logWriter forwards subprocess stdout/stderr into the atlas.llm log with a
// tag prefix so server messages show up alongside our own.
type logWriter struct{ tag string }

func newLogWriter(tag string) *logWriter { return &logWriter{tag: tag} }
func (w *logWriter) Write(p []byte) (int, error) {
	log.Printf("[%s] %s", w.tag, bytes.TrimRight(p, "\r\n "))
	return len(p), nil
}
