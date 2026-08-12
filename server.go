package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strings"
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
	ctxN     int // per-slot context — what one conversation can use
	slots    int // llama-server slots (-np); total -c is ctxN*slots
	gpuLayer int
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
	// A -ngl change only takes effect at process start, so treat it as
	// part of the server's identity alongside the model.
	cfg, _ := loadConfig()
	wantNGL := resolveGPULayers(cfg)
	wantCtx := resolveCtxSize(cfg)
	if activeServer != nil && activeServer.model.Name == m.Name &&
		activeServer.gpuLayer == wantNGL && activeServer.ctxN == wantCtx {
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

// kvCacheSlots is how many server slots the optimized launch runs. Two
// slots keep one-shot utility calls (/summarize, /grep, compact's
// summarizer) from evicting the conversation's cached prefix: llama-server
// routes each request to the idle slot with the longest matching prefix,
// so one-shots land in the second slot and the conversation's KV survives.
const kvCacheSlots = 2

func startLlamaServer(m Model) (*llamaServer, error) {
	bin, err := findEngineServer()
	if err != nil {
		return nil, fmt.Errorf("llama-server: %w", err)
	}
	modelPath, err := requireModel(m)
	if err != nil {
		return nil, err
	}

	threads := runtime.NumCPU() - 1
	if threads < 1 {
		threads = 1
	}
	if threads > 6 {
		threads = 6
	}

	cfg, _ := loadConfig()
	ngl := resolveGPULayers(cfg)
	ctxN := resolveCtxSize(cfg)

	s, err := launchLlamaServer(bin, modelPath, m, threads, ngl, ctxN, true)
	if errors.Is(err, errServerExitedEarly) {
		// The optimized flag set is rejected by engines that predate the
		// `-fa on` syntax, and by backend/model combos that can't force
		// flash attention. The base flags work everywhere.
		log.Printf("llama-server rejected optimized flags (%v) — retrying with base flags", err)
		s, err = launchLlamaServer(bin, modelPath, m, threads, ngl, ctxN, false)
	}
	return s, err
}

func launchLlamaServer(bin, modelPath string, m Model, threads, ngl, ctxN int, optimized bool) (*llamaServer, error) {
	port, err := pickFreePort()
	if err != nil {
		return nil, fmt.Errorf("pick port: %w", err)
	}

	slots := 1
	serverCtx := ctxN // -c is the total across slots; each slot gets an equal share
	args := []string{
		"-m", modelPath,
		"--host", "127.0.0.1",
		"--port", fmt.Sprintf("%d", port),
		"-t", fmt.Sprintf("%d", threads),
		"-ngl", fmt.Sprintf("%d", ngl),
		"--log-disable",
	}
	if optimized {
		slots = kvCacheSlots
		serverCtx = ctxN * kvCacheSlots
		args = append(args,
			// q8_0 halves KV memory vs f16, which is what pays for the
			// second slot: two q8_0 slots cost what one f16 slot did, so
			// per-slot context is unchanged on the 16GB machines this
			// targets. Quantizing V requires flash attention; forcing it
			// on is also a prefill speedup where supported, and the base
			// fallback covers where it isn't.
			"-fa", "on",
			"--cache-type-k", "q8_0",
			"--cache-type-v", "q8_0",
			"--parallel", fmt.Sprintf("%d", slots),
			// Salvage still-matching KV chunks (min 256 tokens) past the
			// first divergence instead of reprocessing everything after
			// it — mainly softens the full re-prefill after /compact
			// rewrites history. Runtime no-op for SWA models (Gemma).
			"--cache-reuse", "256",
		)
	}
	args = append(args, "-c", fmt.Sprintf("%d", serverCtx))
	cmd := exec.Command(bin, args...)
	cmd.Stdin = bytes.NewReader(nil)
	cmd.Stdout = newLogWriter("llama-server stdout")
	cmd.Stderr = newLogWriter("llama-server stderr")
	cmd.Env = append(os.Environ(), "OMP_STACKSIZE=64M")
	applyEngineSysProcAttr(cmd)

	log.Printf("starting llama-server on :%d (model=%s threads=%d ngl=%d ctx=%d slots=%d optimized=%v)",
		port, m.Name, threads, ngl, ctxN, slots, optimized)
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start llama-server: %w", err)
	}

	s := &llamaServer{
		cmd:      cmd,
		port:     port,
		model:    m,
		ctxN:     ctxN,
		slots:    slots,
		gpuLayer: ngl,
		client:   &http.Client{Timeout: 10 * time.Minute},
		waitErr:  make(chan error, 1),
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

// errServerExitedEarly marks a llama-server process that died before its
// /health endpoint came up — the signature of a rejected flag, as opposed
// to a slow model load (timeout) or a missing binary. Only this case is
// worth retrying with different flags.
var errServerExitedEarly = errors.New("llama-server exited before ready")

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
			return fmt.Errorf("%w: %v", errServerExitedEarly, err)
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

// ChatMsg is a single turn passed to /v1/chat/completions. Role is one of
// "system", "user", "assistant", or "tool". The tool-call fields only apply
// to assistant messages (ToolCalls) and tool messages (Name, ToolCallID).
type ChatMsg struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

// ToolCall is one requested function invocation emitted by the assistant.
// Arguments is a JSON-encoded string per the OpenAI schema, not a decoded
// object — the caller is responsible for unmarshalling.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type chatRequest struct {
	Messages    []ChatMsg        `json:"messages"`
	MaxTokens   int              `json:"max_tokens"`
	Temperature float64          `json:"temperature"`
	Stream      bool             `json:"stream"`
	CachePrompt bool             `json:"cache_prompt"`
	Tools       []map[string]any `json:"tools,omitempty"`
	// ChatTemplateKwargs is forwarded to the model's Jinja chat template.
	// {"enable_thinking": false} is how reasoning models (Qwen3.5) are told
	// to skip the <think> block. Note reasoning_budget does NOT work for
	// this — it truncates the thinking rather than preventing it.
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs,omitempty"`
}

type chatChoice struct {
	Message      assistantMessage `json:"message"`
	FinishReason string           `json:"finish_reason"`
}

// assistantMessage is the response-side shape of an assistant reply. Content
// can be null when the model chose to emit tool_calls instead.
type assistantMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	// ReasoningContent holds a reasoning model's thinking, which
	// llama-server separates from the answer. We don't display it, but we
	// need it to tell "the model said nothing" apart from "the model spent
	// its whole budget thinking".
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
}

type chatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type chatResponse struct {
	Choices []chatChoice `json:"choices"`
	Usage   chatUsage    `json:"usage"`
	Error   struct {
		Message string `json:"message"`
	} `json:"error"`
}

// UsageStats mirrors the last /v1/chat/completions usage block so the TUI
// can render a context-usage indicator. Total == prompt + completion.
type UsageStats struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	ContextSize      int
}

var (
	lastUsageMu sync.RWMutex
	lastUsage   UsageStats
)

func setLastUsage(u UsageStats) {
	lastUsageMu.Lock()
	lastUsage = u
	lastUsageMu.Unlock()
}

// GetLastUsage returns the most recent token usage reported by
// llama-server, including the active context-window size. ContextSize
// defaults to 0 until the first request completes.
func GetLastUsage() UsageStats {
	lastUsageMu.RLock()
	defer lastUsageMu.RUnlock()
	return lastUsage
}

// ResetUsage zeroes the usage counter (e.g. after /reset). Preserves the
// context-size hint so the TUI can still render the denominator.
func ResetUsage() {
	lastUsageMu.Lock()
	ctx := lastUsage.ContextSize
	lastUsage = UsageStats{ContextSize: ctx}
	lastUsageMu.Unlock()
}

// ChatComplete sends a chat-style request to the running server. llama-server
// applies the model's own chat template from the GGUF metadata (Gemma-3's
// <start_of_turn>/<end_of_turn> sentinels, ChatML for other families, etc.)
// and returns only the assistant's reply — so the model stops at the turn
// boundary instead of spewing fake "User:/Assistant:" continuations.
func (s *llamaServer) ChatComplete(ctx context.Context, msgs []ChatMsg, maxTokens int) (string, error) {
	content, _, err := s.chatCompleteCore(ctx, msgs, maxTokens, nil, false)
	return content, err
}

// ChatCompleteNoThinking is for one-shot utility calls — summarize, grep,
// compact — where a reasoning model's <think> block is pure waste: it
// consumes the entire token budget and can leave the answer empty.
func (s *llamaServer) ChatCompleteNoThinking(ctx context.Context, msgs []ChatMsg, maxTokens int) (string, error) {
	content, _, err := s.chatCompleteCore(ctx, msgs, maxTokens, nil, true)
	return content, err
}

// ChatCompleteWithTools is the tool-enabled variant: advertises `tools` on
// the request and returns the (possibly empty) list of tool_calls the model
// emitted in addition to any assistant content. The caller is responsible
// for executing the calls and re-invoking with the tool results appended.
func (s *llamaServer) ChatCompleteWithTools(ctx context.Context, msgs []ChatMsg, tools []map[string]any, maxTokens int) (string, []ToolCall, error) {
	return s.chatCompleteCore(ctx, msgs, maxTokens, tools, false)
}

func (s *llamaServer) chatCompleteCore(ctx context.Context, msgs []ChatMsg, maxTokens int, tools []map[string]any, noThinking bool) (string, []ToolCall, error) {
	var kwargs map[string]any
	if noThinking {
		kwargs = map[string]any{"enable_thinking": false}
	}
	reqBody, _ := json.Marshal(chatRequest{
		Messages:           msgs,
		MaxTokens:          maxTokens,
		Temperature:        0.2,
		Stream:             false,
		CachePrompt:        true,
		Tools:              tools,
		ChatTemplateKwargs: kwargs,
	})
	log.Printf("→ POST /v1/chat/completions port=%d max_tokens=%d msgs=%d", s.port, maxTokens, len(msgs))
	for i, m := range msgs {
		log.Printf("  msg[%d] %s (%d bytes): %s", i, m.Role, len(m.Content), truncateForLog(m.Content, 500))
	}

	url := fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", s.port)
	start := time.Now()
	if ctx == nil {
		ctx = context.Background()
	}
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return "", nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(req)
	if err != nil {
		// Cancellation is a user action, not a failure to report as one.
		if ctx.Err() != nil {
			return "", nil, ctx.Err()
		}
		return "", nil, fmt.Errorf("POST /v1/chat/completions: %w", err)
	}
	defer resp.Body.Close()

	raw, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return "", nil, fmt.Errorf("read chat completion body: %w", readErr)
	}
	log.Printf("← HTTP %d in %s (body=%d bytes)", resp.StatusCode, time.Since(start), len(raw))
	log.Printf("  body: %s", truncateForLog(string(raw), 2000))

	if resp.StatusCode != 200 {
		return "", nil, fmt.Errorf("llama-server HTTP %d: %s", resp.StatusCode, truncateForLog(string(raw), 300))
	}
	var cr chatResponse
	if err := json.Unmarshal(raw, &cr); err != nil {
		return "", nil, fmt.Errorf("decode chat completion: %w", err)
	}
	if len(cr.Choices) == 0 {
		return "", nil, fmt.Errorf("empty chat completion response")
	}
	msg := cr.Choices[0].Message
	content := msg.Content
	if content == "" && len(msg.ToolCalls) == 0 {
		// A reasoning model that spends its whole budget on <think> returns
		// an empty answer. Say so, rather than surfacing a blank reply.
		if msg.ReasoningContent != "" {
			log.Printf("  reply empty, reasoning_content=%d bytes, finish_reason=%s",
				len(msg.ReasoningContent), cr.Choices[0].FinishReason)
			return "", nil, fmt.Errorf(
				"the model used its entire %d-token budget on internal reasoning and never produced an answer "+
					"— raise it with `/set max_tokens N`", maxTokens)
		}
		if cr.Choices[0].FinishReason == "length" {
			log.Printf("  WARNING: reply was empty and finish_reason=length — max_tokens=%d likely too small", maxTokens)
		}
	}
	setLastUsage(UsageStats{
		PromptTokens:     cr.Usage.PromptTokens,
		CompletionTokens: cr.Usage.CompletionTokens,
		TotalTokens:      cr.Usage.TotalTokens,
		ContextSize:      s.ctxN,
	})
	return content, msg.ToolCalls, nil
}

// DropKVCache asks llama-server to forget any cached prompt prefix so the
// next request starts from a clean slate. Called from the TUI's /reset so
// the server doesn't silently reuse tokens from the previous conversation.
func (s *llamaServer) DropKVCache() error {
	// A stale prefix could survive in any slot, so erase them all.
	for i := 0; i < s.slots; i++ {
		url := fmt.Sprintf("http://127.0.0.1:%d/slots/%d?action=erase", s.port, i)
		req, err := http.NewRequest("POST", url, nil)
		if err != nil {
			return err
		}
		resp, err := s.client.Do(req)
		if err != nil {
			return err
		}
		resp.Body.Close()
		if resp.StatusCode >= 400 {
			// Older llama-server builds don't expose this; don't treat it as fatal.
			log.Printf("/slots erase returned %d (probably unsupported on this llama-server)", resp.StatusCode)
			return nil
		}
	}
	return nil
}

func truncateForLog(s string, n int) string {
	// Collapse newlines so each log line stays one line.
	s = strings.ReplaceAll(s, "\r", " ")
	s = strings.ReplaceAll(s, "\n", " ⏎ ")
	if len(s) > n {
		return s[:n] + "…(+" + fmt.Sprintf("%d", len(s)-n) + " more)"
	}
	return s
}

// logWriter forwards subprocess stdout/stderr into the atlas.llm log with a
// tag prefix so server messages show up alongside our own.
type logWriter struct{ tag string }

func newLogWriter(tag string) *logWriter { return &logWriter{tag: tag} }
func (w *logWriter) Write(p []byte) (int, error) {
	log.Printf("[%s] %s", w.tag, bytes.TrimRight(p, "\r\n "))
	return len(p), nil
}

// StreamDelta is one increment from a streaming completion. Reasoning
// models emit their thinking on a separate channel from the answer, so the
// two are kept apart: only Content belongs in the reply.
type StreamDelta struct {
	Content   string
	Reasoning string
}

// streamChunk is the SSE frame shape llama-server emits when stream=true.
type streamChunk struct {
	Choices []struct {
		Delta struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage chatUsage `json:"usage"`
	Error struct {
		Message string `json:"message"`
	} `json:"error"`
}

// ChatCompleteStream runs a completion with stream=true, invoking onDelta as
// tokens arrive, and returns the assembled answer.
//
// onDelta is called from this goroutine, so callers must not block in it;
// the TUI forwards each delta into the bubbletea event loop instead.
func (s *llamaServer) ChatCompleteStream(ctx context.Context, msgs []ChatMsg, maxTokens int, onDelta func(StreamDelta)) (string, error) {
	return s.ChatCompleteStreamOpt(ctx, msgs, maxTokens, false, onDelta)
}

// ChatCompleteStreamOpt is ChatCompleteStream with explicit control over the
// reasoning block.
func (s *llamaServer) ChatCompleteStreamOpt(ctx context.Context, msgs []ChatMsg, maxTokens int, noThinking bool, onDelta func(StreamDelta)) (string, error) {
	var kwargs map[string]any
	if noThinking {
		kwargs = map[string]any{"enable_thinking": false}
	}
	reqBody, _ := json.Marshal(chatRequest{
		Messages:           msgs,
		MaxTokens:          maxTokens,
		Temperature:        0.2,
		Stream:             true,
		CachePrompt:        true,
		ChatTemplateKwargs: kwargs,
	})
	url := fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", s.port)
	if ctx == nil {
		ctx = context.Background()
	}
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	log.Printf("→ POST /v1/chat/completions (stream) port=%d max_tokens=%d msgs=%d", s.port, maxTokens, len(msgs))
	start := time.Now()
	resp, err := s.client.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return "", ctx.Err()
		}
		return "", fmt.Errorf("POST /v1/chat/completions: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		raw, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("llama-server HTTP %d: %s", resp.StatusCode, truncateForLog(string(raw), 300))
	}

	var answer strings.Builder
	var reasoningLen int
	var usage chatUsage
	sc := bufio.NewScanner(resp.Body)
	// Individual SSE frames can exceed the default 64KB scanner limit.
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			break
		}
		var chunk streamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue // a frame we don't understand shouldn't kill the stream
		}
		if chunk.Error.Message != "" {
			return answer.String(), fmt.Errorf("llama-server: %s", chunk.Error.Message)
		}
		if chunk.Usage.TotalTokens > 0 {
			usage = chunk.Usage
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		d := chunk.Choices[0].Delta
		if d.Content == "" && d.ReasoningContent == "" {
			continue
		}
		answer.WriteString(d.Content)
		reasoningLen += len(d.ReasoningContent)
		if onDelta != nil {
			onDelta(StreamDelta{Content: d.Content, Reasoning: d.ReasoningContent})
		}
	}
	if err := sc.Err(); err != nil {
		if ctx.Err() != nil {
			return answer.String(), ctx.Err()
		}
		return answer.String(), fmt.Errorf("read stream: %w", err)
	}

	log.Printf("← stream complete in %s (%d bytes answer, %d bytes reasoning)",
		time.Since(start), answer.Len(), reasoningLen)
	if usage.TotalTokens > 0 {
		setLastUsage(UsageStats{
			PromptTokens:     usage.PromptTokens,
			CompletionTokens: usage.CompletionTokens,
			TotalTokens:      usage.TotalTokens,
			ContextSize:      s.ctxN,
		})
	}
	if answer.Len() == 0 && reasoningLen > 0 {
		return "", fmt.Errorf(
			"the model used its entire %d-token budget on internal reasoning and never produced an answer "+
				"— raise it with `/set max_tokens N`", maxTokens)
	}
	return answer.String(), nil
}

// chatCompleteRespectingReasoning is ChatComplete with the reasoning
// setting applied.
func (s *llamaServer) chatCompleteRespectingReasoning(ctx context.Context, msgs []ChatMsg, maxTokens int, cfg Config) (string, error) {
	content, _, err := s.chatCompleteCore(ctx, msgs, maxTokens, nil, !reasoningEnabled(cfg))
	return content, err
}

// ChatCompleteWithToolsOpt is ChatCompleteWithTools with explicit control
// over the reasoning block.
func (s *llamaServer) ChatCompleteWithToolsOpt(ctx context.Context, msgs []ChatMsg, tools []map[string]any, maxTokens int, noThinking bool) (string, []ToolCall, error) {
	return s.chatCompleteCore(ctx, msgs, maxTokens, tools, noThinking)
}
