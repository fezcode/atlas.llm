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

// llamaServer wraps the HTTP endpoint that answers inference requests, and
// for a local server the subprocess behind it. One instance per (model,
// context-size); the model stays loaded so subsequent calls don't pay the
// GGUF mmap + warmup cost on every turn.
//
// A llamaServer is either local — a subprocess we spawned and own — or
// remote, an endpoint someone else is running. Every field below except cmd
// and waitErr means the same thing in both cases, because inference is HTTP
// either way; what differs is lifecycle. cmd == nil marks the remote case,
// and the rule that follows from it is: never kill what we didn't start.
type llamaServer struct {
	cmd      *exec.Cmd
	base     string // scheme://host:port, no trailing slash
	port     int
	model    Model
	ctxN     int // per-slot context — what one conversation can use
	slots    int // llama-server slots (-np); total -c is ctxN*slots
	gpuLayer int
	cpuMoE   int // --n-cpu-moe layers, 0 when the flag was not passed
	// gpuSetting is the gpu_layers *setting* this server was started under,
	// not the resolved layer count. See serverMatches.
	gpuSetting int
	apiKey     string
	client     *http.Client
	waitOnce   sync.Once
	waitErr    chan error
}

// isRemote reports whether this server is someone else's process. Remote
// servers are shared, so anything destructive — killing the process, erasing
// KV slots — has to be suppressed for them.
func (s *llamaServer) isRemote() bool { return s.cmd == nil }

// url builds an absolute request URL. Every caller goes through this rather
// than formatting a host, so there is one place that knows where the model
// lives.
func (s *llamaServer) url(path string) string {
	return s.base + path
}

// do is the single exit point for every request, so a remote's API key is
// attached in exactly one place instead of at each call site.
func (s *llamaServer) do(req *http.Request) (*http.Response, error) {
	if s.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+s.apiKey)
	}
	return s.client.Do(req)
}

var (
	serverMu     sync.Mutex
	activeServer *llamaServer
)

// gpuSettingAuto marks "gpu_layers is on auto" in a server's identity.
const gpuSettingAuto = -1

// gpuLayersSetting collapses the gpu_layers config into a comparable int:
// what the user asked for, not what it resolved to.
func gpuLayersSetting(cfg Config) int {
	if cfg.GPULayers == nil {
		return gpuSettingAuto
	}
	if *cfg.GPULayers < 0 {
		return 0
	}
	return *cfg.GPULayers
}

// serverMatches reports whether a running server can take the next request
// without being restarted.
//
// It compares the gpu_layers *setting*, never a freshly resolved layer
// count. Resolving reads live VRAM, and once a server is up, live VRAM
// includes the several gigabytes that server is holding. Comparing against
// that asks "would this model fit in a machine that is already running it",
// which answers no — so the server is torn down, which frees the VRAM that
// made it look too big, so the replacement loads at full offload, and the
// next message repeats it. That loop reloaded the model from disk on every
// tool round of an agent turn.
func serverMatches(s *llamaServer, modelName string, ctxN, gpuSetting int) bool {
	return s != nil && !s.isRemote() &&
		s.model.Name == modelName && s.ctxN == ctxN && s.gpuSetting == gpuSetting
}

// ensureServer returns a running llamaServer for the current model, spawning
// one if needed and restarting if the caller switched models. Safe to call
// concurrently.
func ensureServer() (*llamaServer, error) {
	serverMu.Lock()
	defer serverMu.Unlock()

	// A configured endpoint takes over entirely: no engine, no model file,
	// no subprocess. Model and ctx are the remote's to decide, so none of
	// the local-identity checks below apply.
	if cfgEndpoint, key := remoteEndpoint(); cfgEndpoint != "" {
		if activeServer != nil && activeServer.isRemote() && activeServer.base == cfgEndpoint {
			return activeServer, nil
		}
		if activeServer != nil {
			activeServer.stopLocked()
			activeServer = nil
		}
		s, err := newRemoteServer(cfgEndpoint, key)
		if err != nil {
			return nil, err
		}
		activeServer = s
		return s, nil
	}

	m, err := currentModel()
	if err != nil {
		return nil, err
	}
	// Offload only takes effect at process start, so it is part of the
	// server's identity alongside the model and context.
	cfg, _ := loadConfig()
	if activeServer != nil && serverMatches(activeServer, m.Name, resolveCtxSize(cfg), gpuLayersSetting(cfg)) {
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

// serveCapacity decides the slot count and the total -c for a launch.
//
// The KV budget is what ctx_size implies at the default slot count, and it
// does not grow with the slot count: extra slots divide that budget rather
// than multiplying it. Serving four clients must not silently quadruple VRAM
// use and fail to load the model — each client gets less context instead,
// which is the honest trade and the one the operator can see.
//
// wantSlots of 0 means "the default", which is what the TUI's own private
// server always asks for.
func serveCapacity(ctxN, wantSlots int) (slots, totalCtx int) {
	slots = kvCacheSlots
	if wantSlots > 0 {
		slots = wantSlots
	}
	return slots, ctxN * kvCacheSlots
}

func startLlamaServer(m Model) (*llamaServer, error) {
	return startLlamaServerWith(m, serveOptions{})
}

// startLlamaServerWith spawns the engine, retrying without the optimized
// flag set if this engine/model/backend combination rejects it. opts is the
// zero value for the TUI's private loopback server and carries the bind
// address for `--serve`.
func startLlamaServerWith(m Model, opts serveOptions) (*llamaServer, error) {
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
	off := resolveOffload(cfg)
	ctxN := resolveCtxSize(cfg)

	s, err := launchLlamaServerOn(bin, modelPath, m, threads, off, ctxN, true, opts)
	if errors.Is(err, errServerExitedEarly) {
		// The optimized flag set is rejected by engines that predate the
		// `-fa on` syntax, by backend/model combos that can't force flash
		// attention, and by builds older than --n-cpu-moe. The base flags
		// work everywhere. Dropping --n-cpu-moe means the model may no
		// longer fit with everything offloaded, so the retry falls back to
		// plain layer offload too.
		log.Printf("llama-server rejected optimized flags (%v) — retrying with base flags", err)
		s, err = launchLlamaServerOn(bin, modelPath, m, threads, baseOffload(cfg, off), ctxN, false, opts)
	}
	return s, err
}

// baseOffload is the offload plan for the fallback launch, which cannot pass
// --n-cpu-moe. Without it the experts come back to the GPU, so the layer
// count has to absorb them instead.
func baseOffload(cfg Config, off offloadPlan) offloadPlan {
	if off.CPUMoE == 0 {
		return off
	}
	if m, err := currentModel(); err == nil {
		if n, total, ok := fitGPULayers(m, resolveCtxSize(cfg)); ok && n < total {
			return offloadPlan{NGL: n, Setting: off.Setting}
		}
	}
	return offloadPlan{NGL: off.NGL, Setting: off.Setting}
}

func launchLlamaServer(bin, modelPath string, m Model, threads, ngl, ctxN int, optimized bool) (*llamaServer, error) {
	return launchLlamaServerOn(bin, modelPath, m, threads, offloadPlan{NGL: ngl}, ctxN, optimized, serveOptions{})
}

// serveOptions carries the LAN-facing overrides used by `--serve`. The zero
// value is the private, loopback-only server the TUI has always started.
type serveOptions struct {
	Bind   string // listen address; "" means 127.0.0.1
	Port   int    // fixed port; 0 means pick a free one
	Slots  int    // llama-server -np; 0 means kvCacheSlots
	APIKey string
}

func (o serveOptions) bindHost() string {
	if o.Bind == "" {
		return "127.0.0.1"
	}
	return o.Bind
}

// clientHost is the host a request should be addressed to. A server bound to
// a wildcard is not reachable *at* that wildcard, so requests from this
// process still go to loopback.
func (o serveOptions) clientHost() string {
	switch o.bindHost() {
	case "0.0.0.0", "::", "[::]":
		return "127.0.0.1"
	}
	return o.bindHost()
}

func launchLlamaServerOn(bin, modelPath string, m Model, threads int, off offloadPlan, ctxN int, optimized bool, opts serveOptions) (*llamaServer, error) {
	port := opts.Port
	if port == 0 {
		var err error
		if port, err = pickFreePort(); err != nil {
			return nil, fmt.Errorf("pick port: %w", err)
		}
	}

	slots := 1
	serverCtx := ctxN // -c is the total across slots; each slot gets an equal share
	args := []string{
		"-m", modelPath,
		"--host", opts.bindHost(),
		"--port", fmt.Sprintf("%d", port),
		"-t", fmt.Sprintf("%d", threads),
		"-ngl", fmt.Sprintf("%d", off.NGL),
		// Not --log-disable. That discarded llama.cpp's own account of
		// itself — the model load, the slot layout, and the reason behind a
		// failed start — leaving a bare "exit status 1" as the only symptom.
		// Default verbosity is about a dozen lines per start; stdout and
		// stderr are piped into the atlas log, so none of it reaches the
		// terminal.
		// Colours would arrive as ANSI escapes in a log file.
		"--log-colors", "off",
	}
	if opts.APIKey != "" {
		args = append(args, "--api-key", opts.APIKey)
	}
	if optimized {
		slots, serverCtx = serveCapacity(ctxN, opts.Slots)
		if off.CPUMoE > 0 {
			// Keep the experts of the first N layers in system RAM while
			// everything else stays on the GPU. For a mixture-of-experts
			// model this beats dropping whole layers: attention runs for
			// every token, expert weights mostly do not.
			args = append(args, "--n-cpu-moe", fmt.Sprintf("%d", off.CPUMoE))
		}
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
	// -c is the total across slots, so what one conversation actually gets
	// is the share, not the total.
	perSlot := serverCtx / slots
	cmd := exec.Command(bin, args...)
	cmd.Stdin = bytes.NewReader(nil)
	cmd.Stdout = newLogWriter("llama-server stdout")
	cmd.Stderr = newLogWriter("llama-server stderr")
	cmd.Env = append(os.Environ(), "OMP_STACKSIZE=64M")
	applyEngineSysProcAttr(cmd)

	log.Printf("starting llama-server on :%d (model=%s threads=%d ngl=%d cpu_moe=%d ctx=%d slots=%d optimized=%v)",
		port, m.Name, threads, off.NGL, off.CPUMoE, perSlot, slots, optimized)
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start llama-server: %w", err)
	}

	s := &llamaServer{
		cmd:        cmd,
		base:       fmt.Sprintf("http://%s:%d", opts.clientHost(), port),
		port:       port,
		model:      m,
		ctxN:       perSlot,
		slots:      slots,
		gpuLayer:   off.NGL,
		cpuMoE:     off.CPUMoE,
		gpuSetting: off.Setting,
		apiKey:     opts.APIKey,
		client:     &http.Client{Timeout: 10 * time.Minute},
		waitErr:    make(chan error, 1),
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
	url := s.url("/health")

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
		resp, err := s.do(req)
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
	if s.isRemote() {
		// A local server not coming up is a startup failure; a remote one is
		// almost always a wrong address, a firewall, or a host that isn't
		// serving — say that instead of "did not become ready".
		return fmt.Errorf("no atlas.llm server answering at %s — check it is running with --serve "+
			"and that the port is reachable", s.base)
	}
	return fmt.Errorf("llama-server did not become ready in %s", maxWait)
}

// remoteReadyTimeout is short on purpose: a remote is either already serving
// or it isn't, so there is nothing to wait out the way a local model load
// has to be waited out.
const remoteReadyTimeout = 5 * time.Second

// remoteProps is the subset of llama-server's /props we can use to describe a
// server we did not configure.
type remoteProps struct {
	DefaultGenerationSettings struct {
		NCtx int `json:"n_ctx"`
	} `json:"default_generation_settings"`
	ModelPath  string `json:"model_path"`
	BuildInfo  string `json:"build_info"`
	TotalSlots int    `json:"total_slots"`
}

// newRemoteServer attaches to a llama-server someone else is running. It owns
// no process, so it never sets cmd — see isRemote.
func newRemoteServer(endpoint, apiKey string) (*llamaServer, error) {
	s := &llamaServer{
		base:   strings.TrimRight(endpoint, "/"),
		apiKey: apiKey,
		slots:  1,
		client: &http.Client{Timeout: 10 * time.Minute},
	}
	if err := s.waitReady(remoteReadyTimeout); err != nil {
		return nil, err
	}
	s.model = Model{Name: remoteModelName(s), Size: "remote"}
	s.ctxN = remoteContext(s)
	log.Printf("attached to remote server %s (model=%s ctx=%d)", s.base, s.model.Name, s.ctxN)
	return s, nil
}

// fetchProps reads /props, which is how a client learns what it is talking to
// — there is no local GGUF to inspect.
func (s *llamaServer) fetchProps() (remoteProps, bool) {
	var p remoteProps
	req, err := http.NewRequest("GET", s.url("/props"), nil)
	if err != nil {
		return p, false
	}
	resp, err := s.do(req)
	if err != nil {
		return p, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return p, false
	}
	if err := json.NewDecoder(resp.Body).Decode(&p); err != nil {
		return p, false
	}
	return p, true
}

func remoteModelName(s *llamaServer) string {
	p, ok := s.fetchProps()
	if !ok || strings.TrimSpace(p.ModelPath) == "" {
		return "remote"
	}
	// model_path is the server's filesystem path; only the basename is
	// meaningful here, and the .gguf suffix is noise in a header.
	name := p.ModelPath
	if i := strings.LastIndexAny(name, `/\`); i >= 0 {
		name = name[i+1:]
	}
	return strings.TrimSuffix(name, ".gguf")
}

func remoteContext(s *llamaServer) int {
	if p, ok := s.fetchProps(); ok && p.DefaultGenerationSettings.NCtx > 0 {
		return p.DefaultGenerationSettings.NCtx
	}
	return 0
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
	log.Printf("→ POST %s max_tokens=%d msgs=%d", s.url("/v1/chat/completions"), maxTokens, len(msgs))
	for i, m := range msgs {
		log.Printf("  msg[%d] %s (%d bytes): %s", i, m.Role, len(m.Content), truncateForLog(m.Content, 500))
	}

	url := s.url("/v1/chat/completions")
	start := time.Now()
	if ctx == nil {
		ctx = context.Background()
	}
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return "", nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.do(req)
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
	if s.isRemote() {
		// Slots are shared on a served engine, so erasing them would throw
		// away other clients' cached prefixes and make their next message
		// re-prefill from scratch. The caller has already dropped this
		// conversation's history, which is the part that's ours to drop.
		log.Printf("/reset: leaving slots on remote %s alone (shared with other clients)", s.base)
		return nil
	}
	// A stale prefix could survive in any slot, so erase them all.
	for i := 0; i < s.slots; i++ {
		url := s.url(fmt.Sprintf("/slots/%d?action=erase", i))
		req, err := http.NewRequest("POST", url, nil)
		if err != nil {
			return err
		}
		resp, err := s.do(req)
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
//
// It buffers to line boundaries rather than logging each Write: llama.cpp
// emits a line's timestamp and its text as separate writes, so logging chunks
// verbatim split every message across two entries.
type logWriter struct {
	tag string
	mu  sync.Mutex
	buf []byte
}

// logWriterMaxBuffer bounds the partial-line buffer, so a subprocess that
// somehow never emits a newline can't grow it without limit.
const logWriterMaxBuffer = 64 << 10

func newLogWriter(tag string) *logWriter { return &logWriter{tag: tag} }

func (w *logWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.buf = append(w.buf, p...)
	for {
		i := bytes.IndexByte(w.buf, '\n')
		if i < 0 {
			break
		}
		w.emit(w.buf[:i])
		w.buf = w.buf[i+1:]
	}
	if len(w.buf) > logWriterMaxBuffer {
		w.emit(w.buf)
		w.buf = w.buf[:0]
	}
	return len(p), nil
}

func (w *logWriter) emit(line []byte) {
	if line = bytes.TrimRight(line, "\r \t"); len(line) > 0 {
		log.Printf("[%s] %s", w.tag, line)
	}
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
	url := s.url("/v1/chat/completions")
	if ctx == nil {
		ctx = context.Background()
	}
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	log.Printf("→ POST %s (stream) max_tokens=%d msgs=%d", s.url("/v1/chat/completions"), maxTokens, len(msgs))
	start := time.Now()
	resp, err := s.do(req)
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
