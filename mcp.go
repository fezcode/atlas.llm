package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// MCPServerConfig describes one MCP server in mcp.json. The shape follows
// the de-facto Claude Desktop / VS Code convention so existing configs can
// be pasted in: `command`+`args`+`env` selects the stdio transport, `url`
// selects a remote HTTP transport.
type MCPServerConfig struct {
	// stdio transport — the server runs as a local subprocess.
	Command string            `json:"command,omitempty"`
	Args    []string          `json:"args,omitempty"`
	Env     map[string]string `json:"env,omitempty"`

	// Remote transport — streamable HTTP by default, SSE when the endpoint
	// only speaks the older 2024-11-05 protocol.
	URL     string            `json:"url,omitempty"`
	Headers map[string]string `json:"headers,omitempty"`
	// Transport optionally pins the remote transport: "http" (default) or "sse".
	Transport string `json:"transport,omitempty"`

	// OAuth opts the server into the interactive authorization-code flow.
	// Required for hosted servers like Atlassian's Confluence endpoint.
	OAuth  bool     `json:"oauth,omitempty"`
	Scopes []string `json:"scopes,omitempty"`
	// ClientID is optional; when empty atlas.llm uses Dynamic Client
	// Registration, which is what the hosted Atlassian/Slack servers expect.
	ClientID     string `json:"client_id,omitempty"`
	ClientSecret string `json:"client_secret,omitempty"`

	// Trust controls the confirm modal. When true, this server's tools run
	// without prompting. When false or absent, every call is confirmed.
	Trust bool `json:"trust,omitempty"`

	// Disabled keeps a server in the config without connecting to it.
	Disabled bool `json:"disabled,omitempty"`

	// StandaloneSSE opts a remote server back into the long-lived
	// server-initiated SSE stream, which is off by default. nil means off.
	StandaloneSSE *bool `json:"standalone_sse,omitempty"`

	// AuthServer names the OAuth authorization server explicitly, for
	// servers that publish no protected-resource metadata. Atlassian is one:
	// discovery falls back to treating the MCP host as the auth server, but
	// its metadata declares a different issuer (cf.mcp.atlassian.com), and
	// RFC 8414 requires those to match — so discovery fails with
	// "failed to get authorization server metadata".
	AuthServer string `json:"auth_server,omitempty"`
}

// standaloneSSEEnabled reports whether to open the optional server-initiated
// SSE stream. Off unless explicitly requested: atlas.llm doesn't act on
// server-initiated messages, and a server that won't hold the stream open
// takes the whole session down with it.
func (c MCPServerConfig) standaloneSSEEnabled() bool {
	return c.StandaloneSSE != nil && *c.StandaloneSSE
}

// isRemote reports whether this server is reached over HTTP rather than by
// spawning a subprocess.
func (c MCPServerConfig) isRemote() bool { return strings.TrimSpace(c.URL) != "" }

func (c MCPServerConfig) validate() error {
	if c.isRemote() && c.Command != "" {
		return fmt.Errorf("set either \"command\" (stdio) or \"url\" (remote), not both")
	}
	if !c.isRemote() && c.Command == "" {
		return fmt.Errorf("needs either \"command\" (stdio) or \"url\" (remote)")
	}
	if c.OAuth && !c.isRemote() {
		return fmt.Errorf("\"oauth\" only applies to remote (\"url\") servers")
	}
	switch strings.ToLower(c.Transport) {
	case "", "http", "streamable", "sse":
	default:
		return fmt.Errorf("unknown transport %q (expected \"http\" or \"sse\")", c.Transport)
	}
	return nil
}

// MCPConfig is the on-disk mcp.json document.
type MCPConfig struct {
	Servers map[string]MCPServerConfig `json:"mcpServers"`
}

func mcpConfigPath() (string, error) {
	base, err := atlasDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(base, "mcp.json"), nil
}

// loadMCPConfig reads mcp.json. A missing file is not an error — it just
// means no MCP servers are configured.
func loadMCPConfig() (MCPConfig, error) {
	cfg := MCPConfig{Servers: map[string]MCPServerConfig{}}
	p, err := mcpConfigPath()
	if err != nil {
		return cfg, err
	}
	data, err := os.ReadFile(p)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return cfg, err
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("parse %s: %w", p, err)
	}
	if cfg.Servers == nil {
		cfg.Servers = map[string]MCPServerConfig{}
	}
	return cfg, nil
}

// mcpConn is a live connection to one MCP server plus the tool names it
// contributed to the registry.
type mcpConn struct {
	name    string
	cfg     MCPServerConfig
	session *mcp.ClientSession
	tools   []string
}

var (
	// mcpMu guards both maps. Connections are established on background
	// goroutines while the TUI reads the tool registry, so every access is
	// serialized through here.
	mcpMu    sync.RWMutex
	mcpConns = map[string]*mcpConn{}
	mcpTools = map[string]Tool{}
)

// mcpConnectTimeout bounds a single server handshake. Remote servers doing
// an OAuth round-trip get their own, longer budget inside the auth flow.
//
// Sized for the worst honest case, not the usual one: an npx/uvx stdio
// server's first connect includes downloading the package, and a slow npm
// registry alone has been measured eating a whole minute of this budget.
// A hung server still fails — just later; a healthy-but-cold one connects.
const mcpConnectTimeout = 3 * time.Minute

// mcpCallTimeout bounds one tool invocation.
const mcpCallTimeout = 2 * time.Minute

// toolNameSanitizer strips characters the OpenAI tool-name grammar rejects.
// llama-server forwards names verbatim, and models are trained on
// ^[a-zA-Z0-9_-]+$, so anything else gets folded to '_'.
var toolNameSanitizer = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// mcpToolName namespaces a server's tool so two servers exposing `search`
// don't collide. Capped at 64 chars, the documented tool-name limit.
func mcpToolName(server, tool string) string {
	name := toolNameSanitizer.ReplaceAllString(server, "_") + "__" +
		toolNameSanitizer.ReplaceAllString(tool, "_")
	if len(name) > 64 {
		name = name[:64]
	}
	return name
}

// lookupTool resolves a tool by name across the built-in registry and every
// connected MCP server.
func lookupTool(name string) (Tool, bool) {
	if t, ok := toolRegistry[name]; ok {
		return t, true
	}
	// ask_user isn't in the registry (it's mode-gated and intercepted, not
	// run), but it must still resolve by name so a stray call is handled.
	if name == askUserToolName {
		return askUserTool, true
	}
	mcpMu.RLock()
	defer mcpMu.RUnlock()
	t, ok := mcpTools[name]
	return t, ok
}

// mcpToolSnapshot returns the connected MCP tools, sorted by name.
func mcpToolSnapshot() []Tool {
	mcpMu.RLock()
	defer mcpMu.RUnlock()
	out := make([]Tool, 0, len(mcpTools))
	for _, t := range mcpTools {
		out = append(out, t)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// buildTransport selects the transport for a server config, wiring OAuth in
// for remote servers that ask for it.
func buildTransport(ctx context.Context, name string, cfg MCPServerConfig) (mcp.Transport, error) {
	if !cfg.isRemote() {
		cmd := exec.Command(cfg.Command, cfg.Args...)
		cmd.Env = os.Environ()
		for k, v := range cfg.Env {
			cmd.Env = append(cmd.Env, k+"="+v)
		}
		// Keep the child's stderr in the atlas log rather than letting it
		// scribble over the TUI.
		cmd.Stderr = newLogWriter("mcp/" + name)
		applyEngineSysProcAttr(cmd)
		return &mcp.CommandTransport{Command: cmd}, nil
	}

	httpClient := mcpHTTPClient(cfg.Headers)
	if cfg.AuthServer != "" {
		httpClient = withDeclaredAuthServer(httpClient, cfg.URL, cfg.AuthServer)
	}
	if strings.EqualFold(cfg.Transport, "sse") {
		if cfg.OAuth {
			return nil, fmt.Errorf("oauth is only supported on the streamable HTTP transport; remove \"transport\": \"sse\"")
		}
		return &mcp.SSEClientTransport{Endpoint: cfg.URL, HTTPClient: httpClient}, nil
	}

	t := &mcp.StreamableClientTransport{
		Endpoint:   cfg.URL,
		HTTPClient: httpClient,
		// By default the SDK opens a long-lived GET stream for
		// server-initiated messages. atlas.llm never consumes those — tools
		// are snapshotted at connect time — so the stream is pure liability:
		// when a server doesn't hold it open (DeepWiki, among others), the
		// SDK's retries exhaust and it closes the whole session, failing any
		// in-flight tools/call with "exceeded 5 retries without progress".
		// Opt back in per server with "standalone_sse": true.
		DisableStandaloneSSE: !cfg.standaloneSSEEnabled(),
	}
	if cfg.OAuth {
		h, err := newOAuthHandler(ctx, name, cfg, httpClient)
		if err != nil {
			return nil, fmt.Errorf("oauth setup: %w", err)
		}
		t.OAuthHandler = h
	}
	return t, nil
}

// connectMCPServer establishes one session and registers its tools. It
// replaces any existing connection under the same name.
func connectMCPServer(ctx context.Context, name string, cfg MCPServerConfig) (int, error) {
	if err := cfg.validate(); err != nil {
		return 0, err
	}
	transport, err := buildTransport(ctx, name, cfg)
	if err != nil {
		return 0, err
	}

	client := mcp.NewClient(&mcp.Implementation{Name: "atlas.llm", Version: Version}, nil)
	session, err := client.Connect(ctx, transport, nil)
	if err != nil {
		return 0, fmt.Errorf("connect: %w", err)
	}

	res, err := session.ListTools(ctx, nil)
	if err != nil {
		_ = session.Close()
		return 0, fmt.Errorf("list tools: %w", err)
	}

	conn := &mcpConn{name: name, cfg: cfg, session: session}
	bridged := make(map[string]Tool, len(res.Tools))
	for _, mt := range res.Tools {
		t := bridgeMCPTool(name, cfg, session, mt)
		bridged[t.Name] = t
		conn.tools = append(conn.tools, t.Name)
	}
	sort.Strings(conn.tools)

	disconnectMCPServer(name)

	mcpMu.Lock()
	mcpConns[name] = conn
	for n, t := range bridged {
		mcpTools[n] = t
	}
	mcpMu.Unlock()

	log.Printf("mcp: connected %q with %d tools (trust=%v)", name, len(conn.tools), cfg.Trust)
	return len(conn.tools), nil
}

// bridgeMCPTool adapts an MCP tool descriptor into atlas.llm's Tool shape so
// it flows through the same tool-call loop and confirm modal as the built-ins.
func bridgeMCPTool(server string, cfg MCPServerConfig, session *mcp.ClientSession, mt *mcp.Tool) Tool {
	params, ok := mt.InputSchema.(map[string]any)
	if !ok || params == nil {
		// The model still needs a schema-shaped object even when the server
		// omits one, or llama-server rejects the tool definition.
		params = map[string]any{"type": "object", "properties": map[string]any{}}
	}

	desc := mt.Description
	if desc == "" {
		desc = mt.Title
	}
	desc = fmt.Sprintf("[mcp:%s] %s", server, strings.TrimSpace(desc))

	remote := mt.Name
	return Tool{
		Name:        mcpToolName(server, mt.Name),
		Description: desc,
		Parameters:  params,
		// Per-server trust: an untrusted server confirms every call. We
		// deliberately do not consult the server's own readOnlyHint here —
		// annotations come from the third party we're gating.
		Destructive: !cfg.Trust,
		// Resolve the session at call time rather than capturing it, so a
		// reconnect swaps the live session underneath already-registered
		// tools instead of leaving them pointed at a dead one.
		Run: func(args map[string]any) (string, error) {
			return callMCPToolOn(server, remote, args)
		},
	}
}

// currentSession returns the live session for a server, if connected.
func currentSession(server string) (*mcp.ClientSession, bool) {
	mcpMu.RLock()
	defer mcpMu.RUnlock()
	conn, ok := mcpConns[server]
	if !ok {
		return nil, false
	}
	return conn.session, true
}

// callMCPToolOn invokes a tool against whichever session is currently
// registered for the server.
//
// If the session has died — a dropped stream, an idle timeout, a server
// restart — the call fails once and the connection is re-established in the
// background so the next call works. The failed call is deliberately not
// retried automatically: "connection closed" doesn't tell us whether the
// server already processed the request, and silently re-running a tool that
// posts a message or edits a page could duplicate the side effect.
func callMCPToolOn(server, tool string, args map[string]any) (string, error) {
	session, ok := currentSession(server)
	if !ok {
		return "", fmt.Errorf("MCP server %q is not connected — run `/mcp connect %s`", server, server)
	}
	out, err := callMCPTool(session, tool, args)
	if err != nil && isMCPConnectionError(err) {
		go reconnectMCPServer(server)
		return "", fmt.Errorf("%w — the connection to %q dropped; it is reconnecting, "+
			"so try again (not retried automatically in case the server already acted on it)", err, server)
	}
	return out, err
}

// isMCPConnectionError distinguishes a dead transport from an error the tool
// itself reported.
func isMCPConnectionError(err error) bool {
	if errors.Is(err, mcp.ErrConnectionClosed) {
		return true
	}
	msg := strings.ToLower(err.Error())
	for _, s := range []string{
		"connection closed", "client is closing", "session not found",
		"broken pipe", "eof", "use of closed network connection",
	} {
		if strings.Contains(msg, s) {
			return true
		}
	}
	return false
}

// reconnectMCPServer re-establishes a dropped session using its config from
// mcp.json. Best effort — failures are logged, not surfaced mid-turn.
func reconnectMCPServer(server string) {
	cfg, err := loadMCPConfig()
	if err != nil {
		log.Printf("mcp: reconnect %q: %v", server, err)
		return
	}
	sc, ok := cfg.Servers[server]
	if !ok || sc.Disabled {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), mcpConnectTimeout)
	defer cancel()
	if _, err := connectMCPServer(ctx, server, sc); err != nil {
		log.Printf("mcp: reconnect %q failed: %v", server, err)
		return
	}
	log.Printf("mcp: reconnected %q after a dropped connection", server)
}

func callMCPTool(session *mcp.ClientSession, name string, args map[string]any) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), mcpCallTimeout)
	defer cancel()
	if args == nil {
		args = map[string]any{}
	}
	res, err := session.CallTool(ctx, &mcp.CallToolParams{Name: name, Arguments: args})
	if err != nil {
		return "", err
	}
	out := flattenMCPContent(res)
	if res.IsError {
		// Surface it as a tool error so the model can self-correct rather
		// than treating the message as a successful result.
		if out == "" {
			out = "tool reported an error"
		}
		return "", fmt.Errorf("%s", out)
	}
	if out == "" {
		return "(no output)", nil
	}
	return truncateForModel(out), nil
}

// flattenMCPContent renders an MCP result into the plain text we feed back
// to the model. Text content passes through; anything else is described or
// JSON-encoded so the model at least knows what came back.
func flattenMCPContent(res *mcp.CallToolResult) string {
	var b strings.Builder
	for _, c := range res.Content {
		switch v := c.(type) {
		case *mcp.TextContent:
			b.WriteString(v.Text)
			b.WriteByte('\n')
		case *mcp.ImageContent:
			fmt.Fprintf(&b, "[image: %s, %d bytes — not shown]\n", v.MIMEType, len(v.Data))
		case *mcp.AudioContent:
			fmt.Fprintf(&b, "[audio: %s, %d bytes — not shown]\n", v.MIMEType, len(v.Data))
		case *mcp.ResourceLink:
			fmt.Fprintf(&b, "[resource: %s (%s)]\n", v.URI, v.Name)
		default:
			if raw, err := json.Marshal(c); err == nil {
				b.Write(raw)
				b.WriteByte('\n')
			}
		}
	}
	if b.Len() == 0 && res.StructuredContent != nil {
		if raw, err := json.MarshalIndent(res.StructuredContent, "", "  "); err == nil {
			b.Write(raw)
		}
	}
	return strings.TrimSpace(b.String())
}

// disconnectMCPServer closes a session and removes its tools. Safe to call
// for a name that isn't connected.
func disconnectMCPServer(name string) bool {
	mcpMu.Lock()
	conn, ok := mcpConns[name]
	if ok {
		delete(mcpConns, name)
		for _, t := range conn.tools {
			delete(mcpTools, t)
		}
	}
	mcpMu.Unlock()
	if !ok {
		return false
	}
	if err := conn.session.Close(); err != nil {
		log.Printf("mcp: closing %q: %v", name, err)
	}
	return true
}

// mcpConnectResult reports the outcome of one server's connection attempt.
type mcpConnectResult struct {
	Name  string
	Tools int
	Err   error
}

// connectAllMCP connects every enabled server in mcp.json, in parallel.
// Servers are independent: one failure doesn't block the others.
func connectAllMCP(ctx context.Context) ([]mcpConnectResult, error) {
	cfg, err := loadMCPConfig()
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(cfg.Servers))
	for n, sc := range cfg.Servers {
		if !sc.Disabled {
			names = append(names, n)
		}
	}
	sort.Strings(names)

	results := make([]mcpConnectResult, len(names))
	var wg sync.WaitGroup
	for i, name := range names {
		wg.Add(1)
		go func(i int, name string) {
			defer wg.Done()
			// OAuth servers block on a browser round-trip, so they get the
			// caller's context rather than a short per-server deadline.
			cctx := ctx
			if !cfg.Servers[name].OAuth {
				var cancel context.CancelFunc
				cctx, cancel = context.WithTimeout(ctx, mcpConnectTimeout)
				defer cancel()
			}
			n, err := connectMCPServer(cctx, name, cfg.Servers[name])
			results[i] = mcpConnectResult{Name: name, Tools: n, Err: err}
		}(i, name)
	}
	wg.Wait()
	return results, nil
}

// autoConnectMCP connects configured servers in the background as the TUI
// starts, and reports the outcome into the viewport once it's up.
//
// Servers needing a fresh OAuth authorization are skipped: launching a
// browser unprompted at startup is hostile. Those connect on `/mcp connect`,
// or automatically here once credentials have been stored by an earlier run.
func autoConnectMCP() {
	cfg, err := loadMCPConfig()
	if err != nil {
		log.Printf("mcp: %v", err)
		return
	}
	var eligible, deferred []string
	for name, sc := range cfg.Servers {
		if sc.Disabled {
			continue
		}
		if sc.OAuth && !hasStoredMCPAuth(name) {
			deferred = append(deferred, name)
			continue
		}
		eligible = append(eligible, name)
	}
	sort.Strings(eligible)
	sort.Strings(deferred)
	if len(eligible) == 0 && len(deferred) == 0 {
		return
	}

	go func() {
		var results []mcpConnectResult
		for _, name := range eligible {
			ctx, cancel := context.WithTimeout(context.Background(), mcpConnectTimeout)
			n, err := connectMCPServer(ctx, name, cfg.Servers[name])
			cancel()
			results = append(results, mcpConnectResult{Name: name, Tools: n, Err: err})
		}
		if program == nil {
			return
		}
		var b strings.Builder
		if len(results) > 0 {
			b.WriteString(formatMCPResults(results))
		}
		if len(deferred) > 0 {
			if b.Len() > 0 {
				b.WriteByte('\n')
			}
			fmt.Fprintf(&b, "mcp: %s need authorization — run `/mcp connect %s`",
				strings.Join(deferred, ", "), deferred[0])
		}
		program.Send(sysMsg{content: b.String()})
	}()
}

// hasStoredMCPAuth reports whether a server already has credentials on disk.
func hasStoredMCPAuth(name string) bool {
	sa, err := loadMCPAuth(name)
	return err == nil && sa != nil && sa.Token != nil &&
		(sa.Token.RefreshToken != "" || sa.Token.Valid())
}

// shutdownMCP closes every session. Called from startChat's defer so stdio
// subprocesses don't outlive the TUI.
func shutdownMCP() {
	mcpMu.RLock()
	names := make([]string, 0, len(mcpConns))
	for n := range mcpConns {
		names = append(names, n)
	}
	mcpMu.RUnlock()
	for _, n := range names {
		disconnectMCPServer(n)
	}
}

// mcpStatusLines renders `/mcp` output: one line per configured server with
// its connection state, trust setting, and tool count.
func mcpStatusLines() string {
	cfg, err := loadMCPConfig()
	if err != nil {
		return "mcp: " + err.Error()
	}
	if len(cfg.Servers) == 0 {
		return "No MCP servers configured.\n\nRun `/mcp add` to pick one from the built-in list " +
			"(Atlassian/Confluence, Slack, GitHub, Linear, Sentry, …), or `/mcp help` for everything else."
	}
	names := make([]string, 0, len(cfg.Servers))
	for n := range cfg.Servers {
		names = append(names, n)
	}
	sort.Strings(names)

	mcpMu.RLock()
	defer mcpMu.RUnlock()

	var b strings.Builder
	b.WriteString("MCP servers:\n")
	for _, n := range names {
		sc := cfg.Servers[n]
		kind := "stdio"
		if sc.isRemote() {
			kind = "http"
			if strings.EqualFold(sc.Transport, "sse") {
				kind = "sse"
			}
			if sc.OAuth {
				kind += "+oauth"
			}
		}
		state := "disconnected"
		if sc.Disabled {
			state = "disabled"
		} else if conn, ok := mcpConns[n]; ok {
			state = fmt.Sprintf("connected, %d tools", len(conn.tools))
		}
		trust := "confirms each call"
		if sc.Trust {
			trust = "trusted"
		}
		fmt.Fprintf(&b, "  %-16s %-10s %-22s %s\n", n, kind, state, trust)
	}
	return b.String()
}

// handleMCP implements the `/mcp` slash command. Connection work happens in
// a tea.Cmd so a slow handshake (or an OAuth browser round-trip) doesn't
// block the UI goroutine.
func (m *chatModel) handleMCP(args []string) tea.Cmd {
	if len(args) == 0 {
		m.pushSystem(mcpStatusLines())
		return nil
	}
	switch strings.ToLower(args[0]) {
	case "help":
		m.pushSystem(mcpHelpText())
		return nil

	case "add":
		return m.handleMCPAdd(args[1:])

	case "remove", "rm":
		return m.handleMCPRemove(args[1:])

	case "trust":
		return m.handleMCPTrust(args[1:])

	case "env":
		return m.handleMCPEnv(args[1:])

	case "catalog":
		m.pushSystem(mcpCatalogText())
		return nil

	case "tools":
		tools := mcpToolSnapshot()
		if len(tools) == 0 {
			m.pushSystem("No MCP tools available. Use `/mcp connect` to connect configured servers.")
			return nil
		}
		var b strings.Builder
		b.WriteString("MCP tools:\n")
		for _, t := range tools {
			tag := "needs confirm"
			if !t.Destructive {
				tag = "trusted"
			}
			fmt.Fprintf(&b, "  %-32s [%s]  %s\n", t.Name, tag, t.Description)
		}
		b.WriteString("\nEnable tool-use with `/tools on` for the model to call these.")
		m.pushSystem(b.String())
		return nil

	case "connect":
		name := ""
		if len(args) > 1 {
			name = args[1]
		}
		m.pushSystem(mcpConnectingNotice(name))
		return mcpConnectCmd(name)

	case "disconnect":
		if len(args) < 2 {
			m.pushError("usage: /mcp disconnect NAME")
			return nil
		}
		if disconnectMCPServer(args[1]) {
			m.pushSystem(fmt.Sprintf("Disconnected MCP server %q; its tools are no longer offered.", args[1]))
		} else {
			m.pushError(fmt.Sprintf("MCP server %q is not connected", args[1]))
		}
		return nil

	case "logout":
		if len(args) < 2 {
			m.pushError("usage: /mcp logout NAME")
			return nil
		}
		removed, err := deleteMCPAuth(args[1])
		if err != nil {
			m.pushError("clear credentials: " + err.Error())
			return nil
		}
		disconnectMCPServer(args[1])
		if removed {
			m.pushSystem(fmt.Sprintf("Cleared stored OAuth credentials for %q. Next `/mcp connect` will re-authorize.", args[1]))
		} else {
			m.pushSystem(fmt.Sprintf("No stored credentials for %q.", args[1]))
		}
		return nil

	default:
		m.pushError(fmt.Sprintf("unknown /mcp arg: %s (expected add|remove|connect|disconnect|trust|env|tools|catalog|logout|help)", args[0]))
		return nil
	}
}

// openMCPPicker shows the built-in catalog so a user can add a server
// without knowing a package name or touching mcp.json.
func (m *chatModel) openMCPPicker() {
	m.mcpPickerItems = append([]mcpPreset(nil), mcpCatalog...)
	m.pickerIdx = 0
	m.picking = "mcp_add"
	m.renderMCPPicker()
}

func (m *chatModel) mcpPickerCancel() {
	m.picking = ""
	m.mcpPickerItems = nil
	m.pickerIdx = 0
	m.refresh()
	m.pushSystem("Cancelled.")
}

// mcpPickerConfirm adds the highlighted preset. Presets needing a token or
// a path can't be completed from a list, so those pre-fill the input box
// with the exact command to finish — still no file editing.
func (m *chatModel) mcpPickerConfirm() tea.Cmd {
	if m.pickerIdx < 0 || m.pickerIdx >= len(m.mcpPickerItems) {
		m.mcpPickerCancel()
		return nil
	}
	p := m.mcpPickerItems[m.pickerIdx]
	m.picking = ""
	m.mcpPickerItems = nil
	m.pickerIdx = 0
	m.refresh()

	if p.needsInput() {
		m.textarea.SetValue(presetAddCommand(p))
		var b strings.Builder
		fmt.Fprintf(&b, "%s needs a value before it can be added:\n\n", p.Label)
		for _, f := range p.RequiredEnv {
			fmt.Fprintf(&b, "  %-30s e.g. %s\n", f.Key, f.Hint)
		}
		if p.ArgHint != "" {
			fmt.Fprintf(&b, "  %-30s a directory to expose\n", p.ArgHint)
		}
		b.WriteString("\nThe command is filled in below — replace the placeholders and press Enter.")
		m.pushSystem(b.String())
		return nil
	}
	return m.addMCPServer(p.Key, p.Cfg, p.Label)
}

// addMCPServer writes an entry to mcp.json and connects it.
func (m *chatModel) addMCPServer(name string, sc MCPServerConfig, label string) tea.Cmd {
	if err := sc.validate(); err != nil {
		m.pushError(fmt.Sprintf("%s: %v", name, err))
		return nil
	}
	if err := upsertMCPServer(name, sc); err != nil {
		m.pushError("write mcp.json: " + err.Error())
		return nil
	}
	if label == "" {
		label = name
	}
	trust := "every call will ask for confirmation — `/mcp trust " + name + " on` to skip that"
	if sc.Trust {
		trust = "marked trusted; its tools run without confirmation"
	}
	m.pushSystem(fmt.Sprintf("Added %s to mcp.json.\n%s\n\nConnecting…", label, trust))
	return mcpConnectCmd(name)
}

// renderMCPPicker paints the catalog picker over the viewport, mirroring
// the model picker's look and key handling.
func (m *chatModel) renderMCPPicker() {
	title := brandStyle.Render("Add an MCP server") +
		sysStyle.Render("  (↑/↓ move · enter select · esc cancel)")
	rowSelected := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#0B1220")).
		Background(colAccent).
		Bold(true).
		Padding(0, 1)
	rowNormal := lipgloss.NewStyle().Padding(0, 1)

	configured := map[string]bool{}
	for _, n := range configuredMCPNames() {
		configured[n] = true
	}

	lines := []string{title, ""}
	for i, p := range m.mcpPickerItems {
		marker := "  "
		if configured[p.Key] {
			marker = brandStyle.Render("● ")
		}
		kind := "stdio"
		if p.Cfg.isRemote() {
			kind = "oauth"
			if !p.Cfg.OAuth {
				kind = "remote"
			}
		}
		row := fmt.Sprintf("%s%-30s %-8s %s", marker, p.Label, kind, p.Description)
		if i == m.pickerIdx {
			lines = append(lines, rowSelected.Render(row))
		} else {
			lines = append(lines, rowNormal.Render(row))
		}
	}
	lines = append(lines, "", sysStyle.Render("● = already in mcp.json · not listed? /mcp add NAME -- npx -y pkg"))
	m.viewport.SetContent(strings.Join(lines, "\n"))
	// Same reason as the model picker: pinning the view means entries below
	// the fold can be selected but never seen.
	m.scrollSelectionIntoView(pickerHeaderLines + m.pickerIdx)
}

// handleMCPAdd implements `/mcp add`.
func (m *chatModel) handleMCPAdd(args []string) tea.Cmd {
	if len(args) == 0 {
		m.openMCPPicker()
		return nil
	}
	// Custom definitions win over preset lookup so a user can shadow a
	// catalog key with their own endpoint.
	name, sc, custom, err := parseCustomAdd(args)
	if err != nil {
		m.pushError(err.Error())
		return nil
	}
	if custom {
		return m.addMCPServer(name, sc, "")
	}

	p, ok := findMCPPreset(args[0])
	if !ok {
		m.pushError(fmt.Sprintf("no built-in server named %q. `/mcp catalog` lists them, or:\n"+
			"  /mcp add %s -- npx -y some-mcp-package\n"+
			"  /mcp add %s --url=https://host/mcp --oauth", args[0], args[0], args[0]))
		return nil
	}
	built, err := buildPresetConfig(p, args[1:])
	if err != nil {
		m.textarea.SetValue(presetAddCommand(p))
		m.pushError(fmt.Sprintf("%s — %v\n\nCommand filled in below; replace the placeholders.", p.Label, err))
		return nil
	}
	return m.addMCPServer(p.Key, built, p.Label)
}

// handleMCPTrust implements `/mcp trust NAME [on|off]`.
func (m *chatModel) handleMCPTrust(args []string) tea.Cmd {
	if len(args) == 0 {
		m.pushError("usage: /mcp trust NAME [on|off]")
		return nil
	}
	cfg, err := loadMCPConfig()
	if err != nil {
		m.pushError(err.Error())
		return nil
	}
	name := args[0]
	sc, ok := cfg.Servers[name]
	if !ok {
		m.pushError(fmt.Sprintf("no server named %q in mcp.json", name))
		return nil
	}
	want := true
	if len(args) > 1 {
		switch strings.ToLower(args[1]) {
		case "on", "true", "yes":
			want = true
		case "off", "false", "no":
			want = false
		default:
			m.pushError("expected on or off")
			return nil
		}
	}
	sc.Trust = want
	if err := upsertMCPServer(name, sc); err != nil {
		m.pushError("write mcp.json: " + err.Error())
		return nil
	}
	if want {
		m.pushSystem(fmt.Sprintf("%s is now trusted — its tools run without confirmation. Reconnecting…", name))
	} else {
		m.pushSystem(fmt.Sprintf("%s is no longer trusted — every call will ask for confirmation. Reconnecting…", name))
	}
	// Trust is baked into each bridged tool at connect time, so reconnect
	// for the change to take effect on already-registered tools.
	return mcpConnectCmd(name)
}

// handleMCPEnv implements `/mcp env NAME KEY=VALUE ...`, for rotating a
// token without reopening the config file.
func (m *chatModel) handleMCPEnv(args []string) tea.Cmd {
	if len(args) < 2 {
		m.pushError("usage: /mcp env NAME KEY=VALUE [KEY=VALUE ...]")
		return nil
	}
	cfg, err := loadMCPConfig()
	if err != nil {
		m.pushError(err.Error())
		return nil
	}
	name := args[0]
	sc, ok := cfg.Servers[name]
	if !ok {
		m.pushError(fmt.Sprintf("no server named %q in mcp.json", name))
		return nil
	}
	kv, err := parseKeyValues(args[1:])
	if err != nil {
		m.pushError(err.Error())
		return nil
	}
	if sc.Env == nil {
		sc.Env = map[string]string{}
	}
	keys := make([]string, 0, len(kv))
	for k, v := range kv {
		sc.Env[k] = v
		keys = append(keys, k)
	}
	sort.Strings(keys)
	if err := upsertMCPServer(name, sc); err != nil {
		m.pushError("write mcp.json: " + err.Error())
		return nil
	}
	m.pushSystem(fmt.Sprintf("Updated %s on %s. Reconnecting…", strings.Join(keys, ", "), name))
	return mcpConnectCmd(name)
}

// handleMCPRemove implements `/mcp remove NAME`.
func (m *chatModel) handleMCPRemove(args []string) tea.Cmd {
	if len(args) == 0 {
		m.pushError("usage: /mcp remove NAME")
		return nil
	}
	name := args[0]
	disconnectMCPServer(name)
	removed, err := removeMCPServer(name)
	if err != nil {
		m.pushError("write mcp.json: " + err.Error())
		return nil
	}
	if !removed {
		m.pushError(fmt.Sprintf("no server named %q in mcp.json", name))
		return nil
	}
	// Stored OAuth credentials would otherwise linger for a server the
	// user just removed.
	if _, err := deleteMCPAuth(name); err != nil {
		log.Printf("mcp: clearing credentials for %q: %v", name, err)
	}
	m.pushSystem(fmt.Sprintf("Removed %s from mcp.json and dropped its tools.", name))
	return nil
}

func mcpConnectingNotice(name string) string {
	if name == "" {
		return "Connecting to configured MCP servers…"
	}
	return fmt.Sprintf("Connecting to MCP server %q…", name)
}

// mcpConnectCmd connects one server (or all, when name is empty) and reports
// the outcome back into the viewport.
func mcpConnectCmd(name string) tea.Cmd {
	return func() tea.Msg {
		// OAuth flows wait on a human in a browser, so allow for that.
		ctx, cancel := context.WithTimeout(context.Background(), mcpAuthTimeout)
		defer cancel()

		if name != "" {
			cfg, err := loadMCPConfig()
			if err != nil {
				return sysMsg{content: "mcp: " + err.Error()}
			}
			sc, ok := cfg.Servers[name]
			if !ok {
				return sysMsg{content: fmt.Sprintf("mcp: no server named %q in mcp.json", name)}
			}
			n, err := connectMCPServer(ctx, name, sc)
			if err != nil {
				return sysMsg{content: fmt.Sprintf("mcp: %s failed — %v", name, err)}
			}
			return sysMsg{content: fmt.Sprintf("mcp: %s connected (%d tools).", name, n)}
		}

		results, err := connectAllMCP(ctx)
		if err != nil {
			return sysMsg{content: "mcp: " + err.Error()}
		}
		return sysMsg{content: formatMCPResults(results)}
	}
}

// formatMCPResults renders the outcome of a batch connect.
func formatMCPResults(results []mcpConnectResult) string {
	if len(results) == 0 {
		return "mcp: no servers configured — run `/mcp add` to pick one"
	}
	var b strings.Builder
	b.WriteString("MCP connect:\n")
	for _, r := range results {
		if r.Err != nil {
			fmt.Fprintf(&b, "  %-16s failed — %v\n", r.Name, r.Err)
			continue
		}
		fmt.Fprintf(&b, "  %-16s connected, %d tools\n", r.Name, r.Tools)
	}
	return strings.TrimRight(b.String(), "\n")
}

// mcpHelpText documents the config format for `/mcp help`.
func mcpHelpText() string {
	p, _ := mcpConfigPath()
	return "MCP — connect atlas.llm to external tool servers.\n\n" +
		"Start here:\n" +
		"  /mcp add             pick a server from the built-in list\n" +
		"  /tools on            let the model actually call the tools\n\n" +
		"Managing servers:\n" +
		"  /mcp                 show configured servers and connection state\n" +
		"  /mcp catalog         list the built-in servers you can add\n" +
		"  /mcp add NAME        add a built-in by name (e.g. /mcp add slack)\n" +
		"  /mcp remove NAME     delete a server and drop its tools\n" +
		"  /mcp trust NAME on   run that server's tools without confirmation\n" +
		"  /mcp env NAME K=V    set or rotate a token\n\n" +
		"Connections:\n" +
		"  /mcp connect [NAME]  (re)connect every enabled server, or one\n" +
		"  /mcp disconnect NAME drop a server and its tools for this session\n" +
		"  /mcp tools           list the tools MCP servers are contributing\n" +
		"  /mcp logout NAME     forget a server's stored OAuth tokens\n\n" +
		"Anything not in the catalog:\n" +
		"  /mcp add NAME -- npx -y some-mcp-package\n" +
		"  /mcp add NAME --url=https://host/mcp --oauth\n" +
		"  flags: --oauth (browser authorization), --sse (older protocol),\n" +
		"         --trust (skip confirmation)\n\n" +
		"These commands write " + p + " for you; you can also edit it by hand."
}
