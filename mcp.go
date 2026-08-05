package main

import (
	"context"
	"encoding/json"
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
const mcpConnectTimeout = 60 * time.Second

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
	if strings.EqualFold(cfg.Transport, "sse") {
		if cfg.OAuth {
			return nil, fmt.Errorf("oauth is only supported on the streamable HTTP transport; remove \"transport\": \"sse\"")
		}
		return &mcp.SSEClientTransport{Endpoint: cfg.URL, HTTPClient: httpClient}, nil
	}

	t := &mcp.StreamableClientTransport{Endpoint: cfg.URL, HTTPClient: httpClient}
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
		Run: func(args map[string]any) (string, error) {
			return callMCPTool(session, remote, args)
		},
	}
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
		p, _ := mcpConfigPath()
		return "No MCP servers configured.\n\nCreate " + p + " — see `/mcp help` for the format."
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
		m.pushError(fmt.Sprintf("unknown /mcp arg: %s (expected connect|disconnect|tools|logout|help)", args[0]))
		return nil
	}
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
		p, _ := mcpConfigPath()
		return "mcp: no servers configured — create " + p + " (see `/mcp help`)"
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
		"Commands:\n" +
		"  /mcp                 show configured servers and connection state\n" +
		"  /mcp connect         (re)connect every enabled server\n" +
		"  /mcp connect NAME    (re)connect one server\n" +
		"  /mcp disconnect NAME drop a server and its tools\n" +
		"  /mcp tools           list the tools MCP servers are contributing\n" +
		"  /mcp logout NAME     forget a server's stored OAuth tokens\n\n" +
		"Config lives at " + p + ":\n\n" +
		`{
  "mcpServers": {
    "slack": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-slack"],
      "env": { "SLACK_BOT_TOKEN": "xoxb-...", "SLACK_TEAM_ID": "T..." },
      "trust": false
    },
    "confluence": {
      "url": "https://mcp.atlassian.com/v1/mcp",
      "oauth": true,
      "trust": true
    }
  }
}` + "\n\n" +
		"\"trust\": true runs that server's tools without confirmation.\n" +
		"Omit it (or set false) and every call opens the confirm modal.\n" +
		"\"oauth\": true opens a browser to authorize; tokens are stored per server."
}
