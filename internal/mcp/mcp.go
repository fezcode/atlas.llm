package mcpx

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

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"atlas.llm/internal/buildinfo"
	"atlas.llm/internal/config"
	"atlas.llm/internal/engine"
	"atlas.llm/internal/tools"
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
func (c MCPServerConfig) IsRemote() bool { return strings.TrimSpace(c.URL) != "" }

func (c MCPServerConfig) Validate() error {
	if c.IsRemote() && c.Command != "" {
		return fmt.Errorf("set either \"command\" (stdio) or \"url\" (remote), not both")
	}
	if !c.IsRemote() && c.Command == "" {
		return fmt.Errorf("needs either \"command\" (stdio) or \"url\" (remote)")
	}
	if c.OAuth && !c.IsRemote() {
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

func McpConfigPath() (string, error) {
	base, err := config.AtlasDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(base, "mcp.json"), nil
}

// loadMCPConfig reads mcp.json. A missing file is not an error — it just
// means no MCP servers are configured.
func LoadMCPConfig() (MCPConfig, error) {
	cfg := MCPConfig{Servers: map[string]MCPServerConfig{}}
	p, err := McpConfigPath()
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
	McpMu    sync.RWMutex
	McpConns = map[string]*mcpConn{}
	McpTools = map[string]tools.Tool{}
)

// mcpConnectTimeout bounds a single server handshake. Remote servers doing
// an OAuth round-trip get their own, longer budget inside the auth flow.
//
// Sized for the worst honest case, not the usual one: an npx/uvx stdio
// server's first connect includes downloading the package, and a slow npm
// registry alone has been measured eating a whole minute of this budget.
// A hung server still fails — just later; a healthy-but-cold one connects.
const McpConnectTimeout = 3 * time.Minute

// mcpCallTimeout bounds one tool invocation.
const mcpCallTimeout = 2 * time.Minute

// toolNameSanitizer strips characters the OpenAI tool-name grammar rejects.
// llama-server forwards names verbatim, and models are trained on
// ^[a-zA-Z0-9_-]+$, so anything else gets folded to '_'.
var toolNameSanitizer = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// mcpToolName namespaces a server's tool so two servers exposing `search`
// don't collide. Capped at 64 chars, the documented tool-name limit.
func McpToolName(server, tool string) string {
	name := toolNameSanitizer.ReplaceAllString(server, "_") + "__" +
		toolNameSanitizer.ReplaceAllString(tool, "_")
	if len(name) > 64 {
		name = name[:64]
	}
	return name
}

// lookupTool resolves a tool by name across the built-in registry and every
// connected MCP server.
func LookupTool(name string) (tools.Tool, bool) {
	if t, ok := tools.ToolRegistry[name]; ok {
		return t, true
	}
	// ask_user isn't in the registry (it's mode-gated and intercepted, not
	// run), but it must still resolve by name so a stray call is handled.
	if name == tools.AskUserToolName {
		return tools.AskUserTool, true
	}
	McpMu.RLock()
	defer McpMu.RUnlock()
	t, ok := McpTools[name]
	return t, ok
}

// The tool registry asks for MCP tools through a hook so that it need not
// know about this file.
func init() { tools.ExtraTools = McpToolSnapshot }

// mcpToolSnapshot returns the connected MCP tools, sorted by name.
func McpToolSnapshot() []tools.Tool {
	McpMu.RLock()
	defer McpMu.RUnlock()
	out := make([]tools.Tool, 0, len(McpTools))
	for _, t := range McpTools {
		out = append(out, t)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// buildTransport selects the transport for a server config, wiring OAuth in
// for remote servers that ask for it.
func buildTransport(ctx context.Context, name string, cfg MCPServerConfig) (mcp.Transport, error) {
	if !cfg.IsRemote() {
		cmd := exec.Command(cfg.Command, cfg.Args...)
		cmd.Env = os.Environ()
		for k, v := range cfg.Env {
			cmd.Env = append(cmd.Env, k+"="+v)
		}
		// Keep the child's stderr in the atlas log rather than letting it
		// scribble over the TUI.
		cmd.Stderr = engine.NewLogWriter("mcp/" + name)
		engine.ApplyEngineSysProcAttr(cmd)
		return &mcp.CommandTransport{Command: cmd}, nil
	}

	httpClient := mcpHTTPClient(cfg.Headers)
	if cfg.AuthServer != "" {
		httpClient = WithDeclaredAuthServer(httpClient, cfg.URL, cfg.AuthServer)
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
func ConnectMCPServer(ctx context.Context, name string, cfg MCPServerConfig) (int, error) {
	if err := cfg.Validate(); err != nil {
		return 0, err
	}
	transport, err := buildTransport(ctx, name, cfg)
	if err != nil {
		return 0, err
	}

	client := mcp.NewClient(&mcp.Implementation{Name: "atlas.llm", Version: buildinfo.Version}, nil)
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
	bridged := make(map[string]tools.Tool, len(res.Tools))
	for _, mt := range res.Tools {
		t := bridgeMCPTool(name, cfg, session, mt)
		bridged[t.Name] = t
		conn.tools = append(conn.tools, t.Name)
	}
	sort.Strings(conn.tools)

	DisconnectMCPServer(name)

	McpMu.Lock()
	McpConns[name] = conn
	for n, t := range bridged {
		McpTools[n] = t
	}
	McpMu.Unlock()

	log.Printf("mcp: connected %q with %d tools (trust=%v)", name, len(conn.tools), cfg.Trust)
	return len(conn.tools), nil
}

// bridgeMCPTool adapts an MCP tool descriptor into atlas.llm's Tool shape so
// it flows through the same tool-call loop and confirm modal as the built-ins.
func bridgeMCPTool(server string, cfg MCPServerConfig, session *mcp.ClientSession, mt *mcp.Tool) tools.Tool {
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
	return tools.Tool{
		Name:        McpToolName(server, mt.Name),
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
	McpMu.RLock()
	defer McpMu.RUnlock()
	conn, ok := McpConns[server]
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
	cfg, err := LoadMCPConfig()
	if err != nil {
		log.Printf("mcp: reconnect %q: %v", server, err)
		return
	}
	sc, ok := cfg.Servers[server]
	if !ok || sc.Disabled {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), McpConnectTimeout)
	defer cancel()
	if _, err := ConnectMCPServer(ctx, server, sc); err != nil {
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
	out := FlattenMCPContent(res)
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
	return tools.TruncateForModel(out), nil
}

// flattenMCPContent renders an MCP result into the plain text we feed back
// to the model. Text content passes through; anything else is described or
// JSON-encoded so the model at least knows what came back.
func FlattenMCPContent(res *mcp.CallToolResult) string {
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
func DisconnectMCPServer(name string) bool {
	McpMu.Lock()
	conn, ok := McpConns[name]
	if ok {
		delete(McpConns, name)
		for _, t := range conn.tools {
			delete(McpTools, t)
		}
	}
	McpMu.Unlock()
	if !ok {
		return false
	}
	if err := conn.session.Close(); err != nil {
		log.Printf("mcp: closing %q: %v", name, err)
	}
	return true
}

// mcpConnectResult reports the outcome of one server's connection attempt.
type McpConnectResult struct {
	Name  string
	Tools int
	Err   error
}

// connectAllMCP connects every enabled server in mcp.json, in parallel.
// Servers are independent: one failure doesn't block the others.
func ConnectAllMCP(ctx context.Context) ([]McpConnectResult, error) {
	cfg, err := LoadMCPConfig()
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

	results := make([]McpConnectResult, len(names))
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
				cctx, cancel = context.WithTimeout(ctx, McpConnectTimeout)
				defer cancel()
			}
			n, err := ConnectMCPServer(cctx, name, cfg.Servers[name])
			results[i] = McpConnectResult{Name: name, Tools: n, Err: err}
		}(i, name)
	}
	wg.Wait()
	return results, nil
}

// hasStoredMCPAuth reports whether a server already has credentials on disk.
func HasStoredMCPAuth(name string) bool {
	sa, err := LoadMCPAuth(name)
	return err == nil && sa != nil && sa.Token != nil &&
		(sa.Token.RefreshToken != "" || sa.Token.Valid())
}

// shutdownMCP closes every session. Called from startChat's defer so stdio
// subprocesses don't outlive the TUI.
func ShutdownMCP() {
	McpMu.RLock()
	names := make([]string, 0, len(McpConns))
	for n := range McpConns {
		names = append(names, n)
	}
	McpMu.RUnlock()
	for _, n := range names {
		DisconnectMCPServer(n)
	}
}

// mcpStatusLines renders `/mcp` output: one line per configured server with
// its connection state, trust setting, and tool count.
func McpStatusLines() string {
	cfg, err := LoadMCPConfig()
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

	McpMu.RLock()
	defer McpMu.RUnlock()

	var b strings.Builder
	b.WriteString("MCP servers:\n")
	for _, n := range names {
		sc := cfg.Servers[n]
		kind := "stdio"
		if sc.IsRemote() {
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
		} else if conn, ok := McpConns[n]; ok {
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
