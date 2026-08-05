package main

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"golang.org/x/oauth2"
)

func TestMCPToolNameNamespacesAndSanitizes(t *testing.T) {
	tests := []struct {
		server, tool, want string
	}{
		{"slack", "post_message", "slack__post_message"},
		{"confluence", "search", "confluence__search"},
		// Dots and spaces are not legal in a tool name the model can emit.
		{"my.server", "get page", "my_server__get_page"},
		{"a/b", "c:d", "a_b__c_d"},
	}
	for _, tt := range tests {
		if got := mcpToolName(tt.server, tt.tool); got != tt.want {
			t.Errorf("mcpToolName(%q, %q) = %q, want %q", tt.server, tt.tool, got, tt.want)
		}
	}
}

func TestMCPToolNameRespectsLengthLimit(t *testing.T) {
	got := mcpToolName(strings.Repeat("s", 60), strings.Repeat("t", 60))
	if len(got) != 64 {
		t.Errorf("name length = %d, want 64", len(got))
	}
}

func TestMCPServerConfigValidate(t *testing.T) {
	tests := []struct {
		name    string
		cfg     MCPServerConfig
		wantErr bool
	}{
		{"stdio", MCPServerConfig{Command: "npx"}, false},
		{"remote", MCPServerConfig{URL: "https://example.com/mcp"}, false},
		{"remote oauth", MCPServerConfig{URL: "https://example.com/mcp", OAuth: true}, false},
		{"neither", MCPServerConfig{}, true},
		{"both", MCPServerConfig{Command: "npx", URL: "https://example.com/mcp"}, true},
		{"oauth on stdio", MCPServerConfig{Command: "npx", OAuth: true}, true},
		{"bad transport", MCPServerConfig{URL: "https://x/mcp", Transport: "carrier-pigeon"}, true},
		{"sse transport", MCPServerConfig{URL: "https://x/sse", Transport: "sse"}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.cfg.validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// The Claude Desktop config shape should load as-is so users can paste an
// existing mcp.json in.
func TestLoadMCPConfigParsesStandardShape(t *testing.T) {
	withTempHome(t)
	writeMCPConfig(t, `{
	  "mcpServers": {
	    "slack": {
	      "command": "npx",
	      "args": ["-y", "@modelcontextprotocol/server-slack"],
	      "env": {"SLACK_BOT_TOKEN": "xoxb-test"}
	    },
	    "confluence": {
	      "url": "https://mcp.atlassian.com/v1/mcp",
	      "oauth": true,
	      "trust": true
	    },
	    "old": {"command": "foo", "disabled": true}
	  }
	}`)

	cfg, err := loadMCPConfig()
	if err != nil {
		t.Fatalf("loadMCPConfig: %v", err)
	}
	if len(cfg.Servers) != 3 {
		t.Fatalf("got %d servers, want 3", len(cfg.Servers))
	}
	slack := cfg.Servers["slack"]
	if slack.Command != "npx" || len(slack.Args) != 2 || slack.Env["SLACK_BOT_TOKEN"] != "xoxb-test" {
		t.Errorf("slack config parsed wrong: %+v", slack)
	}
	if slack.Trust {
		t.Error("trust should default to false (confirm every call)")
	}
	conf := cfg.Servers["confluence"]
	if !conf.isRemote() || !conf.OAuth || !conf.Trust {
		t.Errorf("confluence config parsed wrong: %+v", conf)
	}
	if !cfg.Servers["old"].Disabled {
		t.Error("disabled flag not parsed")
	}
}

func TestLoadMCPConfigMissingFileIsNotAnError(t *testing.T) {
	withTempHome(t)
	cfg, err := loadMCPConfig()
	if err != nil {
		t.Fatalf("expected no error for missing mcp.json, got %v", err)
	}
	if len(cfg.Servers) != 0 {
		t.Errorf("expected no servers, got %d", len(cfg.Servers))
	}
}

func TestFlattenMCPContent(t *testing.T) {
	res := &mcp.CallToolResult{Content: []mcp.Content{
		&mcp.TextContent{Text: "first"},
		&mcp.TextContent{Text: "second"},
		&mcp.ImageContent{MIMEType: "image/png", Data: []byte{1, 2, 3}},
	}}
	got := flattenMCPContent(res)
	for _, want := range []string{"first", "second", "image/png"} {
		if !strings.Contains(got, want) {
			t.Errorf("flattened output %q missing %q", got, want)
		}
	}
}

func TestFlattenMCPContentFallsBackToStructured(t *testing.T) {
	res := &mcp.CallToolResult{StructuredContent: map[string]any{"count": 3}}
	if got := flattenMCPContent(res); !strings.Contains(got, "count") {
		t.Errorf("expected structured content in output, got %q", got)
	}
}

// A stored token plus its endpoint config must survive a round-trip, or an
// OAuth server would re-prompt on every launch.
func TestMCPAuthStoreRoundTrip(t *testing.T) {
	withTempHome(t)

	oc := &oauth2.Config{
		ClientID:    "client-123",
		RedirectURL: "http://127.0.0.1:5000/callback",
		Scopes:      []string{"read:page", "offline_access"},
		Endpoint: oauth2.Endpoint{
			AuthURL:  "https://auth.example.com/authorize",
			TokenURL: "https://auth.example.com/token",
		},
	}
	tok := &oauth2.Token{
		AccessToken:  "access-abc",
		RefreshToken: "refresh-xyz",
		Expiry:       time.Now().Add(time.Hour),
	}
	saveMCPAuth("confluence", oc, tok)

	got, err := loadMCPAuth("confluence")
	if err != nil {
		t.Fatalf("loadMCPAuth: %v", err)
	}
	if got == nil {
		t.Fatal("no stored auth returned")
	}
	if got.Token.AccessToken != "access-abc" || got.Token.RefreshToken != "refresh-xyz" {
		t.Errorf("token round-trip lost data: %+v", got.Token)
	}
	if rebuilt := got.oauthConfig(); rebuilt.Endpoint.TokenURL != oc.Endpoint.TokenURL ||
		rebuilt.ClientID != oc.ClientID {
		t.Errorf("endpoint config round-trip lost data: %+v", rebuilt)
	}
	if !hasStoredMCPAuth("confluence") {
		t.Error("hasStoredMCPAuth should report stored credentials")
	}

	// Credentials must not be group/world readable.
	p, _ := mcpAuthPath()
	info, err := os.Stat(p)
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm&0077 != 0 {
		t.Errorf("credential file mode is %o, want no group/other access", perm)
	}

	// A refresh must update the token but keep the endpoint config.
	updateMCPToken("confluence", &oauth2.Token{AccessToken: "access-2", RefreshToken: "refresh-xyz"})
	got, _ = loadMCPAuth("confluence")
	if got.Token.AccessToken != "access-2" {
		t.Errorf("token not updated, got %q", got.Token.AccessToken)
	}
	if got.TokenURL != oc.Endpoint.TokenURL {
		t.Error("endpoint config lost on token update")
	}

	removed, err := deleteMCPAuth("confluence")
	if err != nil || !removed {
		t.Fatalf("deleteMCPAuth = %v, %v", removed, err)
	}
	if hasStoredMCPAuth("confluence") {
		t.Error("credentials still present after logout")
	}
}

func TestHasStoredMCPAuthRejectsUnusableToken(t *testing.T) {
	withTempHome(t)
	// Expired with no refresh token: re-authorizing is the only option.
	saveMCPAuth("dead", &oauth2.Config{Endpoint: oauth2.Endpoint{TokenURL: "https://x/token"}},
		&oauth2.Token{AccessToken: "old", Expiry: time.Now().Add(-time.Hour)})
	if hasStoredMCPAuth("dead") {
		t.Error("expired token with no refresh token should not count as stored auth")
	}
}

// Built-in tools must keep resolving once MCP tools are registered.
func TestLookupToolPrefersBuiltins(t *testing.T) {
	if _, ok := lookupTool("read_file"); !ok {
		t.Error("built-in read_file not found")
	}
	if _, ok := lookupTool("definitely__not_a_tool"); ok {
		t.Error("unknown tool resolved")
	}
}

// --- live integration ---

// TestLiveStdioMCPServer exercises the real protocol path against
// @modelcontextprotocol/server-everything. Skipped unless npx is present and
// -short is off, since it downloads a package on first run.
func TestLiveStdioMCPServer(t *testing.T) {
	if testing.Short() {
		t.Skip("live MCP server test skipped in -short mode")
	}
	if _, err := exec.LookPath("npx"); err != nil {
		t.Skip("npx not available")
	}
	withTempHome(t)
	writeMCPConfig(t, `{
	  "mcpServers": {
	    "everything": {
	      "command": "npx",
	      "args": ["-y", "@modelcontextprotocol/server-everything"]
	    },
	    "trusted": {
	      "command": "npx",
	      "args": ["-y", "@modelcontextprotocol/server-everything"],
	      "trust": true
	    }
	  }
	}`)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	results, err := connectAllMCP(ctx)
	if err != nil {
		t.Fatalf("connectAllMCP: %v", err)
	}
	defer shutdownMCP()
	for _, r := range results {
		if r.Err != nil {
			t.Fatalf("server %s failed: %v", r.Name, r.Err)
		}
		if r.Tools == 0 {
			t.Fatalf("server %s exposed no tools", r.Name)
		}
	}

	echo, ok := lookupTool("everything__echo")
	if !ok {
		t.Fatalf("everything__echo not registered; have %v", toolNamesOf(mcpToolSnapshot()))
	}
	if !echo.Destructive {
		t.Error("untrusted server's tools must require confirmation")
	}
	if tr, ok := lookupTool("trusted__echo"); !ok || tr.Destructive {
		t.Errorf(`"trust": true should skip confirmation (found=%v)`, ok)
	}

	// MCP tools must be advertised alongside the built-ins, each with a
	// name and a parameters schema or llama-server rejects the request.
	defs := toolDefsJSON()
	if len(defs) <= len(toolRegistry) {
		t.Errorf("got %d tool defs, want more than the %d built-ins", len(defs), len(toolRegistry))
	}
	for _, d := range defs {
		fn := d["function"].(map[string]any)
		if n, _ := fn["name"].(string); n == "" {
			t.Errorf("tool def with empty name: %v", d)
		}
		if fn["parameters"] == nil {
			t.Errorf("tool def %v has nil parameters", fn["name"])
		}
	}

	out, err := echo.Run(map[string]any{"message": "hello from atlas"})
	if err != nil {
		t.Fatalf("echo.Run: %v", err)
	}
	if !strings.Contains(out, "hello from atlas") {
		t.Errorf("echo returned %q, want it to contain the input", out)
	}

	if !disconnectMCPServer("everything") {
		t.Error("disconnect reported nothing to disconnect")
	}
	if _, ok := lookupTool("everything__echo"); ok {
		t.Error("tool still registered after disconnect")
	}
	if _, ok := lookupTool("trusted__echo"); !ok {
		t.Error("disconnecting one server dropped another server's tools")
	}
}

// --- helpers ---

// withTempHome points atlasDir() at a throwaway directory so tests never
// touch the developer's real ~/.atlas.
func withTempHome(t *testing.T) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home) // os.UserHomeDir on Windows
	return home
}

func writeMCPConfig(t *testing.T, body string) {
	t.Helper()
	// Validate here so a typo in a test fixture fails loudly.
	var probe map[string]any
	if err := json.Unmarshal([]byte(body), &probe); err != nil {
		t.Fatalf("test fixture is not valid JSON: %v", err)
	}
	dir, err := atlasDir()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "mcp.json"), []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
}

func toolNamesOf(tools []Tool) []string {
	out := make([]string, 0, len(tools))
	for _, t := range tools {
		out = append(out, t.Name)
	}
	return out
}
