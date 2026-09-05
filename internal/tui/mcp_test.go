package tui

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"golang.org/x/oauth2"

	"atlas.llm/internal/config"
	mcpx "atlas.llm/internal/mcp"
	"atlas.llm/internal/tools"
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
		if got := mcpx.McpToolName(tt.server, tt.tool); got != tt.want {
			t.Errorf("mcpToolName(%q, %q) = %q, want %q", tt.server, tt.tool, got, tt.want)
		}
	}
}

func TestMCPToolNameRespectsLengthLimit(t *testing.T) {
	got := mcpx.McpToolName(strings.Repeat("s", 60), strings.Repeat("t", 60))
	if len(got) != 64 {
		t.Errorf("name length = %d, want 64", len(got))
	}
}

func TestMCPServerConfigValidate(t *testing.T) {
	tests := []struct {
		name    string
		cfg     mcpx.MCPServerConfig
		wantErr bool
	}{
		{"stdio", mcpx.MCPServerConfig{Command: "npx"}, false},
		{"remote", mcpx.MCPServerConfig{URL: "https://example.com/mcp"}, false},
		{"remote oauth", mcpx.MCPServerConfig{URL: "https://example.com/mcp", OAuth: true}, false},
		{"neither", mcpx.MCPServerConfig{}, true},
		{"both", mcpx.MCPServerConfig{Command: "npx", URL: "https://example.com/mcp"}, true},
		{"oauth on stdio", mcpx.MCPServerConfig{Command: "npx", OAuth: true}, true},
		{"bad transport", mcpx.MCPServerConfig{URL: "https://x/mcp", Transport: "carrier-pigeon"}, true},
		{"sse transport", mcpx.MCPServerConfig{URL: "https://x/sse", Transport: "sse"}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.cfg.Validate()
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

	cfg, err := mcpx.LoadMCPConfig()
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
	if !conf.IsRemote() || !conf.OAuth || !conf.Trust {
		t.Errorf("confluence config parsed wrong: %+v", conf)
	}
	if !cfg.Servers["old"].Disabled {
		t.Error("disabled flag not parsed")
	}
}

func TestLoadMCPConfigMissingFileIsNotAnError(t *testing.T) {
	withTempHome(t)
	cfg, err := mcpx.LoadMCPConfig()
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
	got := mcpx.FlattenMCPContent(res)
	for _, want := range []string{"first", "second", "image/png"} {
		if !strings.Contains(got, want) {
			t.Errorf("flattened output %q missing %q", got, want)
		}
	}
}

func TestFlattenMCPContentFallsBackToStructured(t *testing.T) {
	res := &mcp.CallToolResult{StructuredContent: map[string]any{"count": 3}}
	if got := mcpx.FlattenMCPContent(res); !strings.Contains(got, "count") {
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
	mcpx.SaveMCPAuth("confluence", oc, tok)

	got, err := mcpx.LoadMCPAuth("confluence")
	if err != nil {
		t.Fatalf("loadMCPAuth: %v", err)
	}
	if got == nil {
		t.Fatal("no stored auth returned")
	}
	if got.Token.AccessToken != "access-abc" || got.Token.RefreshToken != "refresh-xyz" {
		t.Errorf("token round-trip lost data: %+v", got.Token)
	}
	if rebuilt := got.OauthConfig(); rebuilt.Endpoint.TokenURL != oc.Endpoint.TokenURL ||
		rebuilt.ClientID != oc.ClientID {
		t.Errorf("endpoint config round-trip lost data: %+v", rebuilt)
	}
	if !mcpx.HasStoredMCPAuth("confluence") {
		t.Error("hasStoredMCPAuth should report stored credentials")
	}

	// Credentials must not be group/world readable.
	p, _ := mcpx.McpAuthPath()
	info, err := os.Stat(p)
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS == "windows" {
		// Go's Chmod on Windows only toggles the read-only attribute, so a
		// writable file always reads back 0666 no matter what mode it was
		// created with — the 0600 in writeAuthStore is a no-op here and
		// asserting it would be asserting a fiction. What actually protects
		// the file is NTFS inheritance from the profile directory, so check
		// it landed there.
		home, err := os.UserHomeDir()
		if err != nil {
			t.Skip("no home directory to check containment against")
		}
		if !strings.HasPrefix(p, filepath.Clean(home)+string(filepath.Separator)) {
			t.Errorf("credential file %q is outside the user profile %q, so it "+
				"inherits no user-restricted ACL", p, home)
		}
	} else if perm := info.Mode().Perm(); perm&0077 != 0 {
		t.Errorf("credential file mode is %o, want no group/other access", perm)
	}

	// A refresh must update the token but keep the endpoint config.
	mcpx.UpdateMCPToken("confluence", &oauth2.Token{AccessToken: "access-2", RefreshToken: "refresh-xyz"})
	got, _ = mcpx.LoadMCPAuth("confluence")
	if got.Token.AccessToken != "access-2" {
		t.Errorf("token not updated, got %q", got.Token.AccessToken)
	}
	if got.TokenURL != oc.Endpoint.TokenURL {
		t.Error("endpoint config lost on token update")
	}

	removed, err := mcpx.DeleteMCPAuth("confluence")
	if err != nil || !removed {
		t.Fatalf("deleteMCPAuth = %v, %v", removed, err)
	}
	if mcpx.HasStoredMCPAuth("confluence") {
		t.Error("credentials still present after logout")
	}
}

func TestHasStoredMCPAuthRejectsUnusableToken(t *testing.T) {
	withTempHome(t)
	// Expired with no refresh token: re-authorizing is the only option.
	mcpx.SaveMCPAuth("dead", &oauth2.Config{Endpoint: oauth2.Endpoint{TokenURL: "https://x/token"}},
		&oauth2.Token{AccessToken: "old", Expiry: time.Now().Add(-time.Hour)})
	if mcpx.HasStoredMCPAuth("dead") {
		t.Error("expired token with no refresh token should not count as stored auth")
	}
}

// Built-in tools must keep resolving once MCP tools are registered.
func TestLookupToolPrefersBuiltins(t *testing.T) {
	if _, ok := mcpx.LookupTool("read_file"); !ok {
		t.Error("built-in read_file not found")
	}
	if _, ok := mcpx.LookupTool("definitely__not_a_tool"); ok {
		t.Error("unknown tool resolved")
	}
}

// --- catalog / config writing ---

// Every preset must produce a config that actually validates, or /mcp add
// would write an entry that can never connect.
func TestCatalogPresetsAreValid(t *testing.T) {
	seen := map[string]bool{}
	for _, p := range mcpx.McpCatalog {
		if p.Key == "" || p.Label == "" || p.Description == "" {
			t.Errorf("preset %+v is missing a key, label, or description", p)
		}
		if seen[p.Key] {
			t.Errorf("duplicate preset key %q", p.Key)
		}
		seen[p.Key] = true

		// Fill in whatever the preset demands, then it must validate.
		var args []string
		for _, f := range p.RequiredEnv {
			args = append(args, f.Key+"=value")
		}
		if p.ArgHint != "" {
			args = append(args, "/tmp")
		}
		sc, err := mcpx.BuildPresetConfig(p, args)
		if err != nil {
			t.Errorf("preset %q could not be built with its own placeholders: %v", p.Key, err)
			continue
		}
		if err := sc.Validate(); err != nil {
			t.Errorf("preset %q produces an invalid config: %v", p.Key, err)
		}
	}
}

func TestBuildPresetConfigReportsMissingInput(t *testing.T) {
	slack, ok := mcpx.FindMCPPreset("slack")
	if !ok {
		t.Fatal("slack preset missing from catalog")
	}
	if _, err := mcpx.BuildPresetConfig(slack, nil); err == nil {
		t.Error("expected an error when required env is absent")
	}
	sc, err := mcpx.BuildPresetConfig(slack, []string{"SLACK_BOT_TOKEN=xoxb-1", "SLACK_TEAM_ID=T1"})
	if err != nil {
		t.Fatalf("buildPresetConfig: %v", err)
	}
	if sc.Env["SLACK_BOT_TOKEN"] != "xoxb-1" || sc.Env["SLACK_TEAM_ID"] != "T1" {
		t.Errorf("env not applied: %+v", sc.Env)
	}
}

// Building a preset twice must not accumulate args onto the catalog entry.
func TestBuildPresetConfigDoesNotMutateCatalog(t *testing.T) {
	fs, ok := mcpx.FindMCPPreset("filesystem")
	if !ok {
		t.Fatal("filesystem preset missing")
	}
	before := len(fs.Cfg.Args)
	for i := 0; i < 3; i++ {
		sc, err := mcpx.BuildPresetConfig(fs, []string{"/tmp"})
		if err != nil {
			t.Fatalf("buildPresetConfig: %v", err)
		}
		if len(sc.Args) != before+1 {
			t.Fatalf("iteration %d: got %d args, want %d", i, len(sc.Args), before+1)
		}
	}
	if got, _ := mcpx.FindMCPPreset("filesystem"); len(got.Cfg.Args) != before {
		t.Errorf("catalog entry mutated: %d args, want %d", len(got.Cfg.Args), before)
	}
}

func TestParseCustomAdd(t *testing.T) {
	// stdio form
	name, sc, ok, err := mcpx.ParseCustomAdd([]string{"mine", "--", "npx", "-y", "pkg"})
	if err != nil || !ok {
		t.Fatalf("stdio form: ok=%v err=%v", ok, err)
	}
	if name != "mine" || sc.Command != "npx" || len(sc.Args) != 2 {
		t.Errorf("stdio parsed wrong: name=%q %+v", name, sc)
	}

	// remote form with flags
	name, sc, ok, err = mcpx.ParseCustomAdd([]string{"remote", "--url=https://h/mcp", "--oauth", "--trust"})
	if err != nil || !ok {
		t.Fatalf("remote form: ok=%v err=%v", ok, err)
	}
	if sc.URL != "https://h/mcp" || !sc.OAuth || !sc.Trust {
		t.Errorf("remote parsed wrong: %+v", sc)
	}

	// A bare preset name is not a custom definition.
	if _, _, ok, _ = mcpx.ParseCustomAdd([]string{"slack"}); ok {
		t.Error("bare name should not parse as a custom add")
	}
	// Unknown flags should be rejected rather than silently dropped.
	if _, _, _, err = mcpx.ParseCustomAdd([]string{"x", "--url=https://h", "--bogus"}); err == nil {
		t.Error("expected unknown flag to error")
	}
	// `--` with nothing after it is a user mistake worth reporting.
	if _, _, _, err = mcpx.ParseCustomAdd([]string{"x", "--"}); err == nil {
		t.Error("expected empty command after -- to error")
	}
}

// /mcp add must create mcp.json from nothing and preserve existing entries.
func TestUpsertAndRemoveMCPServer(t *testing.T) {
	withTempHome(t)

	if err := mcpx.UpsertMCPServer("slack", mcpx.MCPServerConfig{
		Command: "npx", Args: []string{"-y", "pkg"},
		Env: map[string]string{"SLACK_BOT_TOKEN": "xoxb-1"},
	}); err != nil {
		t.Fatalf("upsert into a nonexistent mcp.json: %v", err)
	}
	if err := mcpx.UpsertMCPServer("confluence", mcpx.MCPServerConfig{
		URL: "https://mcp.atlassian.com/v1/mcp", OAuth: true, Trust: true,
	}); err != nil {
		t.Fatalf("second upsert: %v", err)
	}

	cfg, err := mcpx.LoadMCPConfig()
	if err != nil {
		t.Fatalf("loadMCPConfig: %v", err)
	}
	if len(cfg.Servers) != 2 {
		t.Fatalf("got %d servers, want 2", len(cfg.Servers))
	}
	if cfg.Servers["slack"].Env["SLACK_BOT_TOKEN"] != "xoxb-1" {
		t.Error("first entry lost when the second was added")
	}

	// Updating one entry must leave the other alone.
	sc := cfg.Servers["slack"]
	sc.Trust = true
	if err := mcpx.UpsertMCPServer("slack", sc); err != nil {
		t.Fatal(err)
	}
	cfg, _ = mcpx.LoadMCPConfig()
	if !cfg.Servers["slack"].Trust || !cfg.Servers["confluence"].Trust {
		t.Error("trust update clobbered a neighbouring entry")
	}

	removed, err := mcpx.RemoveMCPServer("slack")
	if err != nil || !removed {
		t.Fatalf("removeMCPServer = %v, %v", removed, err)
	}
	cfg, _ = mcpx.LoadMCPConfig()
	if _, ok := cfg.Servers["slack"]; ok {
		t.Error("server still present after remove")
	}
	if _, ok := cfg.Servers["confluence"]; !ok {
		t.Error("remove dropped the wrong entry")
	}
	if removed, _ := mcpx.RemoveMCPServer("slack"); removed {
		t.Error("removing a missing server should report false")
	}
}

// A config written by /mcp add must be readable by loadMCPConfig — i.e. the
// round-trip through the same JSON tags actually works.
func TestSavedConfigRoundTrips(t *testing.T) {
	withTempHome(t)
	for _, p := range mcpx.McpCatalog {
		var args []string
		for _, f := range p.RequiredEnv {
			args = append(args, f.Key+"=secret")
		}
		if p.ArgHint != "" {
			args = append(args, "/tmp")
		}
		sc, err := mcpx.BuildPresetConfig(p, args)
		if err != nil {
			t.Fatalf("%s: %v", p.Key, err)
		}
		if err := mcpx.UpsertMCPServer(p.Key, sc); err != nil {
			t.Fatalf("%s: %v", p.Key, err)
		}
	}
	cfg, err := mcpx.LoadMCPConfig()
	if err != nil {
		t.Fatalf("loadMCPConfig: %v", err)
	}
	if len(cfg.Servers) != len(mcpx.McpCatalog) {
		t.Fatalf("round-tripped %d servers, want %d", len(cfg.Servers), len(mcpx.McpCatalog))
	}
	for _, p := range mcpx.McpCatalog {
		got := cfg.Servers[p.Key]
		if err := got.Validate(); err != nil {
			t.Errorf("%s failed to validate after round-trip: %v", p.Key, err)
		}
	}
}

func TestParseKeyValues(t *testing.T) {
	got, err := mcpx.ParseKeyValues([]string{"A=1", "B=x=y"})
	if err != nil {
		t.Fatalf("parseKeyValues: %v", err)
	}
	if got["A"] != "1" || got["B"] != "x=y" {
		t.Errorf("parsed wrong: %+v", got)
	}
	if _, err := mcpx.ParseKeyValues([]string{"nope"}); err == nil {
		t.Error("expected an error for a value without '='")
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

	results, err := mcpx.ConnectAllMCP(ctx)
	if err != nil {
		t.Fatalf("connectAllMCP: %v", err)
	}
	defer mcpx.ShutdownMCP()
	for _, r := range results {
		if r.Err != nil {
			t.Fatalf("server %s failed: %v", r.Name, r.Err)
		}
		if r.Tools == 0 {
			t.Fatalf("server %s exposed no tools", r.Name)
		}
	}

	echo, ok := mcpx.LookupTool("everything__echo")
	if !ok {
		t.Fatalf("everything__echo not registered; have %v", toolNamesOf(mcpx.McpToolSnapshot()))
	}
	if !echo.Destructive {
		t.Error("untrusted server's tools must require confirmation")
	}
	if tr, ok := mcpx.LookupTool("trusted__echo"); !ok || tr.Destructive {
		t.Errorf(`"trust": true should skip confirmation (found=%v)`, ok)
	}

	// MCP tools must be advertised alongside the built-ins, each with a
	// name and a parameters schema or llama-server rejects the request.
	defs := tools.ToolDefsJSON()
	if len(defs) <= len(tools.ToolRegistry) {
		t.Errorf("got %d tool defs, want more than the %d built-ins", len(defs), len(tools.ToolRegistry))
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

	if !mcpx.DisconnectMCPServer("everything") {
		t.Error("disconnect reported nothing to disconnect")
	}
	if _, ok := mcpx.LookupTool("everything__echo"); ok {
		t.Error("tool still registered after disconnect")
	}
	if _, ok := mcpx.LookupTool("trusted__echo"); !ok {
		t.Error("disconnecting one server dropped another server's tools")
	}
}

// TestLiveAddPresetAndConnect walks the path a user actually takes: pick a
// preset, have it written to mcp.json, then connect — no hand-editing.
func TestLiveAddPresetAndConnect(t *testing.T) {
	if testing.Short() {
		t.Skip("live MCP server test skipped in -short mode")
	}
	if _, err := exec.LookPath("npx"); err != nil {
		t.Skip("npx not available")
	}
	withTempHome(t)

	p, ok := mcpx.FindMCPPreset("everything")
	if !ok {
		t.Fatal("everything preset missing from catalog")
	}
	sc, err := mcpx.BuildPresetConfig(p, nil)
	if err != nil {
		t.Fatalf("buildPresetConfig: %v", err)
	}
	if err := mcpx.UpsertMCPServer(p.Key, sc); err != nil {
		t.Fatalf("upsertMCPServer: %v", err)
	}

	// mcp.json must now exist and be loadable, having been created for us.
	cfgPath, _ := mcpx.McpConfigPath()
	if _, err := os.Stat(cfgPath); err != nil {
		t.Fatalf("mcp.json was not created: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	results, err := mcpx.ConnectAllMCP(ctx)
	if err != nil {
		t.Fatalf("connectAllMCP: %v", err)
	}
	defer mcpx.ShutdownMCP()
	if len(results) != 1 || results[0].Err != nil || results[0].Tools == 0 {
		t.Fatalf("preset did not connect: %+v", results)
	}

	// Untrusted by default; /mcp trust flips it and a reconnect applies it.
	tool, ok := mcpx.LookupTool("everything__echo")
	if !ok {
		t.Fatalf("tool not registered; have %v", toolNamesOf(mcpx.McpToolSnapshot()))
	}
	if !tool.Destructive {
		t.Error("a newly added server should confirm every call")
	}

	sc.Trust = true
	if err := mcpx.UpsertMCPServer(p.Key, sc); err != nil {
		t.Fatal(err)
	}
	if _, err := mcpx.ConnectMCPServer(ctx, p.Key, sc); err != nil {
		t.Fatalf("reconnect after trust change: %v", err)
	}
	if tool, _ := mcpx.LookupTool("everything__echo"); tool.Destructive {
		t.Error("trust change did not take effect after reconnect")
	}
}

// TestLiveRemoteMCPServer covers the remote streamable-HTTP path, which the
// npx-based tests never touch. Uses the no-auth DeepWiki test server.
//
// A connect failure skips rather than fails: this reaches a third-party
// service, so an outage or a sandboxed network shouldn't redden the suite.
// Everything after the connection is asserted normally.
func TestLiveRemoteMCPServer(t *testing.T) {
	if testing.Short() {
		t.Skip("live MCP server test skipped in -short mode")
	}
	withTempHome(t)

	p, ok := mcpx.FindMCPPreset("deepwiki")
	if !ok {
		t.Fatal("deepwiki preset missing from catalog")
	}
	if p.Cfg.IsRemote() == false {
		t.Fatal("deepwiki preset should be a remote server")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	n, err := mcpx.ConnectMCPServer(ctx, p.Key, p.Cfg)
	if err != nil {
		t.Skipf("remote test server unreachable (%v)", err)
	}
	defer mcpx.ShutdownMCP()
	if n == 0 {
		t.Fatal("remote server exposed no tools")
	}

	tool, ok := mcpx.LookupTool("deepwiki__ask_question")
	if !ok {
		t.Fatalf("expected deepwiki__ask_question; have %v", toolNamesOf(mcpx.McpToolSnapshot()))
	}
	// No "trust" in the preset, so it must route through the confirm modal.
	if !tool.Destructive {
		t.Error("remote server without trust should require confirmation")
	}
	if tool.Parameters == nil {
		t.Error("remote tool has no parameters schema")
	}

	out, err := tool.Run(map[string]any{
		"repoName": "modelcontextprotocol/go-sdk",
		"question": "What transports does the client support?",
	})
	if err != nil {
		t.Fatalf("remote tool call: %v", err)
	}
	if strings.TrimSpace(out) == "" {
		t.Error("remote tool returned empty output")
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
	dir, err := config.AtlasDir()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "mcp.json"), []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
}

func toolNamesOf(tools []tools.Tool) []string {
	out := make([]string, 0, len(tools))
	for _, t := range tools {
		out = append(out, t.Name)
	}
	return out
}

// Atlassian publishes no protected-resource metadata, and the spec fallback
// fails RFC 8414's issuer check (the document at mcp.atlassian.com declares
// cf.mcp.atlassian.com). Naming the auth server explicitly has to carry
// discovery all the way to the authorization URL.
//
// Browser consent can't be completed here, so reaching that URL is success —
// it's precisely the step that used to fail with "failed to get
// authorization server metadata".
func TestLiveAtlassianOAuthDiscovery(t *testing.T) {
	if testing.Short() {
		t.Skip("live OAuth discovery skipped in -short mode")
	}
	withTempHome(t)
	p, ok := mcpx.FindMCPPreset("atlassian")
	if !ok {
		t.Fatal("atlassian preset missing")
	}
	if p.Cfg.AuthServer == "" {
		t.Fatal("atlassian preset must name its authorization server")
	}

	authURL := make(chan string, 1)
	orig := mcpx.OpenBrowserHook
	mcpx.OpenBrowserHook = func(u string) error { authURL <- u; return nil }
	defer func() { mcpx.OpenBrowserHook = orig }()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	connErr := make(chan error, 1)
	go func() { _, err := mcpx.ConnectMCPServer(ctx, "atlassian", p.Cfg); connErr <- err }()

	select {
	case u := <-authURL:
		if !strings.Contains(u, "atlassian.com") {
			t.Errorf("authorization URL points somewhere unexpected: %s", u)
		}
		// Dynamic client registration and PKCE both have to have happened
		// for these to be present.
		for _, want := range []string{"client_id=", "code_challenge=", "response_type=code"} {
			if !strings.Contains(u, want) {
				t.Errorf("authorization URL missing %q: %s", want, u)
			}
		}
	case err := <-connErr:
		t.Skipf("could not reach Atlassian (%v)", err)
	case <-ctx.Done():
		t.Skip("timed out reaching Atlassian")
	}
}

// A server with no protected-resource metadata must still discover its auth
// server when the config names one.
func TestSynthesizedProtectedResourceMetadata(t *testing.T) {
	c := mcpx.WithDeclaredAuthServer(&http.Client{}, "https://example.test/v1/mcp", "https://auth.example.test/")
	tr, ok := c.Transport.(*mcpx.PrmTransport)
	if !ok {
		t.Fatal("transport was not wrapped")
	}
	if tr.Host != "example.test" {
		t.Errorf("host = %q, want example.test", tr.Host)
	}
	// A trailing slash on the auth server would produce a mismatched issuer.
	if tr.AuthServer != "https://auth.example.test" {
		t.Errorf("authServer = %q, trailing slash not trimmed", tr.AuthServer)
	}
}

// Datadog, unlike Atlassian, publishes correct protected-resource metadata
// and an authorization server whose issuer matches. It should therefore need
// no auth_server override — a regression there would show up as discovery
// failing before the browser step.
func TestLiveDatadogOAuthDiscovery(t *testing.T) {
	if testing.Short() {
		t.Skip("live OAuth discovery skipped in -short mode")
	}
	withTempHome(t)
	p, ok := mcpx.FindMCPPreset("datadog")
	if !ok {
		t.Fatal("datadog preset missing")
	}
	if p.Cfg.AuthServer != "" {
		t.Errorf("datadog should not need an auth_server override, got %q", p.Cfg.AuthServer)
	}

	authURL := make(chan string, 1)
	orig := mcpx.OpenBrowserHook
	mcpx.OpenBrowserHook = func(u string) error { authURL <- u; return nil }
	defer func() { mcpx.OpenBrowserHook = orig }()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	errc := make(chan error, 1)
	go func() { _, err := mcpx.ConnectMCPServer(ctx, "datadog", p.Cfg); errc <- err }()

	select {
	case u := <-authURL:
		for _, want := range []string{"datadoghq.com", "client_id=", "code_challenge=", "response_type=code"} {
			if !strings.Contains(u, want) {
				t.Errorf("authorization URL missing %q: %s", want, u)
			}
		}
	case err := <-errc:
		t.Skipf("could not reach Datadog (%v)", err)
	case <-ctx.Done():
		t.Skip("timed out reaching Datadog")
	}
}
