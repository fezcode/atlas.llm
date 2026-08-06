package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

// mcpEnvField is a secret a stdio preset needs before it can start.
type mcpEnvField struct {
	Key  string
	Hint string
}

// mcpPreset is a ready-made server definition so users don't have to know a
// package name, an endpoint URL, or the mcp.json schema. Presets are
// starting points: everything they write can be adjusted afterwards with
// /mcp env, /mcp trust, or by editing mcp.json directly.
type mcpPreset struct {
	Key         string
	Label       string
	Description string
	Cfg         MCPServerConfig
	// RequiredEnv must be supplied as KEY=VALUE pairs when adding.
	RequiredEnv []mcpEnvField
	// ArgHint, when non-empty, is a positional argument the server needs
	// appended to its command line (e.g. a directory to expose).
	ArgHint string
}

// needsInput reports whether the preset can't be added without more from
// the user.
func (p mcpPreset) needsInput() bool { return len(p.RequiredEnv) > 0 || p.ArgHint != "" }

// mcpCatalog is the built-in preset list. Remote entries were verified to
// be live OAuth-protected MCP endpoints; stdio entries name published npm
// packages run through `npx -y`.
var mcpCatalog = []mcpPreset{
	{
		Key:         "atlassian",
		Label:       "Atlassian (Confluence + Jira)",
		Description: "Search and edit Confluence pages and Jira issues. Opens a browser to authorize.",
		Cfg: MCPServerConfig{
			URL:   "https://mcp.atlassian.com/v1/mcp",
			OAuth: true,
			// Atlassian publishes no protected-resource metadata, and the
			// spec fallback fails its issuer check. Name the real
			// authorization server so discovery can proceed.
			AuthServer: "https://cf.mcp.atlassian.com",
		},
	},
	{
		Key:         "slack",
		Label:       "Slack (bot token)",
		Description: "Read channels and post messages. Needs a Slack app + bot token, usually admin-approved.",
		Cfg: MCPServerConfig{
			Command: "npx",
			Args:    []string{"-y", "@modelcontextprotocol/server-slack"},
		},
		RequiredEnv: []mcpEnvField{
			{Key: "SLACK_BOT_TOKEN", Hint: "xoxb-..."},
			{Key: "SLACK_TEAM_ID", Hint: "T01234567"},
		},
	},
	{
		Key:   "slack-user",
		Label: "Slack (user token, no app setup)",
		Description: "Same access you have in Slack, with no app to create and no admin approval. " +
			"Uses a user OAuth token.",
		Cfg: MCPServerConfig{
			Command: "npx",
			Args:    []string{"-y", "slack-mcp-server@latest", "--transport", "stdio"},
		},
		RequiredEnv: []mcpEnvField{
			{Key: "SLACK_MCP_XOXP_TOKEN", Hint: "xoxp-..."},
		},
	},
	{
		Key:         "github",
		Label:       "GitHub",
		Description: "Repos, issues, and pull requests. Needs a personal access token.",
		Cfg: MCPServerConfig{
			Command: "npx",
			Args:    []string{"-y", "@modelcontextprotocol/server-github"},
		},
		RequiredEnv: []mcpEnvField{
			{Key: "GITHUB_PERSONAL_ACCESS_TOKEN", Hint: "ghp_..."},
		},
	},
	{
		Key:         "linear",
		Label:       "Linear",
		Description: "Issues, projects, and cycles. Opens a browser to authorize.",
		Cfg: MCPServerConfig{
			URL:   "https://mcp.linear.app/mcp",
			OAuth: true,
		},
	},
	{
		Key:   "datadog",
		Label: "Datadog",
		Description: "Metrics, logs, monitors, traces, and incidents. " +
			"Opens a browser to authorize — no API keys to create.",
		Cfg: MCPServerConfig{
			URL:   "https://mcp.datadoghq.com/api/unstable/mcp-server/mcp",
			OAuth: true,
		},
	},
	{
		Key:         "sentry",
		Label:       "Sentry",
		Description: "Errors, issues, and releases. Opens a browser to authorize.",
		Cfg: MCPServerConfig{
			URL:   "https://mcp.sentry.dev/mcp",
			OAuth: true,
		},
	},
	{
		Key:         "filesystem",
		Label:       "Filesystem",
		Description: "Read and write files under a directory you nominate.",
		Cfg: MCPServerConfig{
			Command: "npx",
			Args:    []string{"-y", "@modelcontextprotocol/server-filesystem"},
		},
		ArgHint: "/path/to/expose",
	},
	{
		Key:         "memory",
		Label:       "Memory",
		Description: "A persistent knowledge graph the model can write notes into.",
		Cfg: MCPServerConfig{
			Command: "npx",
			Args:    []string{"-y", "@modelcontextprotocol/server-memory"},
		},
	},
	{
		Key:         "everything",
		Label:       "Everything (local test server)",
		Description: "Reference server exposing sample tools. Checks the stdio path works.",
		Cfg: MCPServerConfig{
			Command: "npx",
			Args:    []string{"-y", "@modelcontextprotocol/server-everything"},
		},
	},
	{
		Key:         "context7",
		Label:       "Context7 (no setup)",
		Description: "Up-to-date docs and code examples for thousands of libraries. Remote, no auth.",
		Cfg:         MCPServerConfig{URL: "https://mcp.context7.com/mcp"},
	},
	{
		Key:         "gitmcp",
		Label:       "GitMCP (no setup)",
		Description: "Fetch and search documentation and code for any public GitHub repo. Remote, no auth.",
		Cfg:         MCPServerConfig{URL: "https://gitmcp.io/docs"},
	},
	{
		Key:   "duckduckgo",
		Label: "DuckDuckGo search (no setup)",
		Description: "Web search, news, and page fetching. No API key. " +
			"Ask for a region (e.g. us-en) — it defaults to a China locale.",
		Cfg: MCPServerConfig{
			Command: "npx",
			Args:    []string{"-y", "mcp-duckduckgo"},
		},
	},
	{
		Key:         "sequential-thinking",
		Label:       "Sequential Thinking (no setup)",
		Description: "Lets the model break a hard problem into revisable steps. Local, no auth.",
		Cfg: MCPServerConfig{
			Command: "npx",
			Args:    []string{"-y", "@modelcontextprotocol/server-sequential-thinking"},
		},
	},
	{
		Key:   "deepwiki",
		Label: "DeepWiki (remote test server)",
		Description: "Ask questions about any public GitHub repo. No auth — " +
			"the quickest way to check the remote path works.",
		Cfg: MCPServerConfig{URL: "https://mcp.deepwiki.com/mcp"},
	},
}

func findMCPPreset(key string) (mcpPreset, bool) {
	for _, p := range mcpCatalog {
		if strings.EqualFold(p.Key, key) {
			return p, true
		}
	}
	return mcpPreset{}, false
}

// saveMCPConfig writes mcp.json, creating it if absent. Marshalled indented
// so the file stays hand-editable for anything the commands don't cover.
func saveMCPConfig(cfg MCPConfig) error {
	p, err := mcpConfigPath()
	if err != nil {
		return err
	}
	if cfg.Servers == nil {
		cfg.Servers = map[string]MCPServerConfig{}
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(p, append(data, '\n'), 0644)
}

// upsertMCPServer adds or replaces one server entry, preserving the rest of
// the file.
func upsertMCPServer(name string, sc MCPServerConfig) error {
	cfg, err := loadMCPConfig()
	if err != nil {
		return err
	}
	cfg.Servers[name] = sc
	return saveMCPConfig(cfg)
}

// removeMCPServer deletes an entry. Reports whether it existed.
func removeMCPServer(name string) (bool, error) {
	cfg, err := loadMCPConfig()
	if err != nil {
		return false, err
	}
	if _, ok := cfg.Servers[name]; !ok {
		return false, nil
	}
	delete(cfg.Servers, name)
	return true, saveMCPConfig(cfg)
}

// parseKeyValues turns ["A=1", "B=2"] into a map. Values may contain '='.
func parseKeyValues(args []string) (map[string]string, error) {
	out := map[string]string{}
	for _, a := range args {
		k, v, ok := strings.Cut(a, "=")
		if !ok || k == "" {
			return nil, fmt.Errorf("expected KEY=VALUE, got %q", a)
		}
		out[k] = v
	}
	return out, nil
}

// buildPresetConfig fills a preset's template from user-supplied KEY=VALUE
// pairs and positional arguments, reporting anything still missing.
func buildPresetConfig(p mcpPreset, args []string) (MCPServerConfig, error) {
	sc := p.Cfg
	// Copy the slice so repeated adds don't append onto the catalog entry.
	sc.Args = append([]string(nil), p.Cfg.Args...)

	var positionals []string
	var kvArgs []string
	for _, a := range args {
		if strings.Contains(a, "=") && !strings.HasPrefix(a, "-") {
			kvArgs = append(kvArgs, a)
			continue
		}
		positionals = append(positionals, a)
	}

	env, err := parseKeyValues(kvArgs)
	if err != nil {
		return sc, err
	}

	var missing []string
	for _, f := range p.RequiredEnv {
		v, ok := env[f.Key]
		if !ok || v == "" {
			missing = append(missing, f.Key+"="+f.Hint)
			continue
		}
		if sc.Env == nil {
			sc.Env = map[string]string{}
		}
		sc.Env[f.Key] = v
	}
	if p.ArgHint != "" {
		if len(positionals) == 0 {
			missing = append(missing, p.ArgHint)
		} else {
			sc.Args = append(sc.Args, positionals...)
		}
	}
	if len(missing) > 0 {
		return sc, fmt.Errorf("missing: %s", strings.Join(missing, " "))
	}
	return sc, nil
}

// presetAddCommand renders the full command a user should run to add a
// preset, with placeholders in place of the values they need to supply.
func presetAddCommand(p mcpPreset) string {
	parts := []string{"/mcp add " + p.Key}
	for _, f := range p.RequiredEnv {
		parts = append(parts, f.Key+"="+f.Hint)
	}
	if p.ArgHint != "" {
		parts = append(parts, p.ArgHint)
	}
	return strings.Join(parts, " ")
}

// parseCustomAdd handles the escape hatches for servers not in the catalog:
//
//	/mcp add NAME --url=https://host/mcp [--oauth] [--sse] [--trust]
//	/mcp add NAME -- npx -y some-package [ARGS...]
//
// Returns ok=false when args don't look like a custom definition, so the
// caller can fall back to treating the name as a preset key.
func parseCustomAdd(args []string) (name string, sc MCPServerConfig, ok bool, err error) {
	if len(args) == 0 {
		return "", sc, false, nil
	}
	name = args[0]
	rest := args[1:]

	// stdio form: everything after `--` is the command line.
	for i, a := range rest {
		if a == "--" {
			cmdline := rest[i+1:]
			if len(cmdline) == 0 {
				return name, sc, true, fmt.Errorf("nothing after `--` — expected a command to run")
			}
			sc.Command = cmdline[0]
			sc.Args = append([]string(nil), cmdline[1:]...)
			if err := applyAddFlags(&sc, rest[:i]); err != nil {
				return name, sc, true, err
			}
			return name, sc, true, nil
		}
	}

	// remote form: needs an explicit --url.
	hasURL := false
	for _, a := range rest {
		if strings.HasPrefix(a, "--url=") {
			hasURL = true
			break
		}
	}
	if !hasURL {
		return name, sc, false, nil
	}
	if err := applyAddFlags(&sc, rest); err != nil {
		return name, sc, true, err
	}
	return name, sc, true, nil
}

func applyAddFlags(sc *MCPServerConfig, flags []string) error {
	for _, f := range flags {
		switch {
		case strings.HasPrefix(f, "--url="):
			sc.URL = strings.TrimPrefix(f, "--url=")
		case f == "--oauth":
			sc.OAuth = true
		case f == "--sse":
			sc.Transport = "sse"
		case f == "--trust":
			sc.Trust = true
		default:
			return fmt.Errorf("unknown flag %q (expected --url=, --oauth, --sse, --trust)", f)
		}
	}
	return nil
}

// mcpCatalogText renders `/mcp catalog`.
func mcpCatalogText() string {
	var b strings.Builder
	b.WriteString("Built-in MCP servers — add with `/mcp add <name>`:\n\n")
	for _, p := range mcpCatalog {
		kind := "stdio"
		if p.Cfg.isRemote() {
			kind = "remote"
			if p.Cfg.OAuth {
				kind = "remote, oauth"
			}
		}
		fmt.Fprintf(&b, "  %-12s %-15s %s\n", p.Key, "("+kind+")", p.Description)
		if p.needsInput() {
			fmt.Fprintf(&b, "  %-12s %-15s → %s\n", "", "", presetAddCommand(p))
		}
	}
	b.WriteString("\nNot listed here? Add any server directly:\n")
	b.WriteString("  /mcp add NAME -- npx -y some-mcp-package\n")
	b.WriteString("  /mcp add NAME --url=https://host/mcp --oauth\n")
	return b.String()
}

// configuredMCPNames returns the server names in mcp.json, sorted.
func configuredMCPNames() []string {
	cfg, err := loadMCPConfig()
	if err != nil {
		return nil
	}
	names := make([]string, 0, len(cfg.Servers))
	for n := range cfg.Servers {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}
