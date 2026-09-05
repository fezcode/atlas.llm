package tui

import (
	"context"
	"fmt"
	"log"
	"sort"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	mcpx "atlas.llm/internal/mcp"
	"atlas.llm/internal/ui"
)

// handleMCP implements the `/mcp` slash command. Connection work happens in
// a tea.Cmd so a slow handshake (or an OAuth browser round-trip) doesn't
// block the UI goroutine.
func (m *chatModel) handleMCP(args []string) tea.Cmd {
	if len(args) == 0 {
		m.pushSystem(mcpx.McpStatusLines())
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
		m.pushSystem(mcpx.McpCatalogText())
		return nil

	case "tools":
		snap := mcpx.McpToolSnapshot()
		if len(snap) == 0 {
			m.pushSystem("No MCP tools available. Use `/mcp connect` to connect configured servers.")
			return nil
		}
		var b strings.Builder
		b.WriteString("MCP tools:\n")
		for _, t := range snap {
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
		if mcpx.DisconnectMCPServer(args[1]) {
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
		removed, err := mcpx.DeleteMCPAuth(args[1])
		if err != nil {
			m.pushError("clear credentials: " + err.Error())
			return nil
		}
		mcpx.DisconnectMCPServer(args[1])
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
	m.mcpPickerItems = append([]mcpx.McpPreset(nil), mcpx.McpCatalog...)
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

	if p.NeedsInput() {
		m.textarea.SetValue(mcpx.PresetAddCommand(p))
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
func (m *chatModel) addMCPServer(name string, sc mcpx.MCPServerConfig, label string) tea.Cmd {
	if err := sc.Validate(); err != nil {
		m.pushError(fmt.Sprintf("%s: %v", name, err))
		return nil
	}
	if err := mcpx.UpsertMCPServer(name, sc); err != nil {
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
	title := ui.BrandStyle.Render("Add an MCP server") +
		ui.SysStyle.Render("  (↑/↓ move · enter select · esc cancel)")
	rowSelected := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#0B1220")).
		Background(ui.ColAccent).
		Bold(true).
		Padding(0, 1)
	rowNormal := lipgloss.NewStyle().Padding(0, 1)

	configured := map[string]bool{}
	for _, n := range mcpx.ConfiguredMCPNames() {
		configured[n] = true
	}

	lines := []string{title, ""}
	for i, p := range m.mcpPickerItems {
		marker := "  "
		if configured[p.Key] {
			marker = ui.BrandStyle.Render("● ")
		}
		kind := "stdio"
		if p.Cfg.IsRemote() {
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
	lines = append(lines, "", ui.SysStyle.Render("● = already in mcp.json · not listed? /mcp add NAME -- npx -y pkg"))
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
	name, sc, custom, err := mcpx.ParseCustomAdd(args)
	if err != nil {
		m.pushError(err.Error())
		return nil
	}
	if custom {
		return m.addMCPServer(name, sc, "")
	}

	p, ok := mcpx.FindMCPPreset(args[0])
	if !ok {
		m.pushError(fmt.Sprintf("no built-in server named %q. `/mcp catalog` lists them, or:\n"+
			"  /mcp add %s -- npx -y some-mcp-package\n"+
			"  /mcp add %s --url=https://host/mcp --oauth", args[0], args[0], args[0]))
		return nil
	}
	built, err := mcpx.BuildPresetConfig(p, args[1:])
	if err != nil {
		m.textarea.SetValue(mcpx.PresetAddCommand(p))
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
	cfg, err := mcpx.LoadMCPConfig()
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
	if err := mcpx.UpsertMCPServer(name, sc); err != nil {
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
	cfg, err := mcpx.LoadMCPConfig()
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
	kv, err := mcpx.ParseKeyValues(args[1:])
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
	if err := mcpx.UpsertMCPServer(name, sc); err != nil {
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
	mcpx.DisconnectMCPServer(name)
	removed, err := mcpx.RemoveMCPServer(name)
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
	if _, err := mcpx.DeleteMCPAuth(name); err != nil {
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
		ctx, cancel := context.WithTimeout(context.Background(), mcpx.McpAuthTimeout)
		defer cancel()

		if name != "" {
			cfg, err := mcpx.LoadMCPConfig()
			if err != nil {
				return sysMsg{content: "mcp: " + err.Error()}
			}
			sc, ok := cfg.Servers[name]
			if !ok {
				return sysMsg{content: fmt.Sprintf("mcp: no server named %q in mcp.json", name)}
			}
			n, err := mcpx.ConnectMCPServer(ctx, name, sc)
			if err != nil {
				return sysMsg{content: fmt.Sprintf("mcp: %s failed — %v", name, err)}
			}
			return sysMsg{content: fmt.Sprintf("mcp: %s connected (%d tools).", name, n)}
		}

		results, err := mcpx.ConnectAllMCP(ctx)
		if err != nil {
			return sysMsg{content: "mcp: " + err.Error()}
		}
		return sysMsg{content: formatMCPResults(results)}
	}
}

// formatMCPResults renders the outcome of a batch connect.
func formatMCPResults(results []mcpx.McpConnectResult) string {
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
	p, _ := mcpx.McpConfigPath()
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

// autoConnectMCP connects configured servers in the background as the TUI
// starts, and reports the outcome into the viewport once it's up.
//
// Servers needing a fresh OAuth authorization are skipped: launching a
// browser unprompted at startup is hostile. Those connect on `/mcp connect`,
// or automatically here once credentials have been stored by an earlier run.
func autoConnectMCP() {
	cfg, err := mcpx.LoadMCPConfig()
	if err != nil {
		log.Printf("mcp: %v", err)
		return
	}
	var eligible, deferred []string
	for name, sc := range cfg.Servers {
		if sc.Disabled {
			continue
		}
		if sc.OAuth && !mcpx.HasStoredMCPAuth(name) {
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
		var results []mcpx.McpConnectResult
		for _, name := range eligible {
			ctx, cancel := context.WithTimeout(context.Background(), mcpx.McpConnectTimeout)
			n, err := mcpx.ConnectMCPServer(ctx, name, cfg.Servers[name])
			cancel()
			results = append(results, mcpx.McpConnectResult{Name: name, Tools: n, Err: err})
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
