package tui

import (
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"atlas.llm/internal/catalog"
	"atlas.llm/internal/config"
	"atlas.llm/internal/engine"
)

// Slash-command dispatch and tab completion.

func (m *chatModel) handleSlash(input string) tea.Cmd {
	parts := strings.Fields(input)
	cmd := strings.ToLower(parts[0])
	args := parts[1:]

	switch cmd {
	case "/help":
		m.handleHelp(args)
		return nil

	case "/quit", "/exit":
		return tea.Quit

	case "/clear":
		m.rendered = nil
		m.pushSystem("Screen cleared. (Conversation context still active — use /reset to drop it.)")
		return nil

	case "/reset":
		m.history = nil
		m.agentMsgs = nil
		m.pendingCalls = nil
		m.stepCount = 0
		m.rendered = nil
		engine.ResetUsage()
		if s, err := engine.EnsureServer(); err == nil {
			_ = s.DropKVCache()
		}
		m.pushSystem("Conversation reset. Context and KV cache cleared.")
		return nil

	case "/tools":
		m.handleTools(args)
		return nil

	case "/mcp":
		return m.handleMCP(args)

	case "/compact":
		return m.handleCompact()

	case "/config":
		return m.handleConfig(args)

	case "/yesman":
		m.handleYesman(args)
		return nil

	case "/ama":
		m.handleAMA(args)
		return nil

	case "/list":
		var b strings.Builder
		b.WriteString("Available models:\n")
		engineStatus := "engine: not downloaded"
		if engine.IsEngineDownloaded() {
			engineStatus = "engine: downloaded"
		}
		b.WriteString("  " + engineStatus + "\n\n")
		for _, mm := range catalog.AvailableModels {
			status := "not downloaded"
			if config.IsModelDownloaded(mm) {
				status = "downloaded"
			}
			marker := " "
			if mm.Name == m.modelName {
				marker = "*"
			}
			fmt.Fprintf(&b, " %s %-25s %-8s %-16s %s\n",
				marker, mm.Name, mm.Size, status, engine.ModelResourceNote(mm))
		}
		if total, ok := engine.SystemRAM(); ok {
			fmt.Fprintf(&b, "\n(* = current model · RAM share is the weights against %s total;\n"+
				" the context window adds more on top — see /help set ctx_size)", engine.FormatBytes(total))
		} else {
			b.WriteString("\n(* = current model)")
		}
		m.pushSystem(b.String())
		return nil

	case "/model":
		if len(args) == 0 {
			m.openModelPicker()
			return nil
		}
		name := args[0]
		target, ok := config.FindModel(name)
		if !ok {
			m.pushError(fmt.Sprintf("unknown model: %s (try /list)", name))
			return nil
		}
		m.applyModelSelection(target)
		return nil

	case "/download":
		targets, err := resolveDownloadTargets(args)
		if err != nil {
			m.pushError(err.Error())
			return nil
		}
		var summary []string
		if targets.engine && !engine.IsEngineDownloaded() {
			summary = append(summary, "engine")
		}
		for _, mm := range targets.models {
			if !config.IsModelDownloaded(mm) {
				summary = append(summary, mm.Name)
			}
		}
		if len(summary) == 0 {
			m.pushSystem("Nothing to download — already present.")
			return nil
		}
		m.busy = true
		m.busyReason = "downloading"
		m.busyStart = time.Now()
		m.dlName = summary[0]
		m.dlWritten = 0
		m.dlTotal = 0
		_ = m.progress.SetPercent(0)
		m.pushSystem(fmt.Sprintf("Downloading: %s", strings.Join(summary, ", ")))
		return tea.Batch(runDownloadAllCmd(targets), m.spinner.Tick)

	case "/summarize":
		opts, err := parseSummarizeArgs(args)
		if err != nil {
			m.pushError(err.Error())
			return nil
		}
		m.busy = true
		m.busyReason = "summarizing"
		m.busyStart = time.Now()
		m.pushSystem(fmt.Sprintf("Summarizing %s → %s  (max-size=%d, exclude=%v)",
			opts.TargetDir, opts.Output, opts.MaxSize, opts.Exclude))
		return tea.Batch(runSummarizeCmd(opts), m.spinner.Tick)

	case "/set":
		m.handleSet(args)
		return nil

	case "/grep":
		if len(args) == 0 {
			m.pushError("usage: /grep <query>")
			return nil
		}
		query := strings.TrimSpace(strings.TrimPrefix(input, parts[0]))
		m.busy = true
		m.busyReason = "grepping"
		m.busyStart = time.Now()
		m.pushSystem(fmt.Sprintf("Searching current directory for: %s", query))
		return tea.Batch(runGrepCmd(".", query), m.spinner.Tick)

	default:
		m.pushError(fmt.Sprintf("unknown command: %s", cmd))
		return nil
	}
}

// slashCommands is the canonical list of completable command names.
var slashCommands = []string{
	"/ama", "/clear", "/download", "/exit", "/grep", "/help", "/list",
	"/compact", "/config", "/mcp", "/model", "/quit", "/reset", "/set",
	"/summarize", "/tools", "/yesman",
}

// tabComplete handles Tab in the input box. Returns true if it modified
// the textarea (or printed a hint), so the caller can swallow the key.
func (m *chatModel) tabComplete() bool {
	val := m.textarea.Value()
	if !strings.HasPrefix(val, "/") || strings.Contains(val, "\n") {
		return false
	}
	// First-word completion — no space yet typed.
	if !strings.Contains(val, " ") {
		return m.completeToken("", strings.ToLower(val), slashCommands)
	}
	// Sub-arg completion — only when we're still on the first arg.
	parts := strings.SplitN(val, " ", 2)
	head := strings.ToLower(parts[0])
	arg := parts[1]
	if strings.Contains(arg, " ") {
		// Two-level completions. `/help mcp tr<Tab>` finishes the subcommand;
		// `/config load fa<Tab>` finishes a saved profile name.
		sub := strings.SplitN(arg, " ", 2)
		if len(sub) == 2 && !strings.Contains(sub[1], " ") {
			switch head {
			case "/help":
				return m.completeToken(head+" "+sub[0]+" ", sub[1], helpSubNames(sub[0]))
			case "/config":
				switch strings.ToLower(sub[0]) {
				case "load", "use", "show", "view", "cat", "delete", "rm", "remove":
					names, _ := config.ListProfiles()
					return m.completeToken(head+" "+sub[0]+" ", sub[1], names)
				}
			}
		}
		return false
	}
	var pool []string
	switch head {
	case "/model":
		for _, mm := range catalog.AvailableModels {
			pool = append(pool, mm.Name)
		}
	case "/help":
		pool = helpTopicNames()
	case "/set":
		pool = settingKeys()
	case "/tools":
		pool = []string{"on", "off", "list"}
	case "/yesman", "/ama":
		pool = []string{"on", "off"}
	case "/config":
		pool = []string{"save", "load", "show", "list", "delete"}
	case "/mcp":
		pool = []string{"add", "catalog", "connect", "disconnect", "env",
			"help", "logout", "remove", "tools", "trust"}
	case "/download":
		pool = []string{"all", "engine"}
		for _, mm := range catalog.AvailableModels {
			pool = append(pool, mm.Name)
		}
	default:
		return false
	}
	return m.completeToken(head+" ", arg, pool)
}

// completeToken extends `prefix` against `pool`. With one match it
// substitutes; with several it extends to the longest common prefix and
// lists candidates as a system message.
func (m *chatModel) completeToken(head, prefix string, pool []string) bool {
	var matches []string
	for _, s := range pool {
		if strings.HasPrefix(s, prefix) {
			matches = append(matches, s)
		}
	}
	if len(matches) == 0 {
		return false
	}
	if len(matches) == 1 {
		suffix := ""
		// Trailing space only for top-level commands that take args.
		if head == "" && commandTakesArgs(matches[0]) {
			suffix = " "
		}
		m.textarea.SetValue(head + matches[0] + suffix)
		m.textarea.CursorEnd()
		return true
	}
	common := longestCommonPrefix(matches)
	if len(common) > len(prefix) {
		m.textarea.SetValue(head + common)
		m.textarea.CursorEnd()
	}
	m.pushSystem("Completions: " + strings.Join(matches, "  "))
	return true
}

func commandTakesArgs(cmd string) bool {
	switch cmd {
	case "/model", "/download", "/summarize", "/grep", "/set", "/tools", "/config":
		return true
	}
	return false
}

func longestCommonPrefix(ss []string) string {
	if len(ss) == 0 {
		return ""
	}
	p := ss[0]
	for _, s := range ss[1:] {
		n := 0
		for n < len(p) && n < len(s) && p[n] == s[n] {
			n++
		}
		p = p[:n]
		if p == "" {
			break
		}
	}
	return p
}
