package main

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// The agentic loop as the TUI sees it: dispatching tool calls, rendering
// their traces and results, and the confirm prompt that gates the
// destructive ones.

func runChatCmd(ctx context.Context, history []ChatMessage, input string) tea.Cmd {
	return func() tea.Msg {
		reply, err := chat(ctx, history, input)
		if err != nil {
			return inferenceErrMsg{err: err}
		}
		return assistantReplyMsg{content: reply}
	}
}

// runAgentStepCmd posts the current agent message list to llama-server with
// the tool definitions attached, and wraps the response (content + any
// tool_calls) into an agentStepMsg for the update loop.
func runAgentStepCmd(ctx context.Context, msgs []ChatMsg) tea.Cmd {
	snapshot := append([]ChatMsg(nil), msgs...)
	return func() tea.Msg {
		cfg, _ := loadConfig()
		content, calls, err := runAgentStep(ctx, snapshot, cfg.MaxTokens)
		return agentStepMsg{content: content, toolCalls: calls, err: err}
	}
}

// runToolCmd executes a single approved tool call off the UI goroutine so
// slow tools (run_cmd, large reads) don't freeze the TUI.
func runToolCmd(call ToolCall) tea.Cmd {
	return func() tea.Msg {
		t, ok := lookupTool(call.Function.Name)
		if !ok {
			return toolRanMsg{call: call, err: fmt.Errorf("unknown tool: %s", call.Function.Name)}
		}
		var args map[string]any
		if call.Function.Arguments != "" {
			if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
				return toolRanMsg{call: call, err: fmt.Errorf("parse arguments: %w", err)}
			}
		}
		result, err := t.Run(args)
		return toolRanMsg{call: call, result: result, err: err}
	}
}

// handleTools implements `/tools`, `/tools on|off`, and `/tools list`.
func (m *chatModel) handleTools(args []string) {
	cfg, err := loadConfig()
	if err != nil {
		m.pushError("load config: " + err.Error())
		return
	}
	if len(args) == 0 {
		state := "off"
		if cfg.ToolsEnabled {
			state = "on"
		}
		m.pushSystem(fmt.Sprintf("tools = %s  (use `/tools on` to enable agentic tool-use, `/tools list` to see tools)", state))
		return
	}
	switch strings.ToLower(args[0]) {
	case "on":
		cfg.ToolsEnabled = true
		if err := saveConfig(cfg); err != nil {
			m.pushError("save config: " + err.Error())
			return
		}
		m.agentEnabled = true
		m.pushSystem("Agentic tools enabled. Destructive actions (write/edit/run_cmd/browser_open) will prompt for confirmation. Smaller models (e.g. Gemma 3 1B) may not reliably emit tool calls.")
	case "off":
		cfg.ToolsEnabled = false
		if err := saveConfig(cfg); err != nil {
			m.pushError("save config: " + err.Error())
			return
		}
		m.agentEnabled = false
		m.agentMsgs = nil
		m.pendingCalls = nil
		m.pushSystem("Agentic tools disabled.")
	case "list":
		var b strings.Builder
		b.WriteString("Available tools:\n")
		for _, name := range toolNames() {
			t := toolRegistry[name]
			tag := "safe"
			if t.Destructive {
				tag = "needs confirm"
			}
			fmt.Fprintf(&b, "  %-12s [%s]  %s\n", t.Name, tag, t.Description)
		}
		m.pushSystem(b.String())
	default:
		m.pushError(fmt.Sprintf("unknown /tools arg: %s (expected on|off|list)", args[0]))
	}
}

// handleAgentStep processes one assistant response from the model: records
// it, renders any textual content, and either dispatches the next tool call
// or ends the turn when no more calls are pending.
func (m *chatModel) handleAgentStep(msg agentStepMsg) tea.Cmd {
	if m.inflightCanceled(msg.err) {
		return nil
	}
	if m.canceled {
		// Result of a turn the user already stopped.
		m.canceled = false
		return nil
	}
	if msg.err != nil {
		m.busy = false
		m.busyReason = ""
		m.pendingCalls = nil
		m.dropUnansweredUser()
		m.pushError(msg.err.Error())
		return nil
	}
	m.stepCount++
	// Record the assistant turn verbatim so the next call replays the
	// tool_calls the model emitted — llama-server needs that to match up
	// with the `role: tool` replies we're about to send.
	m.agentMsgs = append(m.agentMsgs, ChatMsg{
		Role:      "assistant",
		Content:   msg.content,
		ToolCalls: msg.toolCalls,
	})
	if strings.TrimSpace(msg.content) != "" {
		m.pushAssistant(msg.content)
	}
	if len(msg.toolCalls) == 0 {
		m.busy = false
		m.busyReason = ""
		m.maybeSuggestCompact()
		return nil
	}
	cfgRounds, _ := loadConfig()
	limit := resolveMaxToolRounds(cfgRounds)
	if limit != unlimitedToolRounds && m.stepCount >= limit {
		m.pushError(fmt.Sprintf(
			"Agent stopped after %d tool-call rounds — the current limit for one message.\n\n"+
				"This usually means the model kept calling tools without reaching an "+
				"answer, often retrying something that failed. The trace above shows "+
				"what it tried.\n\n"+
				"Worth checking: is the model big enough for tool use (qwen3.5-4b and "+
				"up), and were the paths it used correct? Paths are relative to %s.\n\n"+
				"Raise the limit with `/set max_tool_rounds N`, or `off` to remove it.",
			m.stepCount, displayRoot()))
		m.busy = false
		m.busyReason = ""
		m.pendingCalls = nil
		return nil
	}
	m.pendingCalls = append(m.pendingCalls, msg.toolCalls...)
	m.busyReason = "using tools"
	return m.dispatchNextTool()
}

// dispatchNextTool pops the head of the pending-call queue and either runs
// it immediately (safe tool), opens the confirm modal (destructive tool),
// or fabricates an error result (unknown tool) and recurses.
func (m *chatModel) dispatchNextTool() tea.Cmd {
	if len(m.pendingCalls) == 0 {
		return runAgentStepCmd(m.newInflight(), m.agentMsgs)
	}
	call := m.pendingCalls[0]
	m.pendingCalls = m.pendingCalls[1:]
	// A model that keeps issuing the identical call is stuck; feeding the
	// same result back again would just spend the remaining rounds.
	if m.noteRepeatedCall(call) {
		m.renderToolTrace(call, "(repeated — stopping)", true)
		m.appendToolResult(call, fmt.Sprintf(
			"This exact call has already been made %d times with the same result. "+
				"Do not repeat it. Either try a different approach or answer with "+
				"what you already know.", maxIdenticalCalls))
		m.pendingCalls = nil
		m.busy = false
		m.busyReason = ""
		m.pushError(fmt.Sprintf(
			"Agent stopped: it called %s with identical arguments %d times in a row "+
				"without making progress.", call.Function.Name, maxIdenticalCalls+1))
		return nil
	}
	t, ok := lookupTool(call.Function.Name)
	if !ok {
		m.renderToolTrace(call, "(unknown tool)", true)
		m.appendToolResult(call, fmt.Sprintf("unknown tool: %s", call.Function.Name))
		return m.dispatchNextTool()
	}
	// ask_user is answered by a person, not run: hand it to the AMA picker.
	if call.Function.Name == askUserToolName {
		return m.startAMA(call)
	}
	if t.Destructive && !m.yesman {
		m.confirmCall = &call
		m.confirmIdx = 0
		m.picking = "tool_confirm"
		m.renderConfirm()
		return nil
	}
	note := ""
	if t.Destructive {
		// Say so every time. A silent destructive call is exactly what the
		// confirm modal existed to prevent.
		note = "(auto-approved by /yesman)"
	}
	m.renderToolTrace(call, note, false)
	return runToolCmd(call)
}

// handleToolRan records a completed tool invocation's result and advances
// the queue: either runs the next pending call or, if the batch is empty,
// re-invokes the model with the updated message list.
func (m *chatModel) handleToolRan(msg toolRanMsg) tea.Cmd {
	if m.canceled {
		m.canceled = false
		return nil
	}
	result := msg.result
	if msg.err != nil {
		result = "Error: " + msg.err.Error()
	}
	m.renderToolResult(msg.call, result, msg.err != nil)
	m.appendToolResult(msg.call, result)
	return m.dispatchNextTool()
}

// appendToolResult pushes a `role: tool` turn into the agent message list.
// The tool_call_id links back to the assistant's request so llama-server
// can pair the reply with the originating call.
func (m *chatModel) appendToolResult(call ToolCall, content string) {
	m.agentMsgs = append(m.agentMsgs, ChatMsg{
		Role:       "tool",
		ToolCallID: call.ID,
		Name:       call.Function.Name,
		Content:    content,
	})
}

// renderToolTrace prints the tool-call invocation as a dim inline block so
// the user can see what the model is doing between visible assistant turns.
func (m *chatModel) renderToolTrace(call ToolCall, note string, bad bool) {
	var args map[string]any
	_ = json.Unmarshal([]byte(call.Function.Arguments), &args)
	argSummary := summarizeToolCallArgs(args)
	style := lipgloss.NewStyle().Foreground(colDim).Italic(true)
	if bad {
		style = lipgloss.NewStyle().Foreground(colErr).Italic(true)
	}
	line := fmt.Sprintf("↳ tool %s %s", call.Function.Name, argSummary)
	if note != "" {
		line += "  " + note
	}
	m.rendered = append(m.rendered, style.Render(line))
	m.refresh()
}

// renderToolResult previews the first couple of lines of a tool result
// inline so the conversation stays scannable without dumping the full
// content (the model still sees the full result).
func (m *chatModel) renderToolResult(_ ToolCall, result string, bad bool) {
	const maxLines = 3
	lines := strings.Split(result, "\n")
	preview := lines
	if len(preview) > maxLines {
		preview = preview[:maxLines]
	}
	for i, ln := range preview {
		if len(ln) > 200 {
			preview[i] = ln[:200] + "…"
		}
	}
	more := ""
	if len(lines) > maxLines {
		more = fmt.Sprintf("  (+%d more lines)", len(lines)-maxLines)
	}
	col := colDim
	if bad {
		col = colErr
	}
	style := lipgloss.NewStyle().Foreground(col)
	for _, ln := range preview {
		m.rendered = append(m.rendered, style.Render("   "+ln))
	}
	if more != "" {
		m.rendered = append(m.rendered, style.Render(more))
	}
	m.refresh()
}

// renderConfirm paints the confirm modal over the viewport. Mirrors the
// model-picker rendering conventions: title, body, two selectable rows.
func (m *chatModel) renderConfirm() {
	if m.confirmCall == nil {
		return
	}
	var args map[string]any
	_ = json.Unmarshal([]byte(m.confirmCall.Function.Arguments), &args)

	title := brandStyle.Render("Tool call requires approval") + sysStyle.Render("  (↑/↓ switch · enter confirm · esc deny)")
	sub := lipgloss.NewStyle().Foreground(colErr).Bold(true).Render("DESTRUCTIVE") +
		"  " + metaLabelStyle.Render(m.confirmCall.Function.Name)
	argStyle := lipgloss.NewStyle().Foreground(colMuted)

	lines := []string{title, "", sub, ""}
	// Argument preview — one line per key, content previewed.
	keys := make([]string, 0, len(args))
	for k := range args {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		val := fmt.Sprintf("%v", args[k])
		if len(val) > 120 {
			val = val[:120] + "…"
		}
		val = strings.ReplaceAll(val, "\n", " ⏎ ")
		lines = append(lines, "  "+argStyle.Render(k+": ")+val)
	}
	lines = append(lines, "")

	rowSelected := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#0B1220")).
		Background(colAccent).
		Bold(true).
		Padding(0, 1)
	rowNormal := lipgloss.NewStyle().Padding(0, 1)
	opts := []string{"▶ Approve and run", "  Deny"}
	for i, o := range opts {
		if i == m.confirmIdx {
			lines = append(lines, rowSelected.Render(o))
		} else {
			lines = append(lines, rowNormal.Render(o))
		}
	}
	m.viewport.SetContent(strings.Join(lines, "\n"))
}

// resolveConfirm closes the confirm modal. approved=true runs the tool;
// approved=false synthesizes a denial result and hands control back to the
// agent loop.
func (m *chatModel) resolveConfirm(approved bool) tea.Cmd {
	if m.confirmCall == nil {
		m.picking = ""
		return nil
	}
	call := *m.confirmCall
	m.confirmCall = nil
	m.picking = ""
	m.refresh()

	if !approved {
		m.renderToolTrace(call, "(denied by user)", true)
		m.appendToolResult(call, "User denied this tool call. Do not retry; continue without it.")
		return m.dispatchNextTool()
	}
	m.renderToolTrace(call, "(approved)", false)
	return runToolCmd(call)
}

// parseSummarizeArgs parses the token list that follows /summarize in the
// TUI. Supports one optional positional DIR and --flag=value options:
//
//	/summarize
//	/summarize ./src
//	/summarize --max-size=131072
//	/summarize ./src --exclude=.min.js,.lock
func parseSummarizeArgs(args []string) (SummarizeOptions, error) {
	opts := SummarizeOptions{
		TargetDir: ".",
		Output:    "SUMMARY.md",
		MaxSize:   DefaultSummarizeMaxSize,
	}
	for _, a := range args {
		switch {
		case strings.HasPrefix(a, "--max-size="):
			v := strings.TrimPrefix(a, "--max-size=")
			n, err := strconv.ParseInt(v, 10, 64)
			if err != nil || n <= 0 {
				return opts, fmt.Errorf("invalid --max-size=%q (expected positive integer bytes)", v)
			}
			opts.MaxSize = n
		case strings.HasPrefix(a, "--exclude="):
			v := strings.TrimPrefix(a, "--exclude=")
			if v != "" {
				opts.Exclude = strings.Split(v, ",")
			}
		case strings.HasPrefix(a, "--"):
			return opts, fmt.Errorf("unknown option: %s (supported: --max-size=N, --exclude=.ext1,.ext2)", a)
		default:
			opts.TargetDir = a
		}
	}
	return opts, nil
}

func runSummarizeCmd(opts SummarizeOptions) tea.Cmd {
	return func() tea.Msg {
		err := summarizeDirectory(opts, progressToSysMsg())
		return summarizeDoneMsg{path: opts.Output, err: err}
	}
}

func runGrepCmd(dir, query string) tea.Cmd {
	return func() tea.Msg {
		hits, err := grepDirectory(dir, query, DefaultGrepMaxSize, progressToSysMsg())
		return grepDoneMsg{query: query, hits: hits, err: err}
	}
}

// handleYesman implements `/yesman`, `/yesman on`, and `/yesman off`.
//
// The setting is intentionally session-scoped and never written to
// config.json. Persisting it would mean a future run silently auto-running
// run_cmd and file writes because of a toggle flipped days earlier.
func (m *chatModel) handleYesman(args []string) {
	want := !m.yesman
	if len(args) > 0 {
		switch strings.ToLower(args[0]) {
		case "on", "true", "yes":
			want = true
		case "off", "false", "no":
			want = false
		default:
			m.pushError(fmt.Sprintf("unknown /yesman arg: %s (expected on|off)", args[0]))
			return
		}
	}
	if want == m.yesman {
		state := "off"
		if m.yesman {
			state = "on"
		}
		m.pushSystem("yesman is already " + state + ".")
		return
	}
	m.yesman = want
	if !want {
		m.pushSystem("yesman off. Destructive tools will ask for confirmation again.")
		m.refresh()
		return
	}
	m.pushSystem(
		"⚠  yesman ON — destructive tools now run WITHOUT asking, for this session only.\n\n" +
			"That includes run_cmd (arbitrary shell commands), write_file, edit_file, " +
			"multi_edit, and any MCP tool from a server you haven't marked trusted — " +
			"so a model mistake can now delete files, push commits, or post to Slack " +
			"with no prompt.\n\n" +
			"It is never written to config.json, so it resets when you quit. " +
			"Turn it off with `/yesman off`, and press esc to stop a turn mid-flight.")
	m.refresh()
}

// maxIdenticalCalls is how many times the same tool call may repeat inside
// one turn before the loop is treated as stuck.
const maxIdenticalCalls = 3

// noteRepeatedCall records a call signature and reports whether the model
// has now repeated it too many times.
func (m *chatModel) noteRepeatedCall(call ToolCall) bool {
	if m.repeatedCalls == nil {
		m.repeatedCalls = map[string]int{}
	}
	sig := call.Function.Name + "(" + call.Function.Arguments + ")"
	m.repeatedCalls[sig]++
	return m.repeatedCalls[sig] > maxIdenticalCalls
}
