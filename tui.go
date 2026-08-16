package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/atotto/clipboard"
	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/glamour/styles"
	"github.com/charmbracelet/lipgloss"
	"github.com/muesli/termenv"
	"golang.org/x/term"
)

// program is set in startChat so download goroutines can Send progress messages.
var program *tea.Program

// Palette. Hex values degrade gracefully to the nearest 256-color on
// terminals without truecolor support.
var (
	colAccent    = lipgloss.Color("#A78BFA") // violet — brand accent
	colUser      = lipgloss.Color("#38BDF8") // sky — user messages
	colAssistant = lipgloss.Color("#34D399") // emerald — assistant messages
	colMuted     = lipgloss.Color("#9CA3AF") // gray — system/footer
	colDim       = lipgloss.Color("#4B5563") // slate — rules, separators
	colErr       = lipgloss.Color("#F87171") // red
	colBusy      = lipgloss.Color("#FBBF24") // amber
)

var (
	// Pill-style role badges — colored-on-dark backgrounds with one-char padding.
	userPillStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#0B1220")).
			Background(colUser).
			Bold(true).
			Padding(0, 1)
	assistantPillStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#0B1220")).
				Background(colAssistant).
				Bold(true).
				Padding(0, 1)
	errPillStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#0B1220")).
			Background(colErr).
			Bold(true).
			Padding(0, 1)

	sysStyle = lipgloss.NewStyle().Foreground(colMuted).Italic(true)
	// thinkStyle dims the reasoning text a show_thinking transcript carries,
	// so the eye separates it from the reply without a border.
	thinkStyle = lipgloss.NewStyle().Foreground(colMuted).Faint(true)
	errTextStyle = lipgloss.NewStyle().Foreground(colErr)

	// Top bar: accent-colored brand + muted meta, with a thin underline rule.
	brandStyle     = lipgloss.NewStyle().Foreground(colAccent).Bold(true)
	metaLabelStyle = lipgloss.NewStyle().Foreground(colDim)
	metaValueStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#E5E7EB"))
	sepStyle       = lipgloss.NewStyle().Foreground(colDim)
	busyStyle      = lipgloss.NewStyle().Foreground(colBusy).Bold(true)

	footerStyle    = lipgloss.NewStyle().Foreground(colMuted)
	footerKeyStyle = lipgloss.NewStyle().Foreground(colAccent).Bold(true)

	ruleStyle = lipgloss.NewStyle().Foreground(colDim)
)

type (
	assistantReplyMsg struct{ content string }
	inferenceErrMsg   struct{ err error }
	agentStepMsg      struct {
		content   string
		toolCalls []ToolCall
		err       error
	}
	toolRanMsg struct {
		call   ToolCall
		result string
		err    error
	}
	// assistantDeltaMsg carries one streamed increment. reasoning is
	// non-empty while the model is still thinking and has produced no
	// answer yet; the transcript shows it as text or as a byte counter
	// depending on show_thinking.
	assistantDeltaMsg struct {
		content   string
		reasoning string
	}
	sysMsg           struct{ content string }
	summarizeDoneMsg struct {
		path string
		err  error
	}
	serverReadyMsg struct {
		model string
		err   error
	}
	grepDoneMsg struct {
		query string
		hits  []GrepHit
		err   error
	}
	downloadDoneMsg struct {
		what string
		err  error
	}
	downloadProgressMsg struct {
		name           string
		written, total int64
	}
)

type chatModel struct {
	viewport   viewport.Model
	textarea   textarea.Model
	spinner    spinner.Model
	progress   progress.Model
	history    []ChatMessage
	rendered   []string
	width      int
	height     int
	busy       bool
	busyReason string
	busyStart  time.Time
	modelName  string
	cwd        string

	// Active download state (only set while busyReason == "downloading")
	dlName    string
	dlWritten int64
	dlTotal   int64
	dlMeter   downloadMeter

	// Model picker state. When picking != "", key events route to the
	// picker instead of the textarea; ↑/↓ move, Enter selects, Esc cancels.
	picking     string // "", "model", "mcp_add", or "tool_confirm"
	pickerIdx   int
	pickerItems []Model
	// mcpPickerItems backs the "mcp_add" picker. Kept separate from
	// pickerItems because that one is typed to the model registry.
	mcpPickerItems []mcpPreset

	// Input history, shell style. inputHistory holds submitted lines oldest
	// first; historyIdx == len(inputHistory) means "not browsing", and
	// historyDraft parks the half-typed line so it survives a round trip
	// through the history and back.
	inputHistory []string
	historyIdx   int
	historyDraft string

	// Markdown renderer for assistant replies. Rebuilt on resize so word
	// wrap tracks the viewport width.
	mdRenderer *glamour.TermRenderer
	mdWidth    int

	// Agent (tool-use) state. Persists across turns when cfg.ToolsEnabled.
	// agentMsgs is the full rolling message list handed to the model —
	// including system prompt, user turns, assistant tool_calls, and tool
	// results. pendingCalls holds tool calls from the latest assistant
	// step that haven't been executed yet. stepCount guards against loops
	// (see /set max_tool_rounds).
	agentEnabled bool
	agentMsgs    []ChatMsg
	pendingCalls []ToolCall
	stepCount    int
	confirmCall  *ToolCall
	confirmIdx   int // 0 = approve, 1 = deny

	// AMA (/ama) state. When the agent calls ask_user, the call is parked in
	// amaCall and rendered as an interactive picker (picking == "ama"); the
	// cursor rides on pickerIdx, and amaChecked tracks the toggles for a
	// checkbox question. Resolving feeds the choice back as the tool result.
	amaCall    *ToolCall
	amaSpec    amaSpec
	amaChecked []bool

	// repeatedCalls counts identical tool calls within the current turn, so
	// a model stuck retrying the same failing call is caught early instead
	// of silently burning the whole round budget.
	repeatedCalls map[string]int

	// compactSuggested debounces the "context is filling up" hint so it
	// fires once per crossing rather than after every turn.
	compactSuggested bool

	// yesman auto-approves destructive tool calls. Deliberately NOT part of
	// Config: it lives and dies with the session, so a forgotten toggle
	// can't silently arm the next run.
	yesman bool

	// streaming holds the reply assembled so far. streamIdx is where it
	// lives in m.rendered so each delta can rewrite that line in place
	// rather than appending. streamThink accumulates the reasoning text
	// seen before any answer, shown verbatim (show_thinking on, decided
	// once per stream in streamShowThink) or as a byte counter so a long
	// think shows progress instead of a frozen screen. streamThinkShown
	// marks that the think block has been sealed into its own transcript
	// line so the answer can stream below it.
	streaming        bool
	streamBuf        string
	streamIdx        int
	streamThink      string
	streamShowThink  bool
	streamThinkShown bool

	// cancelInflight aborts the running generation when Esc is pressed.
	// canceled marks that the abort was deliberate, so the resulting
	// context error is reported as "stopped" rather than as a failure.
	cancelInflight context.CancelFunc
	canceled       bool
}

func newChatModel() chatModel {
	ta := textarea.New()
	ta.Placeholder = "Ask anything, or type /help ..."
	ta.Focus()
	ta.Prompt = ""
	ta.CharLimit = 4000
	ta.SetHeight(2)
	ta.ShowLineNumbers = false
	ta.FocusedStyle.Placeholder = lipgloss.NewStyle().Foreground(colDim).Italic(true)
	ta.BlurredStyle.Placeholder = lipgloss.NewStyle().Foreground(colDim).Italic(true)
	ta.FocusedStyle.CursorLine = lipgloss.NewStyle()
	ta.BlurredStyle.CursorLine = lipgloss.NewStyle()

	vp := viewport.New(80, 20)
	vp.SetContent(welcomeText())
	// Drop the vim-style default bindings outright. Nothing forwards keys
	// here any more, but leaving them armed invites the bug back the moment
	// someone reinstates viewport.Update for key events.
	vp.KeyMap = viewport.KeyMap{}

	sp := spinner.New()
	sp.Spinner = spinner.Dot
	sp.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("205"))

	pr := progress.New(progress.WithDefaultGradient())
	pr.Width = 40

	cfg, _ := loadConfig()
	// Mirror the persisted /ama preference into the global the tool layer
	// reads, so ask_user is advertised from the first turn if it was left on.
	amaOn.Store(cfg.AMAEnabled)

	cm := chatModel{
		viewport:     vp,
		textarea:     ta,
		spinner:      sp,
		progress:     pr,
		modelName:    cfg.CurrentModel,
		cwd:          displayCwd(),
		agentEnabled: cfg.ToolsEnabled,
	}
	if m, ok := findModel(cfg.CurrentModel); ok &&
		isEngineDownloaded() && isModelDownloaded(m) {
		cm.busy = true
		cm.busyReason = "loading model"
		cm.busyStart = time.Now()
	}
	return cm
}

// displayCwd returns the working directory in a compact, user-friendly form:
// substitutes $HOME with ~ so the header stays short on deeply nested paths.
func displayCwd() string {
	wd, err := os.Getwd()
	if err != nil {
		return "?"
	}
	if home, err := os.UserHomeDir(); err == nil {
		if rel, err := filepath.Rel(home, wd); err == nil && !strings.HasPrefix(rel, "..") {
			if rel == "." {
				return "~"
			}
			return "~" + string(filepath.Separator) + rel
		}
	}
	return wd
}

// truncateLeft shortens a path to max runes, keeping the tail (most relevant
// for directory names) and prefixing with an ellipsis if truncated.
func truncateLeft(s string, max int) string {
	r := []rune(s)
	if len(r) <= max || max < 4 {
		return s
	}
	return "…" + string(r[len(r)-(max-1):])
}

func welcomeText() string {
	title := brandStyle.Render("◆ atlas.llm") + sysStyle.Render("  local AI chat · on-device inference")

	groups := []struct {
		heading string
		rows    [][2]string
	}{
		{"Models & downloads", [][2]string{
			{"/list", "available models + download status"},
			{"/model", "open picker (↑/↓ + enter) — or /model <name> for direct"},
			{"/download", "engine + current model"},
			{"/download engine", "engine only"},
			{"/download <name>", "engine + that model"},
			{"/download all", "engine + every registered model"},
		}},
		{"Project tools", [][2]string{
			{"/summarize [dir]", "write SUMMARY.md (flags: --max-size=N, --exclude=.ext,...)"},
			{"/grep <query>", "semantic grep across the current directory"},
		}},
		{"Chat", [][2]string{
			{"/help [cmd [sub]]", "this overview, or full detail for one command"},
			{"/clear", "clear the on-screen scrollback (keeps context)"},
			{"/reset", "drop conversation context + server KV cache"},
			{"/compact", "summarize older turns to free up context"},
			{"/set [k [v]]", "settings: max_tokens, ctx_size, gpu_layers, engine_variant"},
			{"/config", "everything at once; /config save|load|show|list for named profiles"},
			{"/tools [on|off|list]", "agentic tool-use (read/write/grep/run_cmd; off by default)"},
			{"/ama [on|off]", "let the agent ask you questions with interactive lists"},
			{"/mcp [connect|tools]", "connect MCP servers (Slack, Confluence, …); /mcp help for setup"},
			{"/quit  /exit", "leave chat (or press ctrl+c)"},
			{"tab", "complete slash commands and their arguments"},
			{"ctrl+j  /  alt+enter", "insert a newline for a multi-line prompt"},
			{"↑ / ↓", "recall previous / next input (cursor keys inside multi-line)"},
			{"esc", "stop generation in progress"},
			{"pgup / pgdn", "scroll the transcript"},
			{"/yesman", "auto-approve destructive tools for this session only"},
		}},
	}

	// Pad all command strings to a common width so the description column
	// lines up across every group, regardless of the longest command.
	groups = append(groups, struct {
		heading string
		rows    [][2]string
	}{"Performance", gpuHelpRows()})

	cmdWidth := 0
	for _, g := range groups {
		for _, r := range g.rows {
			if w := lipgloss.Width(r[0]); w > cmdWidth {
				cmdWidth = w
			}
		}
	}
	cmdColStyle := lipgloss.NewStyle().Foreground(colAccent).Bold(true).Width(cmdWidth + 4)
	descStyle := lipgloss.NewStyle().Foreground(colMuted)
	headingStyle := lipgloss.NewStyle().Foreground(colDim).Bold(true)

	var lines []string
	lines = append(lines, title, "")
	for i, g := range groups {
		if i > 0 {
			lines = append(lines, "")
		}
		lines = append(lines, "  "+headingStyle.Render(g.heading))
		for _, r := range g.rows {
			lines = append(lines, "    "+cmdColStyle.Render(r[0])+descStyle.Render(r[1]))
		}
	}
	lines = append(lines, "",
		sysStyle.Render("  Dependencies aren't downloaded automatically — start with ")+
			footerKeyStyle.Render("/download")+sysStyle.Render("."))
	return strings.Join(lines, "\n")
}

func (m chatModel) Init() tea.Cmd {
	return tea.Batch(textarea.Blink, m.spinner.Tick, warmupServerCmd())
}

// warmupServerCmd boots llama-server in a goroutine at chat startup so the
// model is loaded before the user's first message lands.
func warmupServerCmd() tea.Cmd {
	return func() tea.Msg {
		if !isEngineDownloaded() {
			return serverReadyMsg{} // nothing to warm up yet
		}
		m, err := currentModel()
		if err != nil {
			return serverReadyMsg{err: err}
		}
		if !isModelDownloaded(m) {
			return serverReadyMsg{} // let /download flow handle it
		}
		s, err := ensureServer()
		if err != nil {
			return serverReadyMsg{err: err}
		}
		return serverReadyMsg{model: s.model.Name}
	}
}

// ruleMarker is a placeholder stored in m.rendered wherever a full-width
// separator belongs. It is expanded at render time rather than baked in, so
// the rules re-fit when the terminal is resized.
const ruleMarker = "\x00rule\x00"

func (m *chatModel) refresh() {
	m.viewport.SetContent(m.renderTranscript())
	m.viewport.GotoBottom()
}

// renderTranscript joins the transcript, expanding rule markers to the
// current width. Built with a single Builder because this runs on every
// streamed token.
func (m *chatModel) renderTranscript() string {
	rule := m.horizontalRule()
	var b strings.Builder
	for i, line := range m.rendered {
		if i > 0 {
			b.WriteByte('\n')
		}
		if line == ruleMarker {
			b.WriteString(rule)
			continue
		}
		b.WriteString(line)
	}
	return b.String()
}

// horizontalRule spans the viewport width.
func (m *chatModel) horizontalRule() string {
	w := m.viewport.Width
	if w <= 0 {
		w = 80
	}
	return ruleStyle.Render(strings.Repeat("─", w))
}

// pushRule appends a separator, collapsing consecutive ones.
func (m *chatModel) pushRule() {
	if n := len(m.rendered); n > 0 && m.rendered[n-1] == ruleMarker {
		return
	}
	m.rendered = append(m.rendered, ruleMarker)
}

// lastRenderedIsBlank reports whether the most recent entry is an empty
// separator line or a rule, so pushers can avoid stacking separators.
func (m *chatModel) lastRenderedIsBlank() bool {
	if len(m.rendered) == 0 {
		return true
	}
	last := m.rendered[len(m.rendered)-1]
	// A rule already separates blocks, so it counts as blank for the
	// purposes of not stacking further separators on top of it.
	return last == ruleMarker || strings.TrimSpace(last) == ""
}

func (m *chatModel) pushBlank() {
	if !m.lastRenderedIsBlank() {
		m.rendered = append(m.rendered, "")
	}
}

func (m *chatModel) pushSystem(s string) {
	m.rendered = append(m.rendered, sysStyle.Render("· "+s))
	m.refresh()
}

func (m *chatModel) pushUser(s string) {
	m.pushBlank()
	m.rendered = append(m.rendered, userPillStyle.Render("YOU")+"  "+s)
	m.refresh()
}

func (m *chatModel) pushAssistant(s string) {
	m.pushBlank()
	m.rendered = append(m.rendered, assistantPillStyle.Render("ATLAS"))
	m.rendered = append(m.rendered, m.renderMarkdown(s))
	m.pushRule()
	m.refresh()
}

// renderMarkdown runs the assistant's reply through glamour so headings,
// lists, code fences, and inline styling render as ANSI. Falls back to the
// raw text if the renderer fails or the content is empty. The renderer is
// cached and rebuilt only when the viewport width changes.
func (m *chatModel) renderMarkdown(s string) string {
	if strings.TrimSpace(s) == "" {
		return s
	}
	width := m.viewport.Width
	if width <= 0 {
		width = 80
	}
	wrap := width - 4
	if wrap < 20 {
		wrap = 20
	}
	if m.mdRenderer == nil || m.mdWidth != wrap {
		r, err := glamour.NewTermRenderer(
			glamour.WithStandardStyle(markdownStyleName()),
			glamour.WithWordWrap(wrap),
		)
		if err != nil {
			return s
		}
		m.mdRenderer = r
		m.mdWidth = wrap
	}
	out, err := m.mdRenderer.Render(s)
	if err != nil {
		return s
	}
	return strings.Trim(out, "\n")
}

func (m *chatModel) pushError(s string) {
	m.pushBlank()
	m.rendered = append(m.rendered, errPillStyle.Render("ERROR")+"  "+errTextStyle.Render(s))
	m.refresh()
}

// handleSet implements `/set`, `/set <key>`, and `/set <key> <value>`. With
// no args it lists current settings; with a key alone it prints that
// setting; with key+value it validates and persists.
func (m *chatModel) handleSet(args []string) {
	cfg, err := loadConfig()
	if err != nil {
		m.pushError("load config: " + err.Error())
		return
	}
	if len(args) == 0 {
		m.pushSystem(renderSettingsList(cfg))
		return
	}
	key := strings.ToLower(args[0])
	// `/set <key>` with no value explains the setting rather than just
	// echoing it — choosing a value needs the limits and the tradeoff.
	if len(args) < 2 {
		if s, ok := findSetting(key); ok {
			m.pushSystem(renderSettingDetail(s, cfg))
			return
		}
	}
	// Some settings configure a llama-server this install isn't running.
	// Saving them is still right — they apply again on `/set endpoint local`
	// — but silently accepting one that changes nothing is the failure mode
	// this whole session has been about.
	if ep, _ := remoteEndpoint(); ep != "" && remoteDecidesSetting(key) {
		m.pushSystem(fmt.Sprintf(
			"Note: %s is decided by the server at %s. This saves the value locally, "+
				"but nothing changes until `/set endpoint local`.", key, ep))
	}
	switch key {
	case "ctx_size":
		val := strings.ToLower(args[1])
		if val == "auto" || val == "default" {
			cfg.CtxSize = 0
		} else {
			n, err := strconv.Atoi(val)
			if err != nil || n <= 0 {
				m.pushError(fmt.Sprintf("invalid ctx_size=%q (expected `auto` or a token count)", args[1]))
				return
			}
			if n < minConfigurableCtx {
				m.pushError(fmt.Sprintf("ctx_size=%d is too small — the system prompt and tool "+
					"definitions alone need more than that (minimum %d)", n, minConfigurableCtx))
				return
			}
			if n > maxConfigurableCtx {
				m.pushError(fmt.Sprintf("ctx_size=%d exceeds the %d ceiling atlas.llm allows — "+
					"no model here is trained past that", n, maxConfigurableCtx))
				return
			}
			if trained := currentModelTrainedContext(); trained > 0 && n > trained {
				m.pushError(fmt.Sprintf("ctx_size=%d exceeds what %s was trained for (%d). "+
					"Going beyond it degrades quality rather than extending memory.",
					n, cfg.CurrentModel, trained))
				return
			}
			cfg.CtxSize = n
		}
		if err := saveConfig(cfg); err != nil {
			m.pushError("save config: " + err.Error())
			return
		}
		m.pushSystem(fmt.Sprintf("ctx_size = %s\nTakes effect on the next message (the model server restarts). "+
			"A larger window uses proportionally more memory for the KV cache.",
			ctxSizeDisplay(cfg)))

	case "reasoning":
		switch strings.ToLower(args[1]) {
		case reasoningOn, "true", "yes":
			cfg.Reasoning = reasoningOn
		case reasoningOff, "false", "no":
			cfg.Reasoning = reasoningOff
		case reasoningAuto, "default":
			cfg.Reasoning = ""
		default:
			m.pushError(fmt.Sprintf("invalid reasoning=%q (expected on, off, or auto)", args[1]))
			return
		}
		if err := saveConfig(cfg); err != nil {
			m.pushError("save config: " + err.Error())
			return
		}
		msg := fmt.Sprintf("reasoning = %s", reasoningDisplay(cfg))
		if !reasoningEnabled(cfg) {
			msg += "\n\nReplies will be markedly faster. Tool-use and multi-step " +
				"reasoning may get less reliable — turn it back on with `/set reasoning on`."
		}
		m.pushSystem(msg)

	case "max_tool_rounds":
		val := strings.ToLower(args[1])
		switch val {
		case "off", "none", "unlimited":
			cfg.MaxToolRounds = -1
		case "auto", "default":
			cfg.MaxToolRounds = 0
		default:
			n, err := strconv.Atoi(val)
			if err != nil || n <= 0 {
				m.pushError(fmt.Sprintf(
					"invalid max_tool_rounds=%q (expected a positive number, `off`, or `default`)", args[1]))
				return
			}
			cfg.MaxToolRounds = n
		}
		if err := saveConfig(cfg); err != nil {
			m.pushError("save config: " + err.Error())
			return
		}
		msg := fmt.Sprintf("max_tool_rounds = %s", maxToolRoundsDisplay(cfg))
		if resolveMaxToolRounds(cfg) == unlimitedToolRounds {
			msg += fmt.Sprintf("\n\nNo round limit. A stuck model is still caught: identical "+
				"repeated calls stop the turn after %d attempts, and esc stops it at any time.",
				maxIdenticalCalls+1)
		}
		m.pushSystem(msg)

	case "gpu_layers":
		val := strings.ToLower(args[1])
		if val == "auto" {
			cfg.GPULayers = nil
		} else {
			n, err := strconv.Atoi(val)
			if err != nil || n < 0 {
				m.pushError(fmt.Sprintf("invalid gpu_layers=%q (expected `auto`, 0 for CPU-only, or a layer count)", args[1]))
				return
			}
			cfg.GPULayers = &n
		}
		if err := saveConfig(cfg); err != nil {
			m.pushError("save config: " + err.Error())
			return
		}
		msg := fmt.Sprintf("gpu_layers = %s", gpuLayersDisplay(cfg))
		if resolveGPULayers(cfg) > 0 && runtime.GOOS != "darwin" &&
			!engineVariantIsGPU(installedEngineVariant()) {
			msg += "\n\nNote: the installed engine is a CPU-only build, so this will have no effect.\n" +
				"Run `/set engine_variant <" + strings.Join(engineVariantNames()[2:], "|") +
				">` then `/download engine` for GPU support."
		}
		msg += "\nTakes effect on the next message (the model server restarts)."
		m.pushSystem(msg)

	case "engine_variant":
		val := strings.ToLower(args[1])
		switch val {
		case engineVariantAuto, "":
			cfg.EngineVariant = ""
		case engineVariantCPU, engineVariantVulkan, engineVariantCUDA, engineVariantHIP:
			cfg.EngineVariant = val
		default:
			m.pushError(fmt.Sprintf("invalid engine_variant=%q (expected %s)",
				args[1], strings.Join(engineVariantNames(), ", ")))
			return
		}
		want := resolveEngineVariant(cfg.EngineVariant)
		if val != engineVariantAuto && val != "" && want != val {
			m.pushError(fmt.Sprintf("no %s llama.cpp build for %s/%s (available here: %s)",
				val, runtime.GOOS, runtime.GOARCH, strings.Join(engineVariantNames(), ", ")))
			return
		}
		asset, err := engineAssetSuffix(want)
		if err != nil {
			m.pushError(err.Error())
			return
		}
		if err := saveConfig(cfg); err != nil {
			m.pushError("save config: " + err.Error())
			return
		}
		msg := fmt.Sprintf("engine_variant = %s", engineVariantDisplay(cfg))
		if installedEngineVariant() != want {
			msg += fmt.Sprintf("\n\nInstalled engine is %q — run `/download engine` to replace it with the %s build (%s).",
				installedEngineVariant(), want, asset.Size)
		}
		m.pushSystem(msg)

	case "max_tokens":
		n, err := strconv.Atoi(args[1])
		if err != nil || n <= 0 {
			m.pushError(fmt.Sprintf("invalid max_tokens=%q (expected positive integer)", args[1]))
			return
		}
		// A reply longer than most of the context window would leave no room
		// for the prompt and history, so the ceiling tracks ctx_size.
		if ceiling := maxTokensCeiling(cfg); n > ceiling {
			m.pushError(fmt.Sprintf(
				"max_tokens=%d exceeds the ceiling of %d for a %d-token context "+
					"(the rest is needed for the prompt and history). "+
					"Raise it with `/set ctx_size N` first.",
				n, ceiling, resolveCtxSize(cfg)))
			return
		}
		cfg.MaxTokens = n
		if err := saveConfig(cfg); err != nil {
			m.pushError("save config: " + err.Error())
			return
		}
		m.pushSystem(fmt.Sprintf("max_tokens = %d", n))

	case "endpoint":
		ep, err := normalizeEndpoint(args[1])
		if err != nil {
			m.pushError(err.Error())
			return
		}
		was, _ := remoteEndpoint()
		cfg.Endpoint = ep
		if err := saveConfig(cfg); err != nil {
			m.pushError("save config: " + err.Error())
			return
		}
		// The old server is now the wrong one either way: switching to a
		// remote makes the local subprocess dead weight, and switching back
		// leaves a remote attachment pointing nowhere.
		if was != ep {
			shutdownServer()
		}
		if ep == "" {
			clearRemoteStatus()
			m.pushSystem("endpoint = local\nInference moves back to this machine on your next message.")
			return
		}
		// Check it now rather than letting a typo look fine until the first
		// message fails. A LAN probe is milliseconds; the alternative is the
		// user discovering the mistake three commands later.
		_, key := remoteEndpoint()
		st := probeRemote(ep, key)
		setRemoteStatus(st)
		startHeartbeat()
		m.pushSystem(renderEndpointProbe(ep, st))

	case "endpoint_key":
		cfg.EndpointKey = strings.TrimSpace(args[1])
		if strings.EqualFold(cfg.EndpointKey, "none") || strings.EqualFold(cfg.EndpointKey, "off") {
			cfg.EndpointKey = ""
		}
		if err := saveConfig(cfg); err != nil {
			m.pushError("save config: " + err.Error())
			return
		}
		shutdownServer() // reattach so the new key is used
		if cfg.EndpointKey == "" {
			m.pushSystem("endpoint_key cleared.")
			return
		}
		m.pushSystem("endpoint_key set.\nTakes effect on your next message.")

	default:
		// Settings that registered an Apply are handled generically: the
		// registry validates and writes, this branch persists and confirms.
		// The older settings keep bespoke cases above for their side
		// effects (probes, server shutdowns).
		s, ok := findSetting(key)
		if !ok || s.Apply == nil {
			m.pushError(fmt.Sprintf("unknown setting: %s (supported: %s)", key, strings.Join(settingKeys(), ", ")))
			return
		}
		if err := s.Apply(&cfg, args[1]); err != nil {
			m.pushError(err.Error())
			return
		}
		if err := saveConfig(cfg); err != nil {
			m.pushError("save config: " + err.Error())
			return
		}
		msg := fmt.Sprintf("%s = %s", s.Key, s.Value(cfg))
		if s.Restart {
			msg += "\nTakes effect on the next message (the model server restarts)."
		} else {
			msg += "\nApplies to your next message."
		}
		m.pushSystem(msg)
	}
}

// lastAssistantContent returns the raw text of the most recent assistant
// reply (pre-markdown-rendering) so Ctrl+Y copies something pasteable
// rather than ANSI-decorated output.
func (m *chatModel) lastAssistantContent() string {
	for i := len(m.history) - 1; i >= 0; i-- {
		if m.history[i].Role == "assistant" {
			return m.history[i].Content
		}
	}
	return ""
}

// openModelPicker populates the picker with the available registry and
// positions the cursor on the currently-selected model.
func (m *chatModel) openModelPicker() {
	m.pickerItems = append([]Model(nil), availableModels...)
	m.pickerIdx = 0
	for i, mm := range m.pickerItems {
		if mm.Name == m.modelName {
			m.pickerIdx = i
			break
		}
	}
	m.picking = "model"
	m.renderPicker()
}

// pickerCancel closes the picker without applying any change.
func (m *chatModel) pickerCancel() {
	m.picking = ""
	m.pickerItems = nil
	m.pickerIdx = 0
	m.refresh()
	m.pushSystem("Model picker cancelled.")
}

// pickerConfirm applies the highlighted choice and closes the picker.
// Returns a tea.Cmd for any follow-up work (re-warming the server).
func (m *chatModel) pickerConfirm() tea.Cmd {
	if m.pickerIdx < 0 || m.pickerIdx >= len(m.pickerItems) {
		m.pickerCancel()
		return nil
	}
	target := m.pickerItems[m.pickerIdx]
	m.picking = ""
	m.pickerItems = nil
	m.pickerIdx = 0
	m.refresh()
	m.applyModelSelection(target)
	return nil
}

// applyModelSelection persists the new choice, updates the header, and
// tells the user whether the model still needs to be /download'ed.
// ensureServer will restart the llama-server subprocess on the next
// inference call if the model actually changed.
func (m *chatModel) applyModelSelection(target Model) {
	// The model is loaded into the server's memory at spawn; a client has no
	// way to change it over HTTP. Saving the selection anyway would leave the
	// header naming a model that isn't answering.
	if ep, _ := remoteEndpoint(); ep != "" {
		m.pushError(fmt.Sprintf(
			"inference runs on %s, and only that machine can change which model is loaded.\n"+
				"Run `/set endpoint local` to use this machine's models instead.", ep))
		return
	}
	cfg, _ := loadConfig()
	cfg.CurrentModel = target.Name
	if err := saveConfig(cfg); err != nil {
		m.pushError(err.Error())
		return
	}
	if m.modelName == target.Name {
		m.pushSystem(fmt.Sprintf("Already using %s.", target.Name))
		return
	}
	m.modelName = target.Name
	msg := fmt.Sprintf("Switched model to %s.", target.Name)
	if !isModelDownloaded(target) {
		msg += fmt.Sprintf(" (not downloaded — run /download %s)", target.Name)
	}
	m.pushSystem(msg)
}

// renderPicker draws the picker into the viewport so it overlays the
// scrollback while active. Re-rendered on every arrow-key press.
func (m *chatModel) renderPicker() {
	title := brandStyle.Render("Select a model") + sysStyle.Render("  (↑/↓ move · enter select · esc cancel)")
	rowSelected := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#0B1220")).
		Background(colAccent).
		Bold(true).
		Padding(0, 1)
	rowNormal := lipgloss.NewStyle().Padding(0, 1)
	dot := sysStyle.Render(" · ")

	lines := []string{title, ""}
	for i, mm := range m.pickerItems {
		marker := "  "
		if mm.Name == m.modelName {
			marker = brandStyle.Render("● ")
		}
		status := sysStyle.Render("not downloaded")
		if isModelDownloaded(mm) {
			status = lipgloss.NewStyle().Foreground(colAssistant).Render("downloaded")
		}
		row := fmt.Sprintf("%s%-28s  %s%s%s", marker, mm.Name,
			metaLabelStyle.Render(mm.Size), dot, status)
		if note := modelResourceNote(mm); note != "" {
			row += dot + sysStyle.Render(note)
		}
		if i == m.pickerIdx {
			lines = append(lines, rowSelected.Render(row))
		} else {
			lines = append(lines, rowNormal.Render(row))
		}
	}
	m.viewport.SetContent(strings.Join(lines, "\n"))
	// The list can be taller than the viewport. Pinning to the top left
	// everything below the fold unreachable: the cursor moved but you could
	// not see it, so later models looked absent from the picker entirely.
	m.scrollSelectionIntoView(pickerHeaderLines + m.pickerIdx)
}

// pickerHeaderLines is how many lines precede the first selectable row —
// the title and the blank line under it.
const pickerHeaderLines = 2

// scrollSelectionIntoView nudges the viewport just far enough to show the
// given content line, leaving the offset alone when it is already visible.
func (m *chatModel) scrollSelectionIntoView(line int) {
	h := m.viewport.Height
	if h <= 0 {
		return
	}
	off := m.viewport.YOffset
	switch {
	case line <= pickerHeaderLines:
		// Near the top, show the title too rather than scrolling the
		// minimum amount and leaving the header off-screen.
		off = 0
	case line < off:
		off = line
	case line >= off+h:
		off = line - h + 1
	default:
		return
	}
	if off < 0 {
		off = 0
	}
	m.viewport.SetYOffset(off)
}

func (m chatModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		// Layout: header(1) + rule(1) + viewport(...) + rule(1) + input(3) + footer(1)
		headerH := 1
		ruleH := 2
		taH := 2
		footerH := 1
		vpH := msg.Height - headerH - ruleH - taH - footerH
		if vpH < 3 {
			vpH = 3
		}
		m.viewport.Width = msg.Width
		m.viewport.Height = vpH
		m.textarea.SetWidth(msg.Width - 2)
		barW := msg.Width - 20
		if barW < 20 {
			barW = 20
		}
		if barW > 60 {
			barW = 60
		}
		m.progress.Width = barW
		m.refresh()

	case tea.KeyMsg:
		if m.picking == "tool_confirm" {
			switch msg.Type {
			case tea.KeyCtrlC, tea.KeyEsc:
				if c := m.resolveConfirm(false); c != nil {
					cmds = append(cmds, c)
				}
			case tea.KeyUp, tea.KeyDown:
				m.confirmIdx = 1 - m.confirmIdx
				m.renderConfirm()
			case tea.KeyEnter:
				if c := m.resolveConfirm(m.confirmIdx == 0); c != nil {
					cmds = append(cmds, c)
				}
			}
			return m, tea.Batch(cmds...)
		}
		if m.picking == "mcp_add" {
			switch msg.Type {
			case tea.KeyCtrlC, tea.KeyEsc:
				m.mcpPickerCancel()
			case tea.KeyUp:
				if m.pickerIdx > 0 {
					m.pickerIdx--
					m.renderMCPPicker()
				}
			case tea.KeyDown:
				if m.pickerIdx < len(m.mcpPickerItems)-1 {
					m.pickerIdx++
					m.renderMCPPicker()
				}
			case tea.KeyEnter:
				if c := m.mcpPickerConfirm(); c != nil {
					cmds = append(cmds, c)
				}
			}
			return m, tea.Batch(cmds...)
		}
		if m.picking == "ama" {
			// The agent's ask_user picker. ↑/↓ move, space toggles a checkbox,
			// Enter submits, Esc dismisses (the model is told to proceed).
			switch msg.Type {
			case tea.KeyCtrlC, tea.KeyEsc:
				if c := m.resolveAMA(false); c != nil {
					cmds = append(cmds, c)
				}
			case tea.KeyUp:
				m.amaMove(-1)
			case tea.KeyDown:
				m.amaMove(1)
			case tea.KeySpace:
				m.amaToggle()
			case tea.KeyEnter:
				if c := m.resolveAMA(true); c != nil {
					cmds = append(cmds, c)
				}
			}
			return m, tea.Batch(cmds...)
		}
		if m.picking != "" {
			// While the picker is open, swallow key events and don't let the
			// textarea see them. ↑/↓ move, Enter selects, Esc/Ctrl+C cancels.
			switch msg.Type {
			case tea.KeyCtrlC, tea.KeyEsc:
				m.pickerCancel()
			case tea.KeyUp:
				if m.pickerIdx > 0 {
					m.pickerIdx--
					m.renderPicker()
				}
			case tea.KeyDown:
				if m.pickerIdx < len(m.pickerItems)-1 {
					m.pickerIdx++
					m.renderPicker()
				}
			case tea.KeyEnter:
				cmd := m.pickerConfirm()
				if cmd != nil {
					cmds = append(cmds, cmd)
				}
			}
			return m, tea.Batch(cmds...)
		}
		switch msg.Type {
		case tea.KeyCtrlC:
			return m, tea.Quit
		case tea.KeyEsc:
			if m.stopInflight() {
				return m, tea.Batch(cmds...)
			}
		case tea.KeyPgUp:
			m.viewport.ViewUp()
			return m, tea.Batch(cmds...)
		case tea.KeyPgDown:
			m.viewport.ViewDown()
			return m, tea.Batch(cmds...)
		case tea.KeyTab:
			if m.tabComplete() {
				return m, tea.Batch(cmds...)
			}
		case tea.KeyUp:
			if m.recallPrev() {
				return m, tea.Batch(cmds...)
			}
		case tea.KeyDown:
			if m.recallNext() {
				return m, tea.Batch(cmds...)
			}
		case tea.KeyCtrlY:
			content := m.lastAssistantContent()
			if content == "" {
				m.pushSystem("Nothing to copy — no assistant reply yet.")
			} else if err := clipboard.WriteAll(content); err != nil {
				m.pushError("copy failed: " + err.Error())
			} else {
				m.pushSystem(fmt.Sprintf("Copied last reply (%d chars) to clipboard.", len(content)))
			}
			return m, tea.Batch(cmds...)
		case tea.KeyCtrlJ:
			// Ctrl+J inserts a newline for a multi-line prompt. Enter submits,
			// so this is the portable way to compose across lines (terminals
			// can't tell Shift+Enter from Enter).
			m.textarea.InsertRune('\n')
			return m, tea.Batch(cmds...)
		case tea.KeyEnter:
			if msg.Alt {
				// Alt+Enter is the other multi-line key: compose a newline
				// instead of submitting.
				m.textarea.InsertRune('\n')
				return m, tea.Batch(cmds...)
			}
			input := strings.TrimSpace(m.textarea.Value())
			if input == "" {
				// Swallow the key instead of breaking: falling through hands
				// Enter to the textarea, which inserts a newline — the cursor
				// dropped a line and the placeholder tip vanished. An empty
				// prompt has nothing to send, so the key means nothing here,
				// busy or not — busy just made it easy to hit, since the
				// model starts in the busy state while the server warms up.
				return m, tea.Batch(cmds...)
			}
			if m.busy {
				// While the model is generating, swallow Enter entirely: it
				// can't submit yet, and falling through to the textarea would
				// just type a newline and drop the cursor a line. Other keys
				// still reach the textarea, so a next message can be drafted
				// while the reply streams; Enter sends it once generation ends.
				return m, tea.Batch(cmds...)
			}
			m.textarea.Reset()
			m.recordHistory(input)

			if strings.HasPrefix(input, "/") {
				cmd := m.handleSlash(input)
				if cmd != nil {
					cmds = append(cmds, cmd)
				}
			} else if m.agentEnabled {
				m.pushUser(input)
				if len(m.agentMsgs) == 0 {
					m.agentMsgs = append(m.agentMsgs, ChatMsg{Role: "system", Content: agentSystemPromptNow()})
				}
				m.agentMsgs = append(m.agentMsgs, ChatMsg{Role: "user", Content: input})
				m.stepCount = 0
				m.repeatedCalls = map[string]int{}
				m.busy = true
				m.busyReason = "thinking"
				m.busyStart = time.Now()
				cmds = append(cmds, runAgentStepCmd(m.newInflight(), m.agentMsgs), m.spinner.Tick)
			} else {
				m.pushUser(input)
				m.history = append(m.history, ChatMessage{Role: "user", Content: input})
				m.busy = true
				m.busyReason = "thinking"
				m.busyStart = time.Now()
				hist := append([]ChatMessage(nil), m.history[:len(m.history)-1]...)
				cmds = append(cmds, runChatStreamCmd(m.newInflight(), hist, input), m.spinner.Tick)
			}
			// Return here: the textarea was just reset, and falling through to
			// the shared textarea.Update below would type this Enter into the
			// empty box, leaving a stray newline that drops the cursor a line.
			return m, tea.Batch(cmds...)
		}

	case assistantDeltaMsg:
		m.applyDelta(msg)

	case assistantReplyMsg:
		if m.canceled {
			// A reply that landed after Esc belongs to an abandoned turn.
			m.canceled = false
			break
		}
		m.busy = false
		m.busyReason = ""
		m.history = append(m.history, ChatMessage{Role: "assistant", Content: msg.content})
		// A streamed reply is already on screen; only re-render it as
		// markdown. Non-streamed replies still need pushing.
		if !m.finishStream(msg.content) {
			m.pushAssistant(msg.content)
		}
		m.maybeSuggestCompact()

	case agentStepMsg:
		if c := m.handleAgentStep(msg); c != nil {
			cmds = append(cmds, c)
		}

	case compactDoneMsg:
		m.applyCompaction(msg)

	case toolRanMsg:
		if c := m.handleToolRan(msg); c != nil {
			cmds = append(cmds, c)
		}

	case inferenceErrMsg:
		if m.inflightCanceled(msg.err) {
			break
		}
		m.busy = false
		m.busyReason = ""
		m.dropUnansweredUser()
		m.pushError(msg.err.Error())

	case sysMsg:
		m.pushSystem(msg.content)

	case serverReadyMsg:
		if msg.err != nil {
			m.busy = false
			m.busyReason = ""
			m.pushError("model load failed: " + msg.err.Error())
		} else if msg.model != "" {
			m.busy = false
			m.busyReason = ""
			m.pushSystem(fmt.Sprintf("Model %s loaded and ready.", msg.model))
		} else {
			m.busy = false
			m.busyReason = ""
		}

	case summarizeDoneMsg:
		m.busy = false
		m.busyReason = ""
		if msg.err != nil {
			m.pushError(msg.err.Error())
		} else {
			m.pushSystem(fmt.Sprintf("Summary written to %s", msg.path))
		}

	case grepDoneMsg:
		m.busy = false
		m.busyReason = ""
		if msg.err != nil {
			m.pushError(msg.err.Error())
		} else {
			m.pushSystem(fmt.Sprintf("grep: %q", msg.query))
			m.pushSystem(formatGrepHits(msg.hits))
		}

	case downloadProgressMsg:
		if cmd := m.applyDownloadProgress(msg); cmd != nil {
			cmds = append(cmds, cmd)
		}

	case remoteStatusMsg:
		// The badge reads cached state, so there is nothing to apply here —
		// receiving the message is what triggers the repaint.
		return m, nil

	case downloadDoneMsg:
		m.busy = false
		m.busyReason = ""
		m.dlName = ""
		m.dlWritten = 0
		m.dlTotal = 0
		m.dlMeter = downloadMeter{}
		if msg.err != nil {
			m.pushError(msg.err.Error())
		} else {
			m.pushSystem(fmt.Sprintf("Downloaded: %s", msg.what))
		}

	case spinner.TickMsg:
		if m.busy {
			var cmd tea.Cmd
			m.spinner, cmd = m.spinner.Update(msg)
			cmds = append(cmds, cmd)
		}

	case progress.FrameMsg:
		pm, cmd := m.progress.Update(msg)
		m.progress = pm.(progress.Model)
		cmds = append(cmds, cmd)
	}

	var cmd tea.Cmd
	m.textarea, cmd = m.textarea.Update(msg)
	cmds = append(cmds, cmd)
	// Deliberately keep key events away from the viewport. Its default
	// keymap is vim-style (j/k/u/d/f/b/space), so forwarding keystrokes
	// meant typing "j" scrolled the transcript instead of entering a
	// character. Scrolling is bound explicitly on PgUp/PgDn instead; other
	// message types (mouse wheel, resize) still reach it.
	if _, isKey := msg.(tea.KeyMsg); !isKey {
		m.viewport, cmd = m.viewport.Update(msg)
		cmds = append(cmds, cmd)
	}

	return m, tea.Batch(cmds...)
}

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
		ResetUsage()
		if s, err := ensureServer(); err == nil {
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
		if isEngineDownloaded() {
			engineStatus = "engine: downloaded"
		}
		b.WriteString("  " + engineStatus + "\n\n")
		for _, mm := range availableModels {
			status := "not downloaded"
			if isModelDownloaded(mm) {
				status = "downloaded"
			}
			marker := " "
			if mm.Name == m.modelName {
				marker = "*"
			}
			fmt.Fprintf(&b, " %s %-25s %-8s %-16s %s\n",
				marker, mm.Name, mm.Size, status, modelResourceNote(mm))
		}
		if total, ok := systemRAM(); ok {
			fmt.Fprintf(&b, "\n(* = current model · RAM share is the weights against %s total;\n"+
				" the context window adds more on top — see /help set ctx_size)", formatBytes(total))
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
		target, ok := findModel(name)
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
		if targets.engine && !isEngineDownloaded() {
			summary = append(summary, "engine")
		}
		for _, mm := range targets.models {
			if !isModelDownloaded(mm) {
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
					names, _ := listProfiles()
					return m.completeToken(head+" "+sub[0]+" ", sub[1], names)
				}
			}
		}
		return false
	}
	var pool []string
	switch head {
	case "/model":
		for _, mm := range availableModels {
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
		for _, mm := range availableModels {
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

func (m chatModel) View() string {
	width := m.width
	if width < 1 {
		width = 80
	}

	header := m.renderHeader(width)
	topRule := ruleStyle.Render(strings.Repeat("─", width))
	body := m.viewport.View()
	midRule := ruleStyle.Render(strings.Repeat("─", width))
	input := m.renderInput(width)
	footer := m.renderFooter(width)

	return lipgloss.JoinVertical(lipgloss.Left, header, topRule, body, midRule, input, footer)
}

func (m chatModel) renderHeader(width int) string {
	dot := sepStyle.Render(" • ")
	brand := brandStyle.Render("◆ atlas.llm")
	model := metaLabelStyle.Render("model ") + metaValueStyle.Render(m.modelName)
	remote := renderRemoteBadge()

	ctxSeg := renderCtxSegment()

	var stateSeg string
	switch {
	case m.busy && m.busyReason == "downloading":
		stateSeg = busyStyle.Render(m.spinner.View() + " downloading")
	case m.busy:
		elapsed := ""
		if !m.busyStart.IsZero() {
			elapsed = fmt.Sprintf(" %ds", int(time.Since(m.busyStart).Seconds()))
		}
		stateSeg = busyStyle.Render(m.spinner.View() + " " + m.busyReason + elapsed)
	default:
		stateSeg = sysStyle.Render("● ready")
	}

	right := stateSeg
	if ctxSeg != "" {
		right = ctxSeg + dot + stateSeg
	}
	// Memory sits next to the context gauge: both answer "how much room is
	// left", one in tokens and one in bytes.
	if memSeg := renderMemSegment(); memSeg != "" {
		right = memSeg + dot + right
	}
	if m.yesman {
		// Persistent, loud, and impossible to miss while it's armed.
		right = lipgloss.NewStyle().Foreground(colErr).Bold(true).Render("⚠ yesman") + dot + right
	}

	dirMax := width - lipgloss.Width(brand) - lipgloss.Width(model) - lipgloss.Width(right) - 12
	if remote != "" {
		dirMax -= lipgloss.Width(remote) + 3
	}
	if dirMax < 12 {
		dirMax = 12
	}
	dir := metaLabelStyle.Render("cwd ") + metaValueStyle.Render(truncateLeft(m.cwd, dirMax))

	left := brand
	// The badge sits immediately after the brand, before the model: which
	// machine is answering outranks what it is running.
	if remote != "" {
		left += dot + remote
	}
	left += dot + model + dot + dir

	gap := width - lipgloss.Width(left) - lipgloss.Width(right) - 2
	if gap < 1 {
		gap = 1
	}
	return " " + left + strings.Repeat(" ", gap) + right + " "
}

// renderCtxSegment formats the last-known prompt/ctx usage as "ctx N/C (P%)"
// with the percent colored by fill level (muted → amber → red). Returns an
// empty string if no inference has completed yet.
func renderCtxSegment() string {
	u := GetLastUsage()
	if u.ContextSize == 0 {
		return ""
	}
	used := u.PromptTokens + u.CompletionTokens
	if u.TotalTokens > used {
		used = u.TotalTokens
	}
	pct := 0
	if u.ContextSize > 0 {
		pct = (used * 100) / u.ContextSize
	}
	label := metaLabelStyle.Render("ctx ")
	value := metaValueStyle.Render(fmt.Sprintf("%s/%s", formatTokens(used), formatTokens(u.ContextSize)))
	pctStyle := sysStyle
	switch {
	case pct >= 90:
		pctStyle = lipgloss.NewStyle().Foreground(colErr).Bold(true)
	case pct >= 70:
		pctStyle = lipgloss.NewStyle().Foreground(colBusy).Bold(true)
	}
	return label + value + " " + pctStyle.Render(fmt.Sprintf("(%d%%)", pct))
}

func formatTokens(n int) string {
	switch {
	case n >= 1000:
		return fmt.Sprintf("%.1fK", float64(n)/1000)
	default:
		return fmt.Sprintf("%d", n)
	}
}

func (m chatModel) renderInput(width int) string {
	prompt := brandStyle.Render("❯ ")
	ta := m.textarea.View()
	_ = width
	return prompt + ta
}

func (m chatModel) renderFooter(width int) string {
	_ = width
	if m.busy && m.busyReason == "downloading" {
		var stats string
		if s := m.dlMeter.speed(); s > 0 {
			stats += "  " + formatSpeed(s)
		}
		if e := m.dlMeter.elapsed(time.Now()); e > 0 {
			stats += "  " + formatElapsed(e)
		}
		if m.dlTotal > 0 {
			return footerStyle.Render(fmt.Sprintf(
				"  %s  %s  %s / %s%s",
				m.dlName, m.progress.View(), formatBytes(m.dlWritten), formatBytes(m.dlTotal), stats,
			))
		}
		return footerStyle.Render(fmt.Sprintf("  %s  %s%s", m.dlName, formatBytes(m.dlWritten), stats))
	}

	// While generating, esc is the only thing most people want, so lead
	// with it and drop hints that don't apply mid-turn.
	if m.busy {
		hints := []string{
			footerKeyStyle.Render("esc") + footerStyle.Render(" stop generating"),
			footerKeyStyle.Render("^C") + footerStyle.Render(" quit"),
		}
		return "  " + strings.Join(hints, sepStyle.Render("  ·  "))
	}

	hints := []string{}
	if m.yesman {
		hints = append(hints, lipgloss.NewStyle().Foreground(colErr).Bold(true).
			Render("⚠ yesman on")+footerStyle.Render(" (/yesman off)"))
	}
	hints = append(hints,
		footerKeyStyle.Render("↵")+footerStyle.Render(" send"),
		footerKeyStyle.Render("⇧↵")+footerStyle.Render(" newline"),
		footerKeyStyle.Render("↑↓")+footerStyle.Render(" history"),
		footerKeyStyle.Render("esc")+footerStyle.Render(" stop"),
		footerKeyStyle.Render("^Y")+footerStyle.Render(" copy reply"),
		footerKeyStyle.Render("/help")+footerStyle.Render(" commands"),
		footerKeyStyle.Render("^C")+footerStyle.Render(" quit"),
	)
	sep := sepStyle.Render("  ·  ")
	return "  " + strings.Join(hints, sep)
}

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

// progressToSysMsg returns a progress callback that forwards each status
// line into the bubbletea event loop as a sysMsg — so per-file progress
// renders as muted log lines in the viewport instead of leaking through
// as raw stdout writes and corrupting the alt-screen.
func progressToSysMsg() func(string) {
	return func(s string) {
		if program != nil {
			program.Send(sysMsg{content: s})
		}
	}
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

type downloadTargets struct {
	engine bool
	models []Model
}

// resolveDownloadTargets parses /download args:
//
//	(none)       -> engine + current model
//	engine       -> engine only
//	all          -> engine + every registered model
//	<model-name> -> engine + that specific model
func resolveDownloadTargets(args []string) (downloadTargets, error) {
	if len(args) == 0 {
		cfg, _ := loadConfig()
		cur, ok := findModel(cfg.CurrentModel)
		if !ok {
			return downloadTargets{}, fmt.Errorf("current model %q not in registry", cfg.CurrentModel)
		}
		return downloadTargets{engine: true, models: []Model{cur}}, nil
	}
	switch strings.ToLower(args[0]) {
	case "engine":
		return downloadTargets{engine: true}, nil
	case "all":
		return downloadTargets{engine: true, models: append([]Model(nil), availableModels...)}, nil
	default:
		m, ok := findModel(args[0])
		if !ok {
			return downloadTargets{}, fmt.Errorf("unknown model: %s (try /list)", args[0])
		}
		return downloadTargets{engine: true, models: []Model{m}}, nil
	}
}

// applyDownloadProgress records a progress message into the download state
// the footer renders. A message for a different file restarts the meter —
// chained downloads (engine, then a model) must not inherit the previous
// file's start time or rate. Returns the progress-bar animation cmd, or nil.
func (m *chatModel) applyDownloadProgress(msg downloadProgressMsg) tea.Cmd {
	if msg.name != m.dlName {
		m.dlMeter = downloadMeter{}
	}
	m.dlMeter.observe(time.Now(), msg.written)
	m.dlName = msg.name
	m.dlWritten = msg.written
	m.dlTotal = msg.total
	if msg.total > 0 {
		return m.progress.SetPercent(float64(msg.written) / float64(msg.total))
	}
	return nil
}

// throttledProgress returns a ProgressFn that forwards updates to the bubbletea
// program at most every 100ms (plus one final update at completion).
func throttledProgress(name string) ProgressFn {
	var last time.Time
	return func(written, total int64) {
		done := total > 0 && written >= total
		if !done && time.Since(last) < 100*time.Millisecond {
			return
		}
		last = time.Now()
		if program != nil {
			program.Send(downloadProgressMsg{name: name, written: written, total: total})
		}
	}
}

func runDownloadAllCmd(t downloadTargets) tea.Cmd {
	return func() tea.Msg {
		var done []string
		if t.engine && engineNeedsDownload() {
			if err := downloadEngine(throttledProgress("engine")); err != nil {
				return downloadDoneMsg{what: strings.Join(done, ", "), err: fmt.Errorf("engine: %w", err)}
			}
			done = append(done, "engine")
		}
		for _, mm := range t.models {
			if isModelDownloaded(mm) {
				continue
			}
			if err := downloadModel(mm, throttledProgress(mm.Name)); err != nil {
				return downloadDoneMsg{what: strings.Join(done, ", "), err: fmt.Errorf("%s: %w", mm.Name, err)}
			}
			done = append(done, mm.Name)
		}
		if len(done) == 0 {
			return downloadDoneMsg{what: "nothing (already present)"}
		}
		return downloadDoneMsg{what: strings.Join(done, ", ")}
	}
}

func startChat() (err error) {
	logFile, closeLog, logErr := setupLogging()
	defer closeLog()
	defer shutdownServer()
	// Closes stdio MCP subprocesses so they don't outlive the TUI.
	defer shutdownMCP()
	defer func() {
		if r := recover(); r != nil {
			logPanicln(r)
			err = fmt.Errorf("panic: %v", r)
		}
	}()

	// Must happen before the program starts: this reads a reply off stdin,
	// and once bubbletea owns stdin the reply becomes a keystroke instead.
	detectMarkdownStyle()

	m := newChatModel()
	if logErr == nil {
		m.rendered = append(m.rendered, sysStyle.Render(fmt.Sprintf("Log file: %s", logFile)))
	}

	p := tea.NewProgram(m, tea.WithAltScreen())
	program = p
	defer func() { program = nil }()
	autoConnectMCP()
	// A session that opens against a configured remote should already know
	// whether it is reachable, rather than showing an unknown-state badge
	// until the first heartbeat 15s later.
	if ep, key := remoteEndpoint(); ep != "" {
		// Publish the endpoint synchronously so the badge is present from the
		// first frame; the probe that fills in its state runs in the
		// background, since it can take up to probeTimeout.
		setRemoteStatus(remoteStatus{Endpoint: ep, State: remoteUnknown})
		go func() {
			setRemoteStatus(probeRemote(ep, key))
			startHeartbeat()
			if program != nil {
				program.Send(remoteStatusMsg{})
			}
		}()
	}
	defer stopHeartbeat()
	_, err = p.Run()
	return err
}

// remoteStatusMsg nudges the TUI to repaint after a background probe changes
// the badge. It carries nothing: the state lives in the cache.
type remoteStatusMsg struct{}

// maxInputHistory bounds the recall buffer. Generous enough to cover a
// working session, small enough to stay irrelevant to memory.
const maxInputHistory = 200

// recordHistory appends a submitted line to the recall buffer and resets
// the browse position. Consecutive duplicates are collapsed, so holding
// Enter on the same command doesn't bury the rest of the history.
func (m *chatModel) recordHistory(input string) {
	if n := len(m.inputHistory); n == 0 || m.inputHistory[n-1] != input {
		m.inputHistory = append(m.inputHistory, input)
		if len(m.inputHistory) > maxInputHistory {
			m.inputHistory = m.inputHistory[len(m.inputHistory)-maxInputHistory:]
		}
	}
	m.historyIdx = len(m.inputHistory)
	m.historyDraft = ""
}

// recallPrev walks back through submitted input. Reports whether it
// consumed the key — when it returns false the textarea gets the event and
// moves the cursor instead, so multi-line editing still works.
func (m *chatModel) recallPrev() bool {
	// Only recall from the top line; elsewhere Up means "move up a line".
	if m.textarea.Line() != 0 || len(m.inputHistory) == 0 || m.historyIdx == 0 {
		return false
	}
	if m.historyIdx == len(m.inputHistory) {
		m.historyDraft = m.textarea.Value()
	}
	m.historyIdx--
	m.setInput(m.inputHistory[m.historyIdx])
	return true
}

// recallNext walks forward, restoring the parked draft past the newest entry.
func (m *chatModel) recallNext() bool {
	// Only recall from the last line; elsewhere Down means "move down a line".
	if m.textarea.Line() != m.textarea.LineCount()-1 || m.historyIdx >= len(m.inputHistory) {
		return false
	}
	m.historyIdx++
	if m.historyIdx == len(m.inputHistory) {
		m.setInput(m.historyDraft)
		m.historyDraft = ""
		return true
	}
	m.setInput(m.inputHistory[m.historyIdx])
	return true
}

// setInput replaces the input box contents and parks the cursor at the end,
// which is where you want it when editing a recalled command.
func (m *chatModel) setInput(s string) {
	m.textarea.SetValue(s)
	m.textarea.CursorEnd()
}

// newInflight creates the context for a generation and stores its cancel
// func so Esc can abort. Any previous in-flight context is cancelled first,
// so a stale generation can't outlive the turn that started it.
func (m *chatModel) newInflight() context.Context {
	if m.cancelInflight != nil {
		m.cancelInflight()
	}
	ctx, cancel := context.WithCancel(context.Background())
	m.cancelInflight = cancel
	m.canceled = false
	return ctx
}

// stopInflight aborts a running generation. Reports whether there was
// anything to stop, so Esc falls through to the textarea when idle.
func (m *chatModel) stopInflight() bool {
	if !m.busy || m.cancelInflight == nil {
		return false
	}
	m.canceled = true
	m.cancelInflight()
	if m.streaming {
		// Keep whatever arrived, minus the cursor, so a stopped reply is
		// still readable.
		m.finishStream(m.streamBuf)
	}
	m.cancelInflight = nil
	m.busy = false
	m.busyReason = ""
	// Abandon any queued tool calls from the interrupted turn.
	m.pendingCalls = nil
	m.confirmCall = nil
	if m.picking == "tool_confirm" {
		m.picking = ""
		m.refresh()
	}
	m.dropUnansweredUser()
	m.pushSystem("Stopped. (The partial reply was discarded; the conversation is unchanged.)")
	return true
}

// dropUnansweredUser removes a trailing user message from the conversation
// state after a turn dies without an assistant reply — an inference error or
// an Esc. Leaving it in place corrupts every later turn: the next send puts
// two user messages back-to-back, and strict chat templates (Mistral,
// Ministral, Llama-2) reject non-alternating roles with raise_exception,
// so the server 500s on every message from then on.
func (m *chatModel) dropUnansweredUser() {
	if n := len(m.history); n > 0 && m.history[n-1].Role == "user" {
		m.history = m.history[:n-1]
	}
	if n := len(m.agentMsgs); n > 0 && m.agentMsgs[n-1].Role == "user" {
		m.agentMsgs = m.agentMsgs[:n-1]
	}
}

// inflightCanceled reports whether an error is the result of the user
// pressing Esc, in which case it has already been reported.
func (m *chatModel) inflightCanceled(err error) bool {
	if !m.canceled {
		return false
	}
	if err == nil || errors.Is(err, context.Canceled) ||
		strings.Contains(err.Error(), context.Canceled.Error()) {
		m.canceled = false
		return true
	}
	return false
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

// configState snapshots the session-only state `/config` reports alongside
// the persisted settings.
func (m *chatModel) configState() configState {
	mcpMu.RLock()
	connected := len(mcpConns)
	tools := len(mcpTools)
	mcpMu.RUnlock()

	return configState{
		toolsEnabled: m.agentEnabled,
		yesman:       m.yesman,
		ama:          amaOn.Load(),
		mcpServers:   len(configuredMCPNames()),
		mcpConnected: connected,
		mcpTools:     tools,
	}
}

// displayRoot renders the jail root for user-facing messages, shortened to
// ~ when it sits under the home directory.
func displayRoot() string {
	root := sessionRoot()
	if home, err := os.UserHomeDir(); err == nil {
		if rel, err := filepath.Rel(home, root); err == nil && !strings.HasPrefix(rel, "..") {
			if rel == "." {
				return "~"
			}
			return "~" + string(filepath.Separator) + rel
		}
	}
	return root
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

// runChatStreamCmd streams a reply, forwarding each delta into the
// bubbletea loop. A tea.Cmd can only return one message, so increments go
// through program.Send and the command's own return value is the final
// reply.
func runChatStreamCmd(ctx context.Context, history []ChatMessage, input string) tea.Cmd {
	snapshot := append([]ChatMessage(nil), history...)
	return func() tea.Msg {
		reply, err := chatStream(ctx, snapshot, input, func(d StreamDelta) {
			if program == nil {
				return
			}
			program.Send(assistantDeltaMsg{
				content:   d.Content,
				reasoning: d.Reasoning,
			})
		})
		if err != nil {
			return inferenceErrMsg{err: err}
		}
		return assistantReplyMsg{content: reply}
	}
}

// applyDelta folds one streamed increment into the transcript.
//
// The reply is rendered as plain text while it streams and re-rendered as
// markdown once complete: running glamour on every token would be both slow
// and visually unstable, since a half-written code fence renders as garbage.
func (m *chatModel) applyDelta(msg assistantDeltaMsg) {
	if m.canceled || !m.busy {
		return // belongs to a turn that was stopped
	}
	if !m.streaming {
		m.streaming = true
		m.streamBuf = ""
		m.streamThink = ""
		m.streamThinkShown = false
		cfg, _ := loadConfig()
		m.streamShowThink = cfg.ShowThinking
		m.pushBlank()
		m.rendered = append(m.rendered, assistantPillStyle.Render("ATLAS"))
		m.rendered = append(m.rendered, "")
		m.streamIdx = len(m.rendered) - 1
	}
	if msg.content == "" && msg.reasoning != "" {
		// Still thinking. Show the thinking itself, or at least that
		// something is happening, rather than an empty pill for minutes.
		m.streamThink += msg.reasoning
		if m.streamBuf == "" {
			if m.streamShowThink {
				m.rendered[m.streamIdx] = thinkStyle.Render(m.streamThink) + streamCursor
			} else {
				m.rendered[m.streamIdx] = sysStyle.Render(
					fmt.Sprintf("thinking… (%s of reasoning so far)", formatBytes(int64(len(m.streamThink)))))
			}
			m.refresh()
		}
		return
	}
	m.sealThinkBlock()
	m.streamBuf += msg.content
	m.rendered[m.streamIdx] = m.streamBuf + streamCursor
	m.refresh()
}

// sealThinkBlock freezes the streamed think text into its own transcript
// line so the answer can stream (and later render as markdown) below it
// rather than overwrite it. No-op unless show_thinking is on and unsealed
// think text exists.
func (m *chatModel) sealThinkBlock() {
	if !m.streamShowThink || m.streamThinkShown || strings.TrimSpace(m.streamThink) == "" {
		return
	}
	m.streamThinkShown = true
	m.rendered[m.streamIdx] = thinkStyle.Render(m.streamThink)
	m.rendered = append(m.rendered, "", "")
	m.streamIdx = len(m.rendered) - 1
}

// streamCursor marks the tail of an in-flight reply.
const streamCursor = "▋"

// finishStream replaces the streamed plain text with the markdown render.
// Reports whether a stream was in progress.
func (m *chatModel) finishStream(final string) bool {
	if !m.streaming {
		return false
	}
	m.streaming = false
	if strings.TrimSpace(final) == "" {
		final = m.streamBuf
	}
	// An all-thinking turn (token cap, stop) must keep its think block on
	// screen — that is exactly the turn worth inspecting.
	m.sealThinkBlock()
	if m.streamIdx >= 0 && m.streamIdx < len(m.rendered) {
		m.rendered[m.streamIdx] = m.renderMarkdown(final)
	}
	m.pushRule()
	m.streamBuf = ""
	m.streamThink = ""
	m.streamThinkShown = false
	m.refresh()
	return true
}

// renderRemoteBadge draws the "which machine is answering" indicator. Empty
// when inference is local, because the absence of a badge is the clearest
// possible statement that nothing remote is involved.
//
// Reads only cached state — see remoteStatus. This renders on every
// keystroke, so it must never touch the network.
func renderRemoteBadge() string {
	// Deliberately not remoteEndpoint(): that reads and parses config.json,
	// and this runs on every keystroke. The cached status carries the
	// endpoint precisely so the render path never touches disk.
	st := getRemoteStatus()
	if st.Endpoint == "" {
		return ""
	}

	// Strip the scheme: host:port is what the user typed and what they need
	// to recognise, and header width is scarce.
	label := strings.TrimPrefix(strings.TrimPrefix(st.Endpoint, "http://"), "https://")

	var colour lipgloss.Color
	var mark string
	switch st.State {
	case remoteHealthy:
		colour, mark = colAssistant, "⇅"
	case remoteDegraded:
		colour, mark = colBusy, "⇅"
	case remoteUnreachable:
		colour, mark = colErr, "✗"
	default:
		colour, mark = colMuted, "⇅"
	}
	style := lipgloss.NewStyle().Foreground(colour).Bold(true)
	return style.Render(mark + " REMOTE " + label)
}

// Markdown style detection happens once, before bubbletea takes over stdin.
//
// glamour.WithAutoStyle() asks termenv whether the background is dark, and
// termenv answers by writing an OSC 11 query to the terminal and reading the
// reply back off stdin. Inside a running TUI, bubbletea owns stdin — so the
// terminal's answer ("\x1b]11;rgb:158e/193a/1e75\x1b\\") races between the two
// readers, and when bubbletea wins it lands in the transcript as text.
//
// The renderer is rebuilt whenever the wrap width changes, and markdown is
// rendered when a reply finishes, which is why the escape sequence appeared
// after a response rather than at startup.
var (
	mdStyleOnce sync.Once
	mdStyle     = styles.DarkStyle
)

// detectMarkdownStyle resolves the style while stdin is still ours. Call it
// before starting the bubbletea program.
//
// This mirrors what WithAutoStyle does internally — notty when stdout isn't a
// terminal, otherwise dark or light by background — with the one difference
// that matters: it runs at a moment when reading the terminal's reply is safe.
func detectMarkdownStyle() {
	mdStyleOnce.Do(func() {
		if !term.IsTerminal(int(os.Stdout.Fd())) {
			mdStyle = styles.NoTTYStyle
			return
		}
		if !termenv.HasDarkBackground() {
			mdStyle = styles.LightStyle
		}
		log.Printf("markdown style: %s", mdStyle)
	})
}

// markdownStyleName returns the resolved style.
//
// If detection never ran — a caller that renders markdown outside the TUI —
// it settles the question without a terminal round trip: IsTerminal is a
// local syscall, while the background query is the thing that must never
// happen off the startup path.
func markdownStyleName() string {
	mdStyleOnce.Do(func() {
		if !term.IsTerminal(int(os.Stdout.Fd())) {
			mdStyle = styles.NoTTYStyle
		}
	})
	return mdStyle
}
