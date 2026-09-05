package tui

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/atotto/clipboard"
	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"

	"atlas.llm/internal/catalog"
	"atlas.llm/internal/cmdops"
	"atlas.llm/internal/config"
	"atlas.llm/internal/engine"
	"atlas.llm/internal/logging"
	mcpx "atlas.llm/internal/mcp"
	"atlas.llm/internal/tools"
	"atlas.llm/internal/ui"
)

// The Bubble Tea model and its event loop. Everything the TUI does starts
// in Update and fans out to the tui_* files beside this one.

// program is set in startChat so download goroutines can Send progress messages.
var program *tea.Program

type (
	assistantReplyMsg struct{ content string }
	inferenceErrMsg   struct{ err error }
	agentStepMsg      struct {
		content   string
		toolCalls []engine.ToolCall
		err       error
	}
	toolRanMsg struct {
		call   engine.ToolCall
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
		hits  []cmdops.GrepHit
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
	history    []engine.ChatMessage
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
	dlMeter   engine.DownloadMeter

	// Model picker state. When picking != "", key events route to the
	// picker instead of the textarea; ↑/↓ move, Enter selects, Esc cancels.
	picking     string // "", "model", "mcp_add", or "tool_confirm"
	pickerIdx   int
	pickerItems []catalog.Model
	// mcpPickerItems backs the "mcp_add" picker. Kept separate from
	// pickerItems because that one is typed to the model registry.
	mcpPickerItems []mcpx.McpPreset

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
	agentMsgs    []engine.ChatMsg
	pendingCalls []engine.ToolCall
	stepCount    int
	confirmCall  *engine.ToolCall
	confirmIdx   int // 0 = approve, 1 = deny

	// AMA (/ama) state. When the agent calls ask_user, the call is parked in
	// amaCall and rendered as an interactive picker (picking == "ama"); the
	// cursor rides on pickerIdx, and amaChecked tracks the toggles for a
	// checkbox question. Resolving feeds the choice back as the tool result.
	amaCall    *engine.ToolCall
	amaSpec    tools.AmaSpec
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
	ta.FocusedStyle.Placeholder = lipgloss.NewStyle().Foreground(ui.ColDim).Italic(true)
	ta.BlurredStyle.Placeholder = lipgloss.NewStyle().Foreground(ui.ColDim).Italic(true)
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

	cfg, _ := config.LoadConfig()
	// Mirror the persisted /ama preference into the global the tool layer
	// reads, so ask_user is advertised from the first turn if it was left on.
	tools.AmaOn.Store(cfg.AMAEnabled)

	cm := chatModel{
		viewport:     vp,
		textarea:     ta,
		spinner:      sp,
		progress:     pr,
		modelName:    cfg.CurrentModel,
		cwd:          displayCwd(),
		agentEnabled: cfg.ToolsEnabled,
	}
	if m, ok := config.FindModel(cfg.CurrentModel); ok &&
		engine.IsEngineDownloaded() && config.IsModelDownloaded(m) {
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
	title := ui.BrandStyle.Render("◆ atlas.llm") + ui.SysStyle.Render("  local AI chat · on-device inference")

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
	cmdColStyle := lipgloss.NewStyle().Foreground(ui.ColAccent).Bold(true).Width(cmdWidth + 4)
	descStyle := lipgloss.NewStyle().Foreground(ui.ColMuted)
	headingStyle := lipgloss.NewStyle().Foreground(ui.ColDim).Bold(true)

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
		ui.SysStyle.Render("  Dependencies aren't downloaded automatically — start with ")+
			ui.FooterKeyStyle.Render("/download")+ui.SysStyle.Render("."))
	return strings.Join(lines, "\n")
}

func (m chatModel) Init() tea.Cmd {
	return tea.Batch(textarea.Blink, m.spinner.Tick, warmupServerCmd())
}

// warmupServerCmd boots llama-server in a goroutine at chat startup so the
// model is loaded before the user's first message lands.
func warmupServerCmd() tea.Cmd {
	return func() tea.Msg {
		if !engine.IsEngineDownloaded() {
			return serverReadyMsg{} // nothing to warm up yet
		}
		m, err := config.CurrentModel()
		if err != nil {
			return serverReadyMsg{err: err}
		}
		if !config.IsModelDownloaded(m) {
			return serverReadyMsg{} // let /download flow handle it
		}
		s, err := engine.EnsureServer()
		if err != nil {
			return serverReadyMsg{err: err}
		}
		return serverReadyMsg{model: s.Model.Name}
	}
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
					m.agentMsgs = append(m.agentMsgs, engine.ChatMsg{Role: "system", Content: engine.AgentSystemPromptNow()})
				}
				m.agentMsgs = append(m.agentMsgs, engine.ChatMsg{Role: "user", Content: input})
				m.stepCount = 0
				m.repeatedCalls = map[string]int{}
				m.busy = true
				m.busyReason = "thinking"
				m.busyStart = time.Now()
				cmds = append(cmds, runAgentStepCmd(m.newInflight(), m.agentMsgs), m.spinner.Tick)
			} else {
				m.pushUser(input)
				m.history = append(m.history, engine.ChatMessage{Role: "user", Content: input})
				m.busy = true
				m.busyReason = "thinking"
				m.busyStart = time.Now()
				hist := append([]engine.ChatMessage(nil), m.history[:len(m.history)-1]...)
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
		m.history = append(m.history, engine.ChatMessage{Role: "assistant", Content: msg.content})
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
			m.pushSystem(cmdops.FormatGrepHits(msg.hits))
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
		m.dlMeter = engine.DownloadMeter{}
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

func StartChat() (err error) {
	logFile, closeLog, logErr := logging.SetupLogging()
	defer closeLog()
	defer engine.ShutdownServer()
	// Closes stdio MCP subprocesses so they don't outlive the TUI.
	defer mcpx.ShutdownMCP()
	defer func() {
		if r := recover(); r != nil {
			logging.LogPanicln(r)
			err = fmt.Errorf("panic: %v", r)
		}
	}()

	// Must happen before the program starts: this reads a reply off stdin,
	// and once bubbletea owns stdin the reply becomes a keystroke instead.
	detectMarkdownStyle()

	m := newChatModel()
	if logErr == nil {
		m.rendered = append(m.rendered, ui.SysStyle.Render(fmt.Sprintf("Log file: %s", logFile)))
	}

	p := tea.NewProgram(m, tea.WithAltScreen())
	program = p
	defer func() { program = nil }()
	autoConnectMCP()
	// A session that opens against a configured remote should already know
	// whether it is reachable, rather than showing an unknown-state badge
	// until the first heartbeat 15s later.
	if ep, key := config.RemoteEndpoint(); ep != "" {
		// Publish the endpoint synchronously so the badge is present from the
		// first frame; the probe that fills in its state runs in the
		// background, since it can take up to probeTimeout.
		engine.SetRemoteStatus(engine.RemoteStatus{Endpoint: ep, State: engine.RemoteUnknown})
		go func() {
			engine.SetRemoteStatus(engine.ProbeRemote(ep, key))
			engine.StartHeartbeat()
			if program != nil {
				program.Send(remoteStatusMsg{})
			}
		}()
	}
	defer engine.StopHeartbeat()
	_, err = p.Run()
	return err
}

// remoteStatusMsg nudges the TUI to repaint after a background probe changes
// the badge. It carries nothing: the state lives in the cache.
type remoteStatusMsg struct{}
