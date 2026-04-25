package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
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

	sysStyle     = lipgloss.NewStyle().Foreground(colMuted).Italic(true)
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
	sysMsg            struct{ content string }
	summarizeDoneMsg  struct {
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
		name            string
		written, total  int64
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

	// Model picker state. When picking != "", key events route to the
	// picker instead of the textarea; ↑/↓ move, Enter selects, Esc cancels.
	picking     string // "" or "model"
	pickerIdx   int
	pickerItems []Model

	// Markdown renderer for assistant replies. Rebuilt on resize so word
	// wrap tracks the viewport width.
	mdRenderer *glamour.TermRenderer
	mdWidth    int
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

	sp := spinner.New()
	sp.Spinner = spinner.Dot
	sp.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("205"))

	pr := progress.New(progress.WithDefaultGradient())
	pr.Width = 40

	cfg, _ := loadConfig()

	cm := chatModel{
		viewport:  vp,
		textarea:  ta,
		spinner:   sp,
		progress:  pr,
		modelName: cfg.CurrentModel,
		cwd:       displayCwd(),
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
			{"/help", "show this help"},
			{"/clear", "clear the on-screen scrollback (keeps context)"},
			{"/reset", "drop conversation context + server KV cache"},
			{"/set [k [v]]", "show or change settings (max_tokens)"},
			{"/quit  /exit", "leave chat (or press ctrl+c)"},
			{"tab", "complete slash commands and their arguments"},
		}},
	}

	// Pad all command strings to a common width so the description column
	// lines up across every group, regardless of the longest command.
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

func (m *chatModel) refresh() {
	m.viewport.SetContent(strings.Join(m.rendered, "\n"))
	m.viewport.GotoBottom()
}

// lastRenderedIsBlank reports whether the most recent entry is an empty
// separator line, so pushers can avoid stacking blank lines.
func (m *chatModel) lastRenderedIsBlank() bool {
	if len(m.rendered) == 0 {
		return true
	}
	return strings.TrimSpace(m.rendered[len(m.rendered)-1]) == ""
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
			glamour.WithAutoStyle(),
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
		m.pushSystem(fmt.Sprintf("Settings:\n  max_tokens = %d  (reply-length cap; default %d)",
			cfg.MaxTokens, defaultMaxTokens))
		return
	}
	key := strings.ToLower(args[0])
	switch key {
	case "max_tokens":
		if len(args) < 2 {
			m.pushSystem(fmt.Sprintf("max_tokens = %d", cfg.MaxTokens))
			return
		}
		n, err := strconv.Atoi(args[1])
		if err != nil || n <= 0 {
			m.pushError(fmt.Sprintf("invalid max_tokens=%q (expected positive integer)", args[1]))
			return
		}
		// llama-server's context is 16384; a reply that large would leave zero
		// room for prompt + history. Cap at a value that still admits a
		// non-trivial conversation.
		if n > 12000 {
			m.pushError(fmt.Sprintf("max_tokens=%d exceeds safe ceiling of 12000 (16K ctx - headroom for prompt/history)", n))
			return
		}
		cfg.MaxTokens = n
		if err := saveConfig(cfg); err != nil {
			m.pushError("save config: " + err.Error())
			return
		}
		m.pushSystem(fmt.Sprintf("max_tokens = %d", n))
	default:
		m.pushError(fmt.Sprintf("unknown setting: %s (supported: max_tokens)", key))
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
		if i == m.pickerIdx {
			lines = append(lines, rowSelected.Render(row))
		} else {
			lines = append(lines, rowNormal.Render(row))
		}
	}
	m.viewport.SetContent(strings.Join(lines, "\n"))
	m.viewport.GotoTop()
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
		case tea.KeyTab:
			if m.tabComplete() {
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
		case tea.KeyEnter:
			if m.busy {
				break
			}
			input := strings.TrimSpace(m.textarea.Value())
			if input == "" {
				break
			}
			m.textarea.Reset()

			if strings.HasPrefix(input, "/") {
				cmd := m.handleSlash(input)
				if cmd != nil {
					cmds = append(cmds, cmd)
				}
			} else {
				m.pushUser(input)
				m.history = append(m.history, ChatMessage{Role: "user", Content: input})
				m.busy = true
				m.busyReason = "thinking"
				m.busyStart = time.Now()
				hist := append([]ChatMessage(nil), m.history[:len(m.history)-1]...)
				cmds = append(cmds, runChatCmd(hist, input), m.spinner.Tick)
			}
		}

	case assistantReplyMsg:
		m.busy = false
		m.busyReason = ""
		m.history = append(m.history, ChatMessage{Role: "assistant", Content: msg.content})
		m.pushAssistant(msg.content)

	case inferenceErrMsg:
		m.busy = false
		m.busyReason = ""
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
		m.dlName = msg.name
		m.dlWritten = msg.written
		m.dlTotal = msg.total
		if msg.total > 0 {
			pct := float64(msg.written) / float64(msg.total)
			cmds = append(cmds, m.progress.SetPercent(pct))
		}

	case downloadDoneMsg:
		m.busy = false
		m.busyReason = ""
		m.dlName = ""
		m.dlWritten = 0
		m.dlTotal = 0
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
	m.viewport, cmd = m.viewport.Update(msg)
	cmds = append(cmds, cmd)

	return m, tea.Batch(cmds...)
}

func (m *chatModel) handleSlash(input string) tea.Cmd {
	parts := strings.Fields(input)
	cmd := strings.ToLower(parts[0])
	args := parts[1:]

	switch cmd {
	case "/help":
		m.rendered = append(m.rendered, welcomeText())
		m.refresh()
		return nil

	case "/quit", "/exit":
		return tea.Quit

	case "/clear":
		m.rendered = nil
		m.pushSystem("Screen cleared. (Conversation context still active — use /reset to drop it.)")
		return nil

	case "/reset":
		m.history = nil
		m.rendered = nil
		ResetUsage()
		if s, err := ensureServer(); err == nil {
			_ = s.DropKVCache()
		}
		m.pushSystem("Conversation reset. Context and KV cache cleared.")
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
			fmt.Fprintf(&b, " %s %-25s %-8s %s\n", marker, mm.Name, mm.Size, status)
		}
		b.WriteString("\n(* = current model)")
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
	"/clear", "/download", "/exit", "/grep", "/help", "/list",
	"/model", "/quit", "/reset", "/set", "/summarize",
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
		return false
	}
	var pool []string
	switch head {
	case "/model":
		for _, mm := range availableModels {
			pool = append(pool, mm.Name)
		}
	case "/set":
		pool = []string{"max_tokens"}
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
	case "/model", "/download", "/summarize", "/grep", "/set":
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

	dirMax := width - lipgloss.Width(brand) - lipgloss.Width(model) - lipgloss.Width(right) - 12
	if dirMax < 12 {
		dirMax = 12
	}
	dir := metaLabelStyle.Render("cwd ") + metaValueStyle.Render(truncateLeft(m.cwd, dirMax))

	left := brand + dot + model + dot + dir

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
		if m.dlTotal > 0 {
			return footerStyle.Render(fmt.Sprintf(
				"  %s  %s  %s / %s",
				m.dlName, m.progress.View(), formatBytes(m.dlWritten), formatBytes(m.dlTotal),
			))
		}
		return footerStyle.Render(fmt.Sprintf("  %s  %s", m.dlName, formatBytes(m.dlWritten)))
	}

	hints := []string{
		footerKeyStyle.Render("↵") + footerStyle.Render(" send"),
		footerKeyStyle.Render("⇧↵") + footerStyle.Render(" newline"),
		footerKeyStyle.Render("^Y") + footerStyle.Render(" copy reply"),
		footerKeyStyle.Render("/help") + footerStyle.Render(" commands"),
		footerKeyStyle.Render("^C") + footerStyle.Render(" quit"),
	}
	sep := sepStyle.Render("  ·  ")
	return "  " + strings.Join(hints, sep)
}

func runChatCmd(history []ChatMessage, input string) tea.Cmd {
	return func() tea.Msg {
		reply, err := chat(history, input)
		if err != nil {
			return inferenceErrMsg{err: err}
		}
		return assistantReplyMsg{content: reply}
	}
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
//   (none)       -> engine + current model
//   engine       -> engine only
//   all          -> engine + every registered model
//   <model-name> -> engine + that specific model
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
		if t.engine && !isEngineDownloaded() {
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
	defer func() {
		if r := recover(); r != nil {
			logPanicln(r)
			err = fmt.Errorf("panic: %v", r)
		}
	}()

	m := newChatModel()
	if logErr == nil {
		m.rendered = append(m.rendered, sysStyle.Render(fmt.Sprintf("Log file: %s", logFile)))
	}

	p := tea.NewProgram(m, tea.WithAltScreen())
	program = p
	defer func() { program = nil }()
	_, err = p.Run()
	return err
}
