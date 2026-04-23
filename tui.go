package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
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

	return chatModel{
		viewport:  vp,
		textarea:  ta,
		spinner:   sp,
		progress:  pr,
		modelName: cfg.CurrentModel,
		cwd:       displayCwd(),
	}
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
	kb := func(k, desc string) string {
		return "  " + footerKeyStyle.Render(fmt.Sprintf("%-16s", k)) + sysStyle.Render(desc)
	}
	lines := []string{
		title,
		"",
		sysStyle.Render("Slash commands"),
		kb("/help", "show this help"),
		kb("/list", "available models + download status"),
		kb("/model [name]", "show or switch current model"),
		kb("/download", "engine + current model"),
		kb("/download engine", "engine only"),
		kb("/download <name>", "engine + that model"),
		kb("/download all", "engine + every registered model"),
		kb("/summarize", "write SUMMARY.md for the current directory"),
		kb("/grep <query>", "semantic grep across the current directory"),
		kb("/clear", "clear on-screen chat history"),
		kb("/quit  /exit", "leave chat (or press ctrl+c)"),
		"",
		sysStyle.Render("Dependencies aren't downloaded automatically — start with ") +
			footerKeyStyle.Render("/download") + sysStyle.Render("."),
	}
	return strings.Join(lines, "\n")
}

func (m chatModel) Init() tea.Cmd {
	return tea.Batch(textarea.Blink, m.spinner.Tick)
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
	m.rendered = append(m.rendered, assistantPillStyle.Render("ATLAS")+"  "+s)
	m.refresh()
}

func (m *chatModel) pushError(s string) {
	m.pushBlank()
	m.rendered = append(m.rendered, errPillStyle.Render("ERROR")+"  "+errTextStyle.Render(s))
	m.refresh()
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
		switch msg.Type {
		case tea.KeyCtrlC:
			return m, tea.Quit
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
		m.history = nil
		m.rendered = nil
		m.pushSystem("Chat cleared.")
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
			m.pushSystem(fmt.Sprintf("Current model: %s", m.modelName))
			return nil
		}
		name := args[0]
		target, ok := findModel(name)
		if !ok {
			m.pushError(fmt.Sprintf("unknown model: %s (try /list)", name))
			return nil
		}
		cfg, _ := loadConfig()
		cfg.CurrentModel = target.Name
		if err := saveConfig(cfg); err != nil {
			m.pushError(err.Error())
			return nil
		}
		m.modelName = target.Name
		msg := fmt.Sprintf("Switched model to %s.", target.Name)
		if !isModelDownloaded(target) {
			msg += fmt.Sprintf(" (not downloaded — run /download %s)", target.Name)
		}
		m.pushSystem(msg)
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
		m.busy = true
		m.busyReason = "summarizing"
		m.busyStart = time.Now()
		m.pushSystem("Summarizing current directory → SUMMARY.md ...")
		return tea.Batch(runSummarizeCmd("."), m.spinner.Tick)

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

	dirMax := width - lipgloss.Width(brand) - lipgloss.Width(model) - 30
	if dirMax < 12 {
		dirMax = 12
	}
	dir := metaLabelStyle.Render("cwd ") + metaValueStyle.Render(truncateLeft(m.cwd, dirMax))

	left := brand + dot + model + dot + dir

	var right string
	switch {
	case m.busy && m.busyReason == "downloading":
		right = busyStyle.Render(m.spinner.View() + " downloading")
	case m.busy:
		elapsed := ""
		if !m.busyStart.IsZero() {
			elapsed = fmt.Sprintf(" %ds", int(time.Since(m.busyStart).Seconds()))
		}
		right = busyStyle.Render(m.spinner.View() + " " + m.busyReason + elapsed)
	default:
		right = sysStyle.Render("● ready")
	}

	gap := width - lipgloss.Width(left) - lipgloss.Width(right) - 2
	if gap < 1 {
		gap = 1
	}
	return " " + left + strings.Repeat(" ", gap) + right + " "
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

func runSummarizeCmd(dir string) tea.Cmd {
	return func() tea.Msg {
		out := "SUMMARY.md"
		err := summarizeDirectory(dir, out, nil)
		return summarizeDoneMsg{path: out, err: err}
	}
}

func runGrepCmd(dir, query string) tea.Cmd {
	return func() tea.Msg {
		hits, err := grepDirectory(dir, query, DefaultGrepMaxSize, nil)
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
