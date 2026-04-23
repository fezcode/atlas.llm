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

var (
	userStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("12")).Bold(true)
	assistantStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("10")).Bold(true)
	sysStyle       = lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Italic(true)
	errStyle       = lipgloss.NewStyle().Foreground(lipgloss.Color("9"))
	headerStyle    = lipgloss.NewStyle().
			Foreground(lipgloss.Color("15")).
			Background(lipgloss.Color("57")).
			Bold(true).
			Padding(0, 1)
	footerStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("8"))
	borderStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("63"))
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
	modelName  string
	cwd        string

	// Active download state (only set while busyReason == "downloading")
	dlName    string
	dlWritten int64
	dlTotal   int64
}

func newChatModel() chatModel {
	ta := textarea.New()
	ta.Placeholder = "Type a message or /help ..."
	ta.Focus()
	ta.Prompt = "│ "
	ta.CharLimit = 4000
	ta.SetHeight(3)
	ta.ShowLineNumbers = false

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
	return sysStyle.Render(strings.Join([]string{
		"Welcome to atlas.llm chat.",
		"",
		"Slash commands:",
		"  /help          Show this help",
		"  /list          List available models (downloaded status shown)",
		"  /model [name]  Show or switch current model (does NOT download)",
		"  /download      Download engine + current model",
		"                 /download engine        – engine only",
		"                 /download <model-name>  – engine + that model",
		"                 /download all           – engine + every registered model",
		"  /summarize     Summarize current directory to SUMMARY.md",
		"  /grep <query>  Semantic grep across current directory",
		"  /clear         Clear chat history",
		"  /quit, /exit   Leave chat",
		"",
		"Press Ctrl+C to quit. Enter to send; Shift+Enter for newline.",
		"Dependencies are never downloaded automatically — use /download.",
	}, "\n"))
}

func (m chatModel) Init() tea.Cmd {
	return tea.Batch(textarea.Blink, m.spinner.Tick)
}

func (m *chatModel) refresh() {
	m.viewport.SetContent(strings.Join(m.rendered, "\n"))
	m.viewport.GotoBottom()
}

func (m *chatModel) pushSystem(s string) {
	m.rendered = append(m.rendered, sysStyle.Render(s))
	m.refresh()
}

func (m *chatModel) pushUser(s string) {
	m.rendered = append(m.rendered, userStyle.Render("you ")+"› "+s)
	m.refresh()
}

func (m *chatModel) pushAssistant(s string) {
	m.rendered = append(m.rendered, assistantStyle.Render("atlas ")+"› "+s)
	m.refresh()
}

func (m *chatModel) pushError(s string) {
	m.rendered = append(m.rendered, errStyle.Render("error: "+s))
	m.refresh()
}

func (m chatModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		headerH := 1
		footerH := 1
		taH := 5
		vpH := msg.Height - headerH - footerH - taH - 2
		if vpH < 3 {
			vpH = 3
		}
		m.viewport.Width = msg.Width - 2
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
		m.pushSystem(strings.TrimPrefix(welcomeText(), sysStyle.Render("")))
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
		m.dlName = summary[0]
		m.dlWritten = 0
		m.dlTotal = 0
		_ = m.progress.SetPercent(0)
		m.pushSystem(fmt.Sprintf("Downloading: %s", strings.Join(summary, ", ")))
		return tea.Batch(runDownloadAllCmd(targets), m.spinner.Tick)

	case "/summarize":
		m.busy = true
		m.busyReason = "summarizing"
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
		m.pushSystem(fmt.Sprintf("Searching current directory for: %s", query))
		return tea.Batch(runGrepCmd(".", query), m.spinner.Tick)

	default:
		m.pushError(fmt.Sprintf("unknown command: %s", cmd))
		return nil
	}
}

func (m chatModel) View() string {
	dirMax := m.width - len(m.modelName) - 30
	if dirMax < 10 {
		dirMax = 10
	}
	header := headerStyle.Render(fmt.Sprintf(
		" atlas.llm  ·  model: %s  ·  dir: %s ",
		m.modelName, truncateLeft(m.cwd, dirMax),
	))

	body := borderStyle.Render(m.viewport.View())
	input := borderStyle.Render(m.textarea.View())

	var footer string
	switch {
	case m.busy && m.busyReason == "downloading":
		var bar string
		if m.dlTotal > 0 {
			bar = m.progress.View()
			footer = footerStyle.Render(fmt.Sprintf(
				"%s  %s  %s / %s",
				m.dlName, bar, formatBytes(m.dlWritten), formatBytes(m.dlTotal),
			))
		} else {
			footer = footerStyle.Render(fmt.Sprintf(
				"%s %s  %s",
				m.spinner.View(), m.dlName, formatBytes(m.dlWritten),
			))
		}
	case m.busy:
		footer = footerStyle.Render(fmt.Sprintf("%s %s ...", m.spinner.View(), m.busyReason))
	default:
		footer = footerStyle.Render("enter: send  ·  /help  ·  ctrl+c: quit")
	}

	return lipgloss.JoinVertical(lipgloss.Left, header, body, input, footer)
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
