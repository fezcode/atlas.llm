package tui

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"

	"atlas.llm/internal/engine"
	mcpx "atlas.llm/internal/mcp"
	"atlas.llm/internal/tools"
	"atlas.llm/internal/ui"
)

// Painting the screen: the transcript, the header, the input line and the
// footer, plus the push* helpers that append to the transcript.

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
		if line == ui.RuleMarker {
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
	return ui.RuleStyle.Render(strings.Repeat("─", w))
}

// pushRule appends a separator, collapsing consecutive ones.
func (m *chatModel) pushRule() {
	if n := len(m.rendered); n > 0 && m.rendered[n-1] == ui.RuleMarker {
		return
	}
	m.rendered = append(m.rendered, ui.RuleMarker)
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
	return last == ui.RuleMarker || strings.TrimSpace(last) == ""
}

func (m *chatModel) pushBlank() {
	if !m.lastRenderedIsBlank() {
		m.rendered = append(m.rendered, "")
	}
}

func (m *chatModel) pushSystem(s string) {
	m.rendered = append(m.rendered, ui.SysStyle.Render("· "+s))
	m.refresh()
}

func (m *chatModel) pushUser(s string) {
	m.pushBlank()
	m.rendered = append(m.rendered, ui.UserPillStyle.Render("YOU")+"  "+s)
	m.refresh()
}

func (m *chatModel) pushAssistant(s string) {
	m.pushBlank()
	m.rendered = append(m.rendered, ui.AssistantPillStyle.Render("ATLAS"))
	m.rendered = append(m.rendered, m.renderMarkdown(s))
	m.pushRule()
	m.refresh()
}

func (m *chatModel) pushError(s string) {
	m.pushBlank()
	m.rendered = append(m.rendered, ui.ErrPillStyle.Render("ERROR")+"  "+ui.ErrTextStyle.Render(s))
	m.refresh()
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

func (m chatModel) View() string {
	width := m.width
	if width < 1 {
		width = 80
	}

	header := m.renderHeader(width)
	topRule := ui.RuleStyle.Render(strings.Repeat("─", width))
	body := m.viewport.View()
	midRule := ui.RuleStyle.Render(strings.Repeat("─", width))
	input := m.renderInput(width)
	footer := m.renderFooter(width)

	return lipgloss.JoinVertical(lipgloss.Left, header, topRule, body, midRule, input, footer)
}

func (m chatModel) renderHeader(width int) string {
	dot := ui.SepStyle.Render(" • ")
	brand := ui.BrandStyle.Render("◆ atlas.llm")
	model := ui.MetaLabelStyle.Render("model ") + ui.MetaValueStyle.Render(m.modelName)
	remote := renderRemoteBadge()

	ctxSeg := renderCtxSegment()

	var stateSeg string
	switch {
	case m.busy && m.busyReason == "downloading":
		stateSeg = ui.BusyStyle.Render(m.spinner.View() + " downloading")
	case m.busy:
		elapsed := ""
		if !m.busyStart.IsZero() {
			elapsed = fmt.Sprintf(" %ds", int(time.Since(m.busyStart).Seconds()))
		}
		stateSeg = ui.BusyStyle.Render(m.spinner.View() + " " + m.busyReason + elapsed)
	default:
		stateSeg = ui.SysStyle.Render("● ready")
	}

	right := stateSeg
	if ctxSeg != "" {
		right = ctxSeg + dot + stateSeg
	}
	// Memory sits next to the context gauge: both answer "how much room is
	// left", one in tokens and one in bytes.
	if memSeg := engine.RenderMemSegment(); memSeg != "" {
		right = memSeg + dot + right
	}
	if m.yesman {
		// Persistent, loud, and impossible to miss while it's armed.
		right = lipgloss.NewStyle().Foreground(ui.ColErr).Bold(true).Render("⚠ yesman") + dot + right
	}

	dirMax := width - lipgloss.Width(brand) - lipgloss.Width(model) - lipgloss.Width(right) - 12
	if remote != "" {
		dirMax -= lipgloss.Width(remote) + 3
	}
	if dirMax < 12 {
		dirMax = 12
	}
	dir := ui.MetaLabelStyle.Render("cwd ") + ui.MetaValueStyle.Render(truncateLeft(m.cwd, dirMax))

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
	u := engine.GetLastUsage()
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
	label := ui.MetaLabelStyle.Render("ctx ")
	value := ui.MetaValueStyle.Render(fmt.Sprintf("%s/%s", formatTokens(used), formatTokens(u.ContextSize)))
	pctStyle := ui.SysStyle
	switch {
	case pct >= 90:
		pctStyle = lipgloss.NewStyle().Foreground(ui.ColErr).Bold(true)
	case pct >= 70:
		pctStyle = lipgloss.NewStyle().Foreground(ui.ColBusy).Bold(true)
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
	prompt := ui.BrandStyle.Render("❯ ")
	ta := m.textarea.View()
	_ = width
	return prompt + ta
}

func (m chatModel) renderFooter(width int) string {
	_ = width
	if m.busy && m.busyReason == "downloading" {
		var stats string
		if s := m.dlMeter.Speed(); s > 0 {
			stats += "  " + engine.FormatSpeed(s)
		}
		if e := m.dlMeter.Elapsed(time.Now()); e > 0 {
			stats += "  " + engine.FormatElapsed(e)
		}
		if m.dlTotal > 0 {
			return ui.FooterStyle.Render(fmt.Sprintf(
				"  %s  %s  %s / %s%s",
				m.dlName, m.progress.View(), engine.FormatBytes(m.dlWritten), engine.FormatBytes(m.dlTotal), stats,
			))
		}
		return ui.FooterStyle.Render(fmt.Sprintf("  %s  %s%s", m.dlName, engine.FormatBytes(m.dlWritten), stats))
	}

	// While generating, esc is the only thing most people want, so lead
	// with it and drop hints that don't apply mid-turn.
	if m.busy {
		hints := []string{
			ui.FooterKeyStyle.Render("esc") + ui.FooterStyle.Render(" stop generating"),
			ui.FooterKeyStyle.Render("^C") + ui.FooterStyle.Render(" quit"),
		}
		return "  " + strings.Join(hints, ui.SepStyle.Render("  ·  "))
	}

	hints := []string{}
	if m.yesman {
		hints = append(hints, lipgloss.NewStyle().Foreground(ui.ColErr).Bold(true).
			Render("⚠ yesman on")+ui.FooterStyle.Render(" (/yesman off)"))
	}
	hints = append(hints,
		ui.FooterKeyStyle.Render("↵")+ui.FooterStyle.Render(" send"),
		ui.FooterKeyStyle.Render("⇧↵")+ui.FooterStyle.Render(" newline"),
		ui.FooterKeyStyle.Render("↑↓")+ui.FooterStyle.Render(" history"),
		ui.FooterKeyStyle.Render("esc")+ui.FooterStyle.Render(" stop"),
		ui.FooterKeyStyle.Render("^Y")+ui.FooterStyle.Render(" copy reply"),
		ui.FooterKeyStyle.Render("/help")+ui.FooterStyle.Render(" commands"),
		ui.FooterKeyStyle.Render("^C")+ui.FooterStyle.Render(" quit"),
	)
	sep := ui.SepStyle.Render("  ·  ")
	return "  " + strings.Join(hints, sep)
}

// configState snapshots the session-only state `/config` reports alongside
// the persisted settings.
func (m *chatModel) configState() configState {
	mcpx.McpMu.RLock()
	connected := len(mcpx.McpConns)
	mcpToolCount := len(mcpx.McpTools)
	mcpx.McpMu.RUnlock()

	return configState{
		toolsEnabled: m.agentEnabled,
		yesman:       m.yesman,
		ama:          tools.AmaOn.Load(),
		mcpServers:   len(mcpx.ConfiguredMCPNames()),
		mcpConnected: connected,
		mcpTools:     mcpToolCount,
	}
}

// displayRoot renders the jail root for user-facing messages, shortened to
// ~ when it sits under the home directory.
func displayRoot() string {
	root := tools.SessionRoot()
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
	st := engine.GetRemoteStatus()
	if st.Endpoint == "" {
		return ""
	}

	// Strip the scheme: host:port is what the user typed and what they need
	// to recognise, and header width is scarce.
	label := strings.TrimPrefix(strings.TrimPrefix(st.Endpoint, "http://"), "https://")

	var colour lipgloss.Color
	var mark string
	switch st.State {
	case engine.RemoteHealthy:
		colour, mark = ui.ColAssistant, "⇅"
	case engine.RemoteDegraded:
		colour, mark = ui.ColBusy, "⇅"
	case engine.RemoteUnreachable:
		colour, mark = ui.ColErr, "✗"
	default:
		colour, mark = ui.ColMuted, "⇅"
	}
	style := lipgloss.NewStyle().Foreground(colour).Bold(true)
	return style.Render(mark + " REMOTE " + label)
}
