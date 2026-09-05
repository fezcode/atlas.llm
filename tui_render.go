package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
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

func (m *chatModel) pushError(s string) {
	m.pushBlank()
	m.rendered = append(m.rendered, errPillStyle.Render("ERROR")+"  "+errTextStyle.Render(s))
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
