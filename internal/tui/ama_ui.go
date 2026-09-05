package tui

import (
	"encoding/json"
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"atlas.llm/internal/config"
	"atlas.llm/internal/engine"
	"atlas.llm/internal/tools"
	"atlas.llm/internal/ui"
)

// startAMA parks an ask_user call and opens the interactive picker. If /ama is
// off, or the arguments are malformed, it feeds an error back to the model and
// moves on rather than stalling the turn.
func (m *chatModel) startAMA(call engine.ToolCall) tea.Cmd {
	if !tools.AmaOn.Load() {
		m.renderToolTrace(call, "(ask_user ignored — /ama is off)", true)
		m.appendToolResult(call, "ask_user is unavailable: /ama is off. Proceed with your best judgment; do not call ask_user again.")
		return m.dispatchNextTool()
	}
	var args map[string]any
	if call.Function.Arguments != "" {
		_ = json.Unmarshal([]byte(call.Function.Arguments), &args)
	}
	spec, err := tools.ParseAMASpec(args)
	if err != nil {
		m.renderToolTrace(call, "(bad ask_user args)", true)
		m.appendToolResult(call, "ask_user: "+err.Error())
		return m.dispatchNextTool()
	}
	m.renderToolTrace(call, "(asking you)", false)
	c := call
	m.amaCall = &c
	m.amaSpec = spec
	m.amaChecked = make([]bool, len(spec.Options))
	m.pickerIdx = 0
	m.picking = "ama"
	m.renderAMA()
	return nil
}

// amaHint is the key legend under the question, tailored to the widget.
func amaHint(s tools.AmaSpec) string {
	if s.MultiSelect() {
		return "↑/↓ move · space toggle · enter submit · esc dismiss"
	}
	return "↑/↓ move · enter select · esc dismiss"
}

// renderAMA paints the ask_user picker over the viewport, mirroring the model
// and confirm modals. A checkbox shows [x]/[ ] boxes; the others show a ▶ on
// the highlighted row.
func (m *chatModel) renderAMA() {
	if m.amaCall == nil {
		return
	}
	spec := m.amaSpec
	title := ui.BrandStyle.Render("The agent is asking") + ui.SysStyle.Render("  ("+amaHint(spec)+")")

	// Wrap the question to the viewport so a long prompt stays readable.
	width := m.viewport.Width
	if width <= 0 {
		width = 80
	}
	qStyle := lipgloss.NewStyle().Bold(true).Width(width - 2)
	lines := []string{title, "", qStyle.Render(spec.Question), ""}

	rowSelected := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#0B1220")).
		Background(ui.ColAccent).
		Bold(true).
		Padding(0, 1)
	rowNormal := lipgloss.NewStyle().Padding(0, 1)

	for i, opt := range spec.Options {
		prefix := "  "
		if spec.MultiSelect() {
			box := "[ ] "
			if i < len(m.amaChecked) && m.amaChecked[i] {
				box = "[x] "
			}
			prefix = box
		} else if i == m.pickerIdx {
			prefix = "▶ "
		}
		row := prefix + opt
		if i == m.pickerIdx {
			lines = append(lines, rowSelected.Render(row))
		} else {
			lines = append(lines, rowNormal.Render(row))
		}
	}
	m.viewport.SetContent(strings.Join(lines, "\n"))
	m.scrollSelectionIntoView(pickerHeaderLines + 2 + m.pickerIdx)
}

// amaMove steps the cursor within the option list, clamped at both ends.
func (m *chatModel) amaMove(delta int) {
	n := len(m.amaSpec.Options)
	if n == 0 {
		return
	}
	m.pickerIdx += delta
	if m.pickerIdx < 0 {
		m.pickerIdx = 0
	}
	if m.pickerIdx >= n {
		m.pickerIdx = n - 1
	}
	m.renderAMA()
}

// amaToggle flips the checkbox at the cursor (no-op for single-select kinds).
func (m *chatModel) amaToggle() {
	if !m.amaSpec.MultiSelect() {
		return
	}
	if m.pickerIdx >= 0 && m.pickerIdx < len(m.amaChecked) {
		m.amaChecked[m.pickerIdx] = !m.amaChecked[m.pickerIdx]
		m.renderAMA()
	}
}

// resolveAMA closes the picker. submit=true feeds the choice back to the
// model as the tool result; submit=false (Esc) feeds a proceed-anyway note so
// the turn continues instead of hanging on an unanswered question.
func (m *chatModel) resolveAMA(submit bool) tea.Cmd {
	if m.amaCall == nil {
		m.picking = ""
		return nil
	}
	call := *m.amaCall
	spec := m.amaSpec
	checked := m.amaChecked
	m.amaCall = nil
	m.amaChecked = nil
	m.picking = ""
	m.refresh()

	if !submit {
		m.renderToolTrace(call, "(dismissed)", true)
		m.appendToolResult(call, "The user dismissed the question without answering. Proceed with your best judgment; do not ask again unless something new makes it necessary.")
		return m.dispatchNextTool()
	}

	var chosen []string
	if spec.MultiSelect() {
		for i, c := range checked {
			if c && i < len(spec.Options) {
				chosen = append(chosen, spec.Options[i])
			}
		}
	} else if m.pickerIdx >= 0 && m.pickerIdx < len(spec.Options) {
		chosen = []string{spec.Options[m.pickerIdx]}
	}

	answer := tools.FormatAMASelection(chosen)
	// Echo the choice into the transcript so the user has a record of what
	// they told the agent, the same way a tool result is shown.
	m.renderToolResult(call, answer, false)
	m.appendToolResult(call, answer)
	return m.dispatchNextTool()
}

// handleAMA implements `/ama`, `/ama on`, and `/ama off`. Persisted like
// /tools so the preference survives a restart.
func (m *chatModel) handleAMA(args []string) {
	cfg, err := config.LoadConfig()
	if err != nil {
		m.pushError("load config: " + err.Error())
		return
	}
	if len(args) == 0 {
		m.pushSystem(fmt.Sprintf("ama = %s  (use `/ama on` to let the agent ask you questions with interactive lists)", onOff(tools.AmaOn.Load())))
		return
	}
	switch strings.ToLower(args[0]) {
	case "on":
		cfg.AMAEnabled = true
		if err := config.SaveConfig(cfg); err != nil {
			m.pushError("save config: " + err.Error())
			return
		}
		tools.AmaOn.Store(true)
		msg := "Ask-me-anything enabled. The agent can now ask you questions with checkboxes, radio lists, and confirmations instead of guessing."
		if !m.agentEnabled {
			msg += "\nIt only takes effect while tools are on — run `/tools on` too."
		}
		m.pushSystem(msg)
	case "off":
		cfg.AMAEnabled = false
		if err := config.SaveConfig(cfg); err != nil {
			m.pushError("save config: " + err.Error())
			return
		}
		tools.AmaOn.Store(false)
		m.pushSystem("Ask-me-anything disabled. The agent will decide on its own instead of asking.")
	default:
		m.pushError(fmt.Sprintf("unknown /ama arg: %s (expected on|off)", args[0]))
	}
}
