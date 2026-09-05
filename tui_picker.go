package main

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// The model picker overlay — the list you get from /model with no argument.

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
