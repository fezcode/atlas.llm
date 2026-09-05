package tui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/viewport"

	"atlas.llm/internal/catalog"
)

// A list taller than the viewport must scroll to follow the cursor.
// Pinning to the top left later entries permanently invisible — you could
// select a model you could not see.
func TestPickerScrollsToSelection(t *testing.T) {
	m := &chatModel{viewport: viewport.New(80, 6)}
	m.pickerItems = append([]catalog.Model(nil), catalog.AvailableModels...)
	if len(m.pickerItems) < 8 {
		t.Skip("registry too small to overflow a 6-line viewport")
	}

	m.pickerIdx = 0
	m.renderPicker()
	if got := m.viewport.YOffset; got != 0 {
		t.Errorf("first item should need no scroll, offset = %d", got)
	}

	last := len(m.pickerItems) - 1
	m.pickerIdx = last
	m.renderPicker()
	line := pickerHeaderLines + last
	off := m.viewport.YOffset
	if line < off || line >= off+m.viewport.Height {
		t.Errorf("last item (line %d) not visible in window [%d,%d)",
			line, off, off+m.viewport.Height)
	}

	// And back to the top.
	m.pickerIdx = 0
	m.renderPicker()
	if m.viewport.YOffset != 0 {
		t.Errorf("returning to the first item should scroll back, offset = %d", m.viewport.YOffset)
	}
}

// A list that fits must not scroll at all.
func TestPickerDoesNotScrollWhenEverythingFits(t *testing.T) {
	m := &chatModel{viewport: viewport.New(80, 100)}
	m.pickerItems = append([]catalog.Model(nil), catalog.AvailableModels...)
	for i := range m.pickerItems {
		m.pickerIdx = i
		m.renderPicker()
		if m.viewport.YOffset != 0 {
			t.Fatalf("scrolled to %d with a viewport that fits everything", m.viewport.YOffset)
		}
	}
}

// Every registry model must actually appear in the rendered picker.
func TestPickerRendersEveryModel(t *testing.T) {
	m := &chatModel{viewport: viewport.New(120, 100)}
	m.pickerItems = append([]catalog.Model(nil), catalog.AvailableModels...)
	m.renderPicker()
	out := m.viewport.View()
	for _, mm := range catalog.AvailableModels {
		if !strings.Contains(out, mm.Name) {
			t.Errorf("picker omits %s", mm.Name)
		}
	}
}
