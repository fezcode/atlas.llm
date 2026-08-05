package main

import (
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
)

// Typing must reach the textarea, not the viewport. The viewport's default
// keymap is vim-style, so forwarding keys made h/j/k/l/u/d scroll the
// transcript instead of entering characters.
func TestVimKeysAreTypedNotScrolled(t *testing.T) {
	m := newChatModel()
	m.viewport = viewport.New(80, 10)
	m.viewport.KeyMap = viewport.KeyMap{}
	m.viewport.SetContent(strings.Repeat("line\n", 200))
	m.viewport.GotoTop()
	start := m.viewport.YOffset

	var model tea.Model = m
	for _, r := range "hjkludfb space" {
		// Space is sent as a rune: it is also a viewport PageDown binding,
		// so it belongs in this test.
		model, _ = model.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
	}
	got := model.(chatModel)

	if got.textarea.Value() != "hjkludfb space" {
		t.Errorf("textarea = %q, want %q", got.textarea.Value(), "hjkludfb space")
	}
	if got.viewport.YOffset != start {
		t.Errorf("viewport scrolled while typing: offset %d -> %d", start, got.viewport.YOffset)
	}
}

// The viewport must never carry scroll bindings, or the bug returns the
// moment key forwarding is reinstated.
func TestViewportHasNoKeyBindings(t *testing.T) {
	m := newChatModel()
	for name, b := range map[string]interface{ Keys() []string }{
		"Up": m.viewport.KeyMap.Up, "Down": m.viewport.KeyMap.Down,
		"HalfPageUp": m.viewport.KeyMap.HalfPageUp, "HalfPageDown": m.viewport.KeyMap.HalfPageDown,
		"PageUp": m.viewport.KeyMap.PageUp, "PageDown": m.viewport.KeyMap.PageDown,
	} {
		if keys := b.Keys(); len(keys) != 0 {
			t.Errorf("viewport keymap %s still bound to %v", name, keys)
		}
	}
}

// PgUp/PgDn replace the removed bindings.
func TestPageKeysScroll(t *testing.T) {
	m := newChatModel()
	m.viewport = viewport.New(80, 10)
	m.viewport.KeyMap = viewport.KeyMap{}
	m.viewport.SetContent(strings.Repeat("line\n", 200))
	m.viewport.GotoTop()

	var model tea.Model = m
	model, _ = model.Update(tea.KeyMsg{Type: tea.KeyPgDown})
	down := model.(chatModel).viewport.YOffset
	if down == 0 {
		t.Error("PgDn did not scroll the transcript")
	}
	model, _ = model.Update(tea.KeyMsg{Type: tea.KeyPgUp})
	if up := model.(chatModel).viewport.YOffset; up >= down {
		t.Errorf("PgUp did not scroll back (%d -> %d)", down, up)
	}
	// And typing still isn't scrolling.
	if model.(chatModel).textarea.Value() != "" {
		t.Error("page keys leaked into the textarea")
	}
}

var _ = textarea.New
