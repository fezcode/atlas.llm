package tui

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

// Enter on an empty prompt used to fall through to the textarea, which
// inserted a newline: the cursor dropped a line and the placeholder tip
// vanished until the stray "\n" was backspaced away. An empty prompt has
// nothing to send, so the key must mean nothing.
func TestEnterOnEmptyPromptDoesNothing(t *testing.T) {
	// busy is part of the test matrix: newChatModel starts busy while the
	// server warms up, which is exactly when an idle Enter was easiest to
	// hit — the empty prompt must swallow the key in both states.
	for _, busy := range []bool{false, true} {
		m := newChatModel()
		m.busy = busy
		var model tea.Model = m
		model, _ = model.Update(tea.KeyMsg{Type: tea.KeyEnter})
		if got := model.(chatModel).textarea.Value(); got != "" {
			t.Errorf("busy=%v: empty-prompt Enter left %q in the textarea, want empty", busy, got)
		}
		// Whitespace-only input is the same non-message.
		m2 := model.(chatModel)
		m2.textarea.SetValue("   ")
		model = m2
		model, _ = model.Update(tea.KeyMsg{Type: tea.KeyEnter})
		if got := model.(chatModel).textarea.Value(); got != "   " {
			t.Errorf("busy=%v: whitespace-prompt Enter changed the textarea to %q", busy, got)
		}
	}
	// Enter on a busy draft is swallowed too — while the model is generating,
	// Enter must not drop the cursor a line. The draft is preserved for the
	// user to send once generation ends.
	m := newChatModel()
	m.busy = true
	m.textarea.SetValue("draft")
	var model tea.Model = m
	model, _ = model.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if got := model.(chatModel).textarea.Value(); got != "draft" {
		t.Errorf("busy draft Enter = %q, want %q (Enter must not add a newline while generating)", got, "draft")
	}
}

// Submitting a prompt must leave the input box empty. The send path resets
// the textarea, but the KeyEnter event then fell through to the shared
// textarea.Update at the bottom of Update, which typed a newline into the
// now-empty box — so after every send the cursor sat on a blank second line.
func TestEnterSubmitsWithoutLeavingNewline(t *testing.T) {
	m := newChatModel()
	m.busy = false
	m.agentEnabled = false // force the plain-chat path, not the agent loop
	m.textarea.SetValue("hello")
	var model tea.Model = m
	model, _ = model.Update(tea.KeyMsg{Type: tea.KeyEnter})
	got := model.(chatModel)
	if v := got.textarea.Value(); v != "" {
		t.Errorf("after submit the textarea holds %q, want empty (a stray newline drops the cursor a line)", v)
	}
	if !got.busy {
		t.Error("submit did not start a turn")
	}
}

// Multi-line prompts: Ctrl+J and Alt+Enter insert a newline instead of
// submitting, so a prompt can span several lines. (Enter alone still sends;
// terminals can't distinguish Shift+Enter, so these are the portable keys.)
func TestNewlineKeysComposeMultiline(t *testing.T) {
	for _, key := range []tea.KeyMsg{
		{Type: tea.KeyCtrlJ},
		{Type: tea.KeyEnter, Alt: true},
	} {
		m := newChatModel()
		m.busy = false
		m.agentEnabled = false
		m.textarea.SetValue("line one")
		var model tea.Model = m
		model, _ = model.Update(key)
		got := model.(chatModel)
		if v := got.textarea.Value(); v != "line one\n" {
			t.Errorf("%v: textarea = %q, want %q", key.Type, v, "line one\n")
		}
		if got.busy {
			t.Errorf("%v: started a turn — it should only add a newline", key.Type)
		}
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
