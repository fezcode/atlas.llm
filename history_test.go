package main

import (
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/textarea"
)

func newHistoryModel() chatModel {
	ta := textarea.New()
	ta.SetWidth(80)
	ta.SetHeight(2)
	ta.Focus()
	return chatModel{textarea: ta}
}

func TestRecordHistoryCollapsesConsecutiveDuplicates(t *testing.T) {
	m := newHistoryModel()
	m.recordHistory("/list")
	m.recordHistory("/list")
	m.recordHistory("hello")
	m.recordHistory("/list")

	want := []string{"/list", "hello", "/list"}
	if len(m.inputHistory) != len(want) {
		t.Fatalf("history = %v, want %v", m.inputHistory, want)
	}
	for i := range want {
		if m.inputHistory[i] != want[i] {
			t.Errorf("history[%d] = %q, want %q", i, m.inputHistory[i], want[i])
		}
	}
	if m.historyIdx != len(m.inputHistory) {
		t.Errorf("historyIdx = %d, want %d (not browsing)", m.historyIdx, len(m.inputHistory))
	}
}

func TestRecallWalksBackAndForward(t *testing.T) {
	m := newHistoryModel()
	for _, s := range []string{"first", "second", "third"} {
		m.recordHistory(s)
	}

	for _, want := range []string{"third", "second", "first"} {
		if !m.recallPrev() {
			t.Fatalf("recallPrev returned false before reaching %q", want)
		}
		if got := m.textarea.Value(); got != want {
			t.Fatalf("recalled %q, want %q", got, want)
		}
	}
	// Past the oldest entry the key must fall through to the textarea.
	if m.recallPrev() {
		t.Error("recallPrev consumed the key at the oldest entry")
	}
	if got := m.textarea.Value(); got != "first" {
		t.Errorf("value changed past the oldest entry: %q", got)
	}

	for _, want := range []string{"second", "third"} {
		if !m.recallNext() {
			t.Fatalf("recallNext returned false before reaching %q", want)
		}
		if got := m.textarea.Value(); got != want {
			t.Fatalf("recalled %q, want %q", got, want)
		}
	}
}

// A half-typed line must survive a trip into the history and back.
func TestRecallRestoresDraft(t *testing.T) {
	m := newHistoryModel()
	m.recordHistory("/help")
	m.setInput("half typed thing")

	if !m.recallPrev() {
		t.Fatal("recallPrev did not consume the key")
	}
	if got := m.textarea.Value(); got != "/help" {
		t.Fatalf("recalled %q, want /help", got)
	}
	if !m.recallNext() {
		t.Fatal("recallNext did not consume the key")
	}
	if got := m.textarea.Value(); got != "half typed thing" {
		t.Errorf("draft not restored, got %q", got)
	}
	// Already at the newest entry — Down belongs to the textarea now.
	if m.recallNext() {
		t.Error("recallNext consumed the key past the newest entry")
	}
}

func TestRecallNoopsOnEmptyHistory(t *testing.T) {
	m := newHistoryModel()
	if m.recallPrev() || m.recallNext() {
		t.Error("recall consumed a key with no history")
	}
}

// Up/Down must still move the cursor inside a multi-line draft; recall only
// takes over at the first and last line.
func TestRecallYieldsToCursorMovementInMultilineInput(t *testing.T) {
	m := newHistoryModel()
	m.recordHistory("earlier")
	m.setInput("line one\nline two\nline three")

	// CursorEnd leaves us on the last line: Up is cursor movement there.
	if m.textarea.Line() != m.textarea.LineCount()-1 {
		t.Fatalf("expected cursor on the last line, got %d of %d",
			m.textarea.Line(), m.textarea.LineCount())
	}
	if m.recallPrev() {
		t.Error("recallPrev hijacked Up from the middle of a multi-line draft")
	}
	// And Down on the last line would recall, but there's nothing newer.
	if m.recallNext() {
		t.Error("recallNext consumed Down with nothing newer to show")
	}
	if got := m.textarea.Value(); !strings.Contains(got, "line two") {
		t.Errorf("multi-line draft was clobbered: %q", got)
	}
}

func TestHistoryIsBounded(t *testing.T) {
	m := newHistoryModel()
	for i := 0; i < maxInputHistory+50; i++ {
		m.recordHistory(strings.Repeat("x", i%7+1) + string(rune('a'+i%26)) + string(rune(i)))
	}
	if len(m.inputHistory) > maxInputHistory {
		t.Errorf("history grew to %d, cap is %d", len(m.inputHistory), maxInputHistory)
	}
	if m.historyIdx != len(m.inputHistory) {
		t.Errorf("historyIdx = %d, want %d after trimming", m.historyIdx, len(m.inputHistory))
	}
}
