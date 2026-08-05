package main

import (
	"strings"
	"testing"
)

func chatModelWithHistory(n int) *chatModel {
	m := &chatModel{}
	for i := 0; i < n; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		m.history = append(m.history, ChatMessage{Role: role, Content: "turn " + string(rune('A'+i%26))})
	}
	return m
}

func TestCompactableTurns(t *testing.T) {
	if got := chatModelWithHistory(0).compactableTurns(); got > 0 {
		t.Errorf("empty history reported %d compactable turns", got)
	}
	if got := chatModelWithHistory(compactKeepTurns).compactableTurns(); got > 0 {
		t.Errorf("history at the keep threshold reported %d compactable", got)
	}
	if got := chatModelWithHistory(compactKeepTurns + 3).compactableTurns(); got != 3 {
		t.Errorf("compactableTurns = %d, want 3", got)
	}
}

// The recent tail must survive verbatim; only older turns get folded.
func TestTranscriptExcludesKeptTail(t *testing.T) {
	m := chatModelWithHistory(compactKeepTurns + 2)
	m.history[0].Content = "OLDEST"
	last := len(m.history) - 1
	m.history[last].Content = "NEWEST"

	transcript, n := m.transcriptForCompaction()
	if n != 2 {
		t.Errorf("folded %d turns, want 2", n)
	}
	if !strings.Contains(transcript, "OLDEST") {
		t.Error("transcript omitted the oldest turn")
	}
	if strings.Contains(transcript, "NEWEST") {
		t.Error("transcript included a turn that should have been kept verbatim")
	}
}

func TestApplyCompactionRebuildsPlainHistory(t *testing.T) {
	m := chatModelWithHistory(compactKeepTurns + 5)
	m.history[len(m.history)-1].Content = "KEEP ME"

	m.applyCompaction(compactDoneMsg{summary: "the gist", dropped: 5})

	if len(m.history) != 2+compactKeepTurns {
		t.Fatalf("history has %d turns, want %d", len(m.history), 2+compactKeepTurns)
	}
	if !strings.Contains(m.history[0].Content, "the gist") {
		t.Errorf("summary not injected: %q", m.history[0].Content)
	}
	if m.history[0].Role != "user" || m.history[1].Role != "assistant" {
		t.Errorf("primer roles are %q/%q, want user/assistant",
			m.history[0].Role, m.history[1].Role)
	}
	if m.history[len(m.history)-1].Content != "KEEP ME" {
		t.Error("the most recent turn was not preserved")
	}
	if m.busy {
		t.Error("busy flag left set after compaction")
	}
}

// Orphaned tool results would reference a tool_call_id the model can no
// longer see, which llama-server rejects.
func TestApplyCompactionDropsOrphanedToolTurns(t *testing.T) {
	m := &chatModel{agentEnabled: true}
	m.agentMsgs = []ChatMsg{{Role: "system", Content: "sys"}}
	for i := 0; i < 6; i++ {
		m.agentMsgs = append(m.agentMsgs, ChatMsg{Role: "user", Content: "q"})
	}
	m.agentMsgs = append(m.agentMsgs,
		ChatMsg{Role: "assistant", ToolCalls: []ToolCall{{ID: "call_1"}}},
		ChatMsg{Role: "tool", ToolCallID: "call_1", Content: "result"},
		ChatMsg{Role: "assistant", Content: "final"},
	)

	m.applyCompaction(compactDoneMsg{summary: "gist", dropped: 6, wasAgent: true})

	if m.agentMsgs[0].Role != "system" {
		t.Error("system prompt lost")
	}
	for i, msg := range m.agentMsgs {
		if msg.Role == "tool" {
			t.Errorf("orphaned tool result survived at %d", i)
		}
		if len(msg.ToolCalls) > 0 {
			t.Errorf("orphaned tool_calls turn survived at %d", i)
		}
	}
}

func TestApplyCompactionRejectsEmptySummary(t *testing.T) {
	m := chatModelWithHistory(10)
	before := len(m.history)
	m.applyCompaction(compactDoneMsg{summary: "   "})
	if len(m.history) != before {
		t.Error("history was rewritten despite an empty summary")
	}
}

func TestApplyCompactionSurfacesError(t *testing.T) {
	m := chatModelWithHistory(10)
	m.busy = true
	before := len(m.history)
	m.applyCompaction(compactDoneMsg{err: errFake{}})
	if len(m.history) != before {
		t.Error("history was rewritten despite a failed summarization")
	}
	if m.busy {
		t.Error("busy flag left set after a failed compaction")
	}
}

type errFake struct{}

func (errFake) Error() string { return "boom" }

// The whole point of the smaller cap: one tool result must not be able to
// swallow the context window.
func TestToolResultCapIsSaneRelativeToContext(t *testing.T) {
	const ctx = 16384
	approxTokens := toolResultSizeLimit / 4
	if pct := approxTokens * 100 / ctx; pct > 15 {
		t.Errorf("one tool result can consume %d%% of a %d-token context "+
			"(%d bytes ≈ %d tokens) — too much", pct, ctx, toolResultSizeLimit, approxTokens)
	}
}
