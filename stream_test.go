package main

import (
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/viewport"
)

func streamModel() *chatModel {
	return &chatModel{viewport: viewport.New(80, 20), busy: true}
}

// Deltas must accumulate into one line that is rewritten in place, not
// appended one row per token.
func TestStreamDeltasAccumulateInPlace(t *testing.T) {
	m := streamModel()
	for _, part := range []string{"Hello", ", ", "world"} {
		m.applyDelta(assistantDeltaMsg{content: part})
	}
	if m.streamBuf != "Hello, world" {
		t.Errorf("buffer = %q", m.streamBuf)
	}
	// The transcript must not grow per token: the reply occupies one line
	// that is rewritten, however many deltas arrive.
	afterThree := len(m.rendered)
	for _, part := range []string{"!", " again", " and again"} {
		m.applyDelta(assistantDeltaMsg{content: part})
	}
	if got := len(m.rendered); got != afterThree {
		t.Errorf("transcript grew from %d to %d lines over further deltas", afterThree, got)
	}
	if m.streamBuf != "Hello, world! again and again" {
		t.Errorf("buffer = %q", m.streamBuf)
	}
	if !strings.Contains(m.rendered[m.streamIdx], "Hello, world") {
		t.Errorf("stream line = %q", m.rendered[m.streamIdx])
	}
	if !strings.HasSuffix(m.rendered[m.streamIdx], streamCursor) {
		t.Error("in-flight reply should end with the cursor")
	}
}

// A reasoning model can think for minutes before emitting an answer. That
// must show progress rather than an empty pill.
func TestStreamShowsThinkingBeforeAnswer(t *testing.T) {
	m := streamModel()
	m.applyDelta(assistantDeltaMsg{reasoningTokens: 1200})
	line := m.rendered[m.streamIdx]
	if !strings.Contains(line, "thinking") {
		t.Errorf("expected a thinking indicator, got %q", line)
	}
	if m.streamBuf != "" {
		t.Errorf("reasoning must not enter the reply buffer, got %q", m.streamBuf)
	}
	// Once the answer starts, it replaces the indicator.
	m.applyDelta(assistantDeltaMsg{content: "The answer"})
	if line := m.rendered[m.streamIdx]; strings.Contains(line, "thinking") {
		t.Errorf("thinking indicator survived into the answer: %q", line)
	}
}

// Finishing swaps the plain streamed text for the markdown render and drops
// the cursor.
func TestFinishStreamRendersMarkdown(t *testing.T) {
	m := streamModel()
	m.applyDelta(assistantDeltaMsg{content: "# Title\n\nbody"})
	if !m.finishStream("# Title\n\nbody") {
		t.Fatal("finishStream reported no stream in progress")
	}
	line := m.rendered[m.streamIdx]
	if strings.HasSuffix(line, streamCursor) {
		t.Error("cursor survived the finish")
	}
	if m.streaming {
		t.Error("still marked as streaming")
	}
	if m.streamBuf != "" {
		t.Error("buffer not cleared")
	}
	// finishStream on a fresh model must be a no-op, so non-streamed
	// replies still get pushed normally.
	if streamModel().finishStream("x") {
		t.Error("finishStream claimed a stream that never started")
	}
}

// A stopped turn keeps whatever text arrived, without a dangling cursor.
func TestStoppingMidStreamKeepsPartialText(t *testing.T) {
	m := streamModel()
	m.cancelInflight = func() {}
	m.applyDelta(assistantDeltaMsg{content: "partial answer"})
	if !m.stopInflight() {
		t.Fatal("stopInflight did not consume the stop")
	}
	line := m.rendered[m.streamIdx]
	if !strings.Contains(line, "partial answer") {
		t.Errorf("partial text lost: %q", line)
	}
	if strings.HasSuffix(line, streamCursor) {
		t.Error("cursor left behind after stopping")
	}
	if m.streaming {
		t.Error("still streaming after a stop")
	}
}

// Deltas arriving after a stop belong to an abandoned turn.
func TestDeltasIgnoredAfterStop(t *testing.T) {
	m := streamModel()
	m.canceled = true
	m.applyDelta(assistantDeltaMsg{content: "late"})
	if m.streaming || m.streamBuf != "" {
		t.Error("a delta was accepted after the turn was stopped")
	}
	// Same when nothing is in flight at all.
	idle := &chatModel{viewport: viewport.New(80, 20)}
	idle.applyDelta(assistantDeltaMsg{content: "stray"})
	if idle.streaming {
		t.Error("a stray delta started a stream while idle")
	}
}
