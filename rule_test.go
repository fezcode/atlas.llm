package main

import (
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/bubbles/viewport"
)

func ruleModel(width int) *chatModel {
	return &chatModel{viewport: viewport.New(width, 20), busy: true}
}

func TestAssistantReplyIsFollowedByRule(t *testing.T) {
	m := ruleModel(60)
	m.pushAssistant("hello")
	if m.rendered[len(m.rendered)-1] != ruleMarker {
		t.Errorf("no rule after a reply; tail = %q", m.rendered[len(m.rendered)-1])
	}
	out := m.renderTranscript()
	if !strings.Contains(out, strings.Repeat("─", 60)) {
		t.Error("rendered transcript has no full-width rule")
	}
}

func TestStreamedReplyIsFollowedByRule(t *testing.T) {
	m := ruleModel(40)
	m.applyDelta(assistantDeltaMsg{content: "streamed"})
	m.finishStream("streamed")
	if m.rendered[len(m.rendered)-1] != ruleMarker {
		t.Error("no rule after a streamed reply")
	}
}

// The rule is stored as a marker so it re-fits when the terminal resizes;
// a baked-in string would keep the old width.
func TestRuleFollowsTerminalWidth(t *testing.T) {
	m := ruleModel(30)
	m.pushAssistant("x")
	if !strings.Contains(m.renderTranscript(), strings.Repeat("─", 30)) {
		t.Error("rule does not match the initial width")
	}
	m.viewport.Width = 100
	out := m.renderTranscript()
	if !strings.Contains(out, strings.Repeat("─", 100)) {
		t.Error("rule did not widen after a resize")
	}
	if strings.Contains(out, strings.Repeat("─", 101)) {
		t.Error("rule overshot the width")
	}
}

func TestRulesDoNotStack(t *testing.T) {
	m := ruleModel(40)
	m.pushRule()
	m.pushRule()
	m.pushRule()
	n := 0
	for _, l := range m.rendered {
		if l == ruleMarker {
			n++
		}
	}
	if n != 1 {
		t.Errorf("%d consecutive rules, want 1", n)
	}
	// Two separate replies should still get one rule each.
	m2 := ruleModel(40)
	m2.pushAssistant("a")
	m2.pushAssistant("b")
	n = 0
	for _, l := range m2.rendered {
		if l == ruleMarker {
			n++
		}
	}
	if n != 2 {
		t.Errorf("%d rules for two replies, want 2", n)
	}
}

// refresh runs on every streamed token, so transcript rendering must stay
// cheap even with a long history.
func TestTranscriptRenderIsCheap(t *testing.T) {
	m := ruleModel(100)
	for i := 0; i < 2000; i++ {
		m.rendered = append(m.rendered, "a line of transcript text")
		if i%20 == 0 {
			m.rendered = append(m.rendered, ruleMarker)
		}
	}
	start := time.Now()
	for i := 0; i < 200; i++ {
		_ = m.renderTranscript()
	}
	if d := time.Since(start); d > 500*time.Millisecond {
		t.Errorf("200 renders of a 2000-line transcript took %s", d)
	}
}
