package main

import (
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/viewport"
)

// showThinkModel builds a stream-ready model over a temp home whose config
// has show_thinking set as given, so the tests never depend on the real
// user config.
func showThinkModel(t *testing.T, show bool) *chatModel {
	t.Helper()
	withTempHome(t)
	if err := saveConfig(Config{CurrentModel: defaultModel,
		MaxTokens: defaultMaxTokens, ShowThinking: show}); err != nil {
		t.Fatal(err)
	}
	return &chatModel{viewport: viewport.New(80, 20), busy: true}
}

// /set show_thinking must exist, parse on/off, and reject anything else.
func TestShowThinkingSetting(t *testing.T) {
	s, ok := findSetting("show_thinking")
	if !ok {
		t.Fatal("show_thinking is not a registered setting")
	}
	var cfg Config
	if err := s.Apply(&cfg, "on"); err != nil || !cfg.ShowThinking {
		t.Errorf("Apply(on): err=%v, ShowThinking=%v", err, cfg.ShowThinking)
	}
	if got := s.Value(cfg); got != "on" {
		t.Errorf("Value = %q, want on", got)
	}
	if err := s.Apply(&cfg, "off"); err != nil || cfg.ShowThinking {
		t.Errorf("Apply(off): err=%v, ShowThinking=%v", err, cfg.ShowThinking)
	}
	if got := s.Value(cfg); got != "off" {
		t.Errorf("Value = %q, want off", got)
	}
	if err := s.Apply(&cfg, "sideways"); err == nil {
		t.Error("Apply(sideways) should error")
	}
}

// The flag must survive the piml profile round-trip like any other field.
func TestShowThinkingProfileRoundTrip(t *testing.T) {
	withTempHome(t)
	want := Config{CurrentModel: defaultModel, MaxTokens: 2048, ShowThinking: true}
	if err := saveProfile("unittest-think", want); err != nil {
		t.Fatal(err)
	}
	got, err := loadProfile("unittest-think")
	if err != nil {
		t.Fatal(err)
	}
	if !got.ShowThinking {
		t.Error("ShowThinking lost in the profile round-trip")
	}
}

// With show_thinking on, the think text itself streams into the transcript
// instead of the byte counter — and never leaks into the reply buffer.
func TestThinkTextStreamsWhenEnabled(t *testing.T) {
	m := showThinkModel(t, true)
	m.applyDelta(assistantDeltaMsg{reasoning: "weighing option A"})
	if line := m.rendered[m.streamIdx]; !strings.Contains(line, "weighing option A") {
		t.Errorf("think text not shown: %q", line)
	}
	m.applyDelta(assistantDeltaMsg{reasoning: ", then B"})
	if line := m.rendered[m.streamIdx]; !strings.Contains(line, "weighing option A, then B") {
		t.Errorf("think text not accumulating: %q", line)
	}
	if m.streamBuf != "" {
		t.Errorf("think text leaked into the reply buffer: %q", m.streamBuf)
	}

	// When the answer starts, the think block stays above it.
	m.applyDelta(assistantDeltaMsg{content: "The answer"})
	if line := m.rendered[m.streamIdx]; !strings.Contains(line, "The answer") {
		t.Errorf("answer line = %q", line)
	}
	all := strings.Join(m.rendered, "\n")
	if !strings.Contains(all, "weighing option A, then B") {
		t.Error("think block vanished when the answer started")
	}
}

// The think block must survive the finish, alongside the rendered answer.
func TestThinkSurvivesFinish(t *testing.T) {
	m := showThinkModel(t, true)
	m.applyDelta(assistantDeltaMsg{reasoning: "let me count the ways"})
	m.applyDelta(assistantDeltaMsg{content: "Two."})
	m.finishStream("Two.")
	all := strings.Join(m.rendered, "\n")
	if !strings.Contains(all, "let me count the ways") {
		t.Error("think block lost on finish")
	}
	if !strings.Contains(all, "Two.") {
		t.Error("answer lost on finish")
	}
}

// A turn that was all thinking and no answer (token cap, stop) must keep
// the thinking on screen — that is exactly the case worth inspecting.
func TestThinkKeptWhenReplyEmpty(t *testing.T) {
	m := showThinkModel(t, true)
	m.applyDelta(assistantDeltaMsg{reasoning: "spent the whole budget here"})
	m.finishStream("")
	if all := strings.Join(m.rendered, "\n"); !strings.Contains(all, "spent the whole budget here") {
		t.Error("think block lost when the reply came back empty")
	}
}

// Off (the default) keeps today's behavior: a byte counter, not the text.
func TestThinkHiddenByDefault(t *testing.T) {
	m := showThinkModel(t, false)
	m.applyDelta(assistantDeltaMsg{reasoning: "secret sauce"})
	line := m.rendered[m.streamIdx]
	if strings.Contains(line, "secret sauce") {
		t.Errorf("think text shown while show_thinking is off: %q", line)
	}
	if !strings.Contains(line, "thinking") {
		t.Errorf("no thinking indicator: %q", line)
	}
}
