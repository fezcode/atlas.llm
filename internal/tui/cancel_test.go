package tui

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"

	"atlas.llm/internal/engine"
)

func newCancelModel() *chatModel {
	return &chatModel{textarea: textarea.New(), viewport: viewport.New(80, 20)}
}

// Esc while idle must fall through to the textarea rather than being eaten.
func TestStopInflightNoopWhenIdle(t *testing.T) {
	m := newCancelModel()
	if m.stopInflight() {
		t.Error("stopInflight consumed Esc while idle")
	}
	m.busy = true // busy but nothing cancellable (e.g. a download)
	if m.stopInflight() {
		t.Error("stopInflight consumed Esc with no in-flight generation")
	}
}

func TestStopInflightCancelsContext(t *testing.T) {
	m := newCancelModel()
	ctx := m.newInflight()
	m.busy = true
	m.busyReason = "thinking"
	m.pendingCalls = []engine.ToolCall{{ID: "call_1"}}

	if !m.stopInflight() {
		t.Fatal("stopInflight did not consume Esc during generation")
	}
	select {
	case <-ctx.Done():
	default:
		t.Error("in-flight context was not cancelled")
	}
	if m.busy || m.busyReason != "" {
		t.Error("busy state not cleared")
	}
	if len(m.pendingCalls) != 0 {
		t.Error("queued tool calls from the stopped turn were not abandoned")
	}
	if !m.canceled {
		t.Error("canceled flag not set")
	}
}

// Starting a new turn must cancel any previous one, so an abandoned
// generation can't outlive the turn that started it.
func TestNewInflightCancelsPrevious(t *testing.T) {
	m := newCancelModel()
	first := m.newInflight()
	second := m.newInflight()
	select {
	case <-first.Done():
	default:
		t.Error("previous in-flight context was not cancelled")
	}
	select {
	case <-second.Done():
		t.Error("new context should still be live")
	default:
	}
}

// The context error from a deliberate stop must not be reported as a failure.
func TestInflightCanceledSwallowsContextError(t *testing.T) {
	m := newCancelModel()
	m.canceled = true
	if !m.inflightCanceled(context.Canceled) {
		t.Error("context.Canceled not recognised as a user stop")
	}
	if m.canceled {
		t.Error("canceled flag should reset after being consumed")
	}
	// Wrapped, as it arrives through the inference layer.
	m.canceled = true
	if !m.inflightCanceled(errors.New("inference failed: context canceled")) {
		t.Error("wrapped cancellation not recognised")
	}
	// A genuine error during a cancelled turn still deserves reporting.
	m.canceled = true
	if m.inflightCanceled(errors.New("connection refused")) {
		t.Error("unrelated error was swallowed as a cancellation")
	}
	// Nothing was cancelled: report normally.
	m.canceled = false
	if m.inflightCanceled(context.Canceled) {
		t.Error("swallowed an error without a user stop")
	}
}

func TestFooterShowsStopWhileBusy(t *testing.T) {
	m := newCancelModel()
	m.busy = true
	m.busyReason = "thinking"
	if got := m.renderFooter(80); !strings.Contains(got, "stop") {
		t.Errorf("busy footer lacks a stop hint: %q", got)
	}
	m.busy = false
	if got := m.renderFooter(80); !strings.Contains(got, "esc") {
		t.Errorf("idle footer lacks the esc hint: %q", got)
	}
}
