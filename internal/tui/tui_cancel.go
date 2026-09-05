package tui

import (
	"context"
	"errors"
	"strings"
)

// Cancellation: the in-flight context behind Ctrl+C / Esc.

// newInflight creates the context for a generation and stores its cancel
// func so Esc can abort. Any previous in-flight context is cancelled first,
// so a stale generation can't outlive the turn that started it.
func (m *chatModel) newInflight() context.Context {
	if m.cancelInflight != nil {
		m.cancelInflight()
	}
	ctx, cancel := context.WithCancel(context.Background())
	m.cancelInflight = cancel
	m.canceled = false
	return ctx
}

// stopInflight aborts a running generation. Reports whether there was
// anything to stop, so Esc falls through to the textarea when idle.
func (m *chatModel) stopInflight() bool {
	if !m.busy || m.cancelInflight == nil {
		return false
	}
	m.canceled = true
	m.cancelInflight()
	if m.streaming {
		// Keep whatever arrived, minus the cursor, so a stopped reply is
		// still readable.
		m.finishStream(m.streamBuf)
	}
	m.cancelInflight = nil
	m.busy = false
	m.busyReason = ""
	// Abandon any queued tool calls from the interrupted turn.
	m.pendingCalls = nil
	m.confirmCall = nil
	if m.picking == "tool_confirm" {
		m.picking = ""
		m.refresh()
	}
	m.dropUnansweredUser()
	m.pushSystem("Stopped. (The partial reply was discarded; the conversation is unchanged.)")
	return true
}

// dropUnansweredUser removes a trailing user message from the conversation
// state after a turn dies without an assistant reply — an inference error or
// an Esc. Leaving it in place corrupts every later turn: the next send puts
// two user messages back-to-back, and strict chat templates (Mistral,
// Ministral, Llama-2) reject non-alternating roles with raise_exception,
// so the server 500s on every message from then on.
func (m *chatModel) dropUnansweredUser() {
	if n := len(m.history); n > 0 && m.history[n-1].Role == "user" {
		m.history = m.history[:n-1]
	}
	if n := len(m.agentMsgs); n > 0 && m.agentMsgs[n-1].Role == "user" {
		m.agentMsgs = m.agentMsgs[:n-1]
	}
}

// inflightCanceled reports whether an error is the result of the user
// pressing Esc, in which case it has already been reported.
func (m *chatModel) inflightCanceled(err error) bool {
	if !m.canceled {
		return false
	}
	if err == nil || errors.Is(err, context.Canceled) ||
		strings.Contains(err.Error(), context.Canceled.Error()) {
		m.canceled = false
		return true
	}
	return false
}
