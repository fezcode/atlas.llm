package main

import (
	"context"
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
)

// Streaming a reply token by token, including the think block.

// runChatStreamCmd streams a reply, forwarding each delta into the
// bubbletea loop. A tea.Cmd can only return one message, so increments go
// through program.Send and the command's own return value is the final
// reply.
func runChatStreamCmd(ctx context.Context, history []ChatMessage, input string) tea.Cmd {
	snapshot := append([]ChatMessage(nil), history...)
	return func() tea.Msg {
		reply, err := chatStream(ctx, snapshot, input, func(d StreamDelta) {
			if program == nil {
				return
			}
			program.Send(assistantDeltaMsg{
				content:   d.Content,
				reasoning: d.Reasoning,
			})
		})
		if err != nil {
			return inferenceErrMsg{err: err}
		}
		return assistantReplyMsg{content: reply}
	}
}

// applyDelta folds one streamed increment into the transcript.
//
// The reply is rendered as plain text while it streams and re-rendered as
// markdown once complete: running glamour on every token would be both slow
// and visually unstable, since a half-written code fence renders as garbage.
func (m *chatModel) applyDelta(msg assistantDeltaMsg) {
	if m.canceled || !m.busy {
		return // belongs to a turn that was stopped
	}
	if !m.streaming {
		m.streaming = true
		m.streamBuf = ""
		m.streamThink = ""
		m.streamThinkShown = false
		cfg, _ := loadConfig()
		m.streamShowThink = cfg.ShowThinking
		m.pushBlank()
		m.rendered = append(m.rendered, assistantPillStyle.Render("ATLAS"))
		m.rendered = append(m.rendered, "")
		m.streamIdx = len(m.rendered) - 1
	}
	if msg.content == "" && msg.reasoning != "" {
		// Still thinking. Show the thinking itself, or at least that
		// something is happening, rather than an empty pill for minutes.
		m.streamThink += msg.reasoning
		if m.streamBuf == "" {
			if m.streamShowThink {
				m.rendered[m.streamIdx] = thinkStyle.Render(m.streamThink) + streamCursor
			} else {
				m.rendered[m.streamIdx] = sysStyle.Render(
					fmt.Sprintf("thinking… (%s of reasoning so far)", formatBytes(int64(len(m.streamThink)))))
			}
			m.refresh()
		}
		return
	}
	m.sealThinkBlock()
	m.streamBuf += msg.content
	m.rendered[m.streamIdx] = m.streamBuf + streamCursor
	m.refresh()
}

// sealThinkBlock freezes the streamed think text into its own transcript
// line so the answer can stream (and later render as markdown) below it
// rather than overwrite it. No-op unless show_thinking is on and unsealed
// think text exists.
func (m *chatModel) sealThinkBlock() {
	if !m.streamShowThink || m.streamThinkShown || strings.TrimSpace(m.streamThink) == "" {
		return
	}
	m.streamThinkShown = true
	m.rendered[m.streamIdx] = thinkStyle.Render(m.streamThink)
	m.rendered = append(m.rendered, "", "")
	m.streamIdx = len(m.rendered) - 1
}

// streamCursor marks the tail of an in-flight reply.
const streamCursor = "▋"

// finishStream replaces the streamed plain text with the markdown render.
// Reports whether a stream was in progress.
func (m *chatModel) finishStream(final string) bool {
	if !m.streaming {
		return false
	}
	m.streaming = false
	if strings.TrimSpace(final) == "" {
		final = m.streamBuf
	}
	// An all-thinking turn (token cap, stop) must keep its think block on
	// screen — that is exactly the turn worth inspecting.
	m.sealThinkBlock()
	if m.streamIdx >= 0 && m.streamIdx < len(m.rendered) {
		m.rendered[m.streamIdx] = m.renderMarkdown(final)
	}
	m.pushRule()
	m.streamBuf = ""
	m.streamThink = ""
	m.streamThinkShown = false
	m.refresh()
	return true
}
