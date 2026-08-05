package main

import (
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// compactDoneMsg carries the result of a background summarization.
type compactDoneMsg struct {
	summary  string
	before   int // approximate tokens before compaction
	dropped  int // conversation turns folded into the summary
	err      error
	wasAgent bool
}

// compactKeepTurns is how many of the most recent turns survive verbatim.
// Compaction is most useful when the tail is still exact — the last thing
// you asked is usually what you're still working on.
const compactKeepTurns = 4

// compactSystemPrompt steers the summarizer. It asks for facts and open
// threads rather than a narrative, because the output is re-injected as
// working context for the model, not shown as prose to the user.
const compactSystemPrompt = `You compress a conversation so it can continue in a smaller context window. Produce a dense factual summary covering: what the user is trying to do, decisions and conclusions reached, concrete details worth keeping (file paths, names, values, error messages), and anything still unresolved. Use terse bullet points. Do not add pleasantries, do not editorialize, and do not invent anything that was not present.`

// compactableTurns reports how many turns are eligible to fold away.
func (m *chatModel) compactableTurns() int {
	if m.agentEnabled {
		// The leading system prompt is never eligible.
		n := len(m.agentMsgs) - 1
		if n < 0 {
			n = 0
		}
		return n - compactKeepTurns
	}
	return len(m.history) - compactKeepTurns
}

// transcriptForCompaction renders the turns to be folded away, and returns
// how many were included.
func (m *chatModel) transcriptForCompaction() (string, int) {
	var b strings.Builder
	n := 0
	if m.agentEnabled {
		end := len(m.agentMsgs) - compactKeepTurns
		for i := 1; i < end; i++ { // skip the system prompt at 0
			msg := m.agentMsgs[i]
			content := strings.TrimSpace(msg.Content)
			// A tool_calls turn often has empty content; record the intent.
			if content == "" && len(msg.ToolCalls) > 0 {
				var names []string
				for _, c := range msg.ToolCalls {
					names = append(names, c.Function.Name)
				}
				content = "(called tools: " + strings.Join(names, ", ") + ")"
			}
			if content == "" {
				continue
			}
			fmt.Fprintf(&b, "%s: %s\n\n", msg.Role, content)
			n++
		}
		return b.String(), n
	}
	end := len(m.history) - compactKeepTurns
	for i := 0; i < end; i++ {
		content := strings.TrimSpace(m.history[i].Content)
		if content == "" {
			continue
		}
		fmt.Fprintf(&b, "%s: %s\n\n", m.history[i].Role, content)
		n++
	}
	return b.String(), n
}

// handleCompact implements `/compact`: fold the older part of the
// conversation into a summary so the context window has room again.
func (m *chatModel) handleCompact() tea.Cmd {
	if m.busy {
		m.pushError("busy — wait for the current turn to finish before compacting")
		return nil
	}
	if m.compactableTurns() <= 0 {
		m.pushSystem(fmt.Sprintf(
			"Nothing to compact — the conversation is %d turns and the most recent %d are always kept verbatim.",
			m.conversationLen(), compactKeepTurns))
		return nil
	}
	transcript, count := m.transcriptForCompaction()
	if strings.TrimSpace(transcript) == "" {
		m.pushSystem("Nothing to compact.")
		return nil
	}
	before := GetLastUsage().TotalTokens
	m.busy = true
	m.busyReason = "compacting"
	m.busyStart = time.Now()
	m.pushSystem(fmt.Sprintf("Compacting %d earlier turns into a summary…", count))
	return tea.Batch(runCompactCmd(transcript, count, before, m.agentEnabled), m.spinner.Tick)
}

func (m *chatModel) conversationLen() int {
	if m.agentEnabled {
		n := len(m.agentMsgs) - 1
		if n < 0 {
			return 0
		}
		return n
	}
	return len(m.history)
}

// runCompactCmd summarizes off the UI goroutine.
func runCompactCmd(transcript string, dropped, before int, wasAgent bool) tea.Cmd {
	return func() tea.Msg {
		summary, err := runSingleUser(compactSystemPrompt,
			"Compress this conversation:\n\n"+transcript, 1024)
		return compactDoneMsg{
			summary: summary, before: before, dropped: dropped,
			err: err, wasAgent: wasAgent,
		}
	}
}

// applyCompaction swaps the folded turns for the summary. The summary is
// injected as a user turn plus a short assistant acknowledgement, which
// every chat template handles — unlike a second system message, which some
// models ignore or refuse to blend with the first.
func (m *chatModel) applyCompaction(msg compactDoneMsg) {
	m.busy = false
	m.busyReason = ""
	if msg.err != nil {
		m.pushError("compact failed: " + msg.err.Error())
		return
	}
	summary := strings.TrimSpace(msg.summary)
	if summary == "" {
		m.pushError("compact failed: the model returned an empty summary")
		return
	}

	primer := "Summary of our earlier conversation:\n\n" + summary
	const ack = "Understood — I'll continue from that summary."

	if msg.wasAgent && len(m.agentMsgs) > 0 {
		tail := m.agentMsgs[max(1, len(m.agentMsgs)-compactKeepTurns):]
		rebuilt := []ChatMsg{m.agentMsgs[0]} // keep the system prompt
		rebuilt = append(rebuilt,
			ChatMsg{Role: "user", Content: primer},
			ChatMsg{Role: "assistant", Content: ack})
		// A tool result whose originating call was folded away would
		// reference a tool_call_id the model can no longer see.
		for _, t := range tail {
			if t.Role == "tool" || len(t.ToolCalls) > 0 {
				continue
			}
			rebuilt = append(rebuilt, t)
		}
		m.agentMsgs = rebuilt
	} else {
		tail := m.history[max(0, len(m.history)-compactKeepTurns):]
		rebuilt := []ChatMessage{
			{Role: "user", Content: primer},
			{Role: "assistant", Content: ack},
		}
		m.history = append(rebuilt, tail...)
	}

	// The cached prefix no longer matches the rewritten history.
	if s, err := ensureServer(); err == nil {
		_ = s.DropKVCache()
	}
	ResetUsage()

	m.pushSystem(fmt.Sprintf(
		"Compacted %d turns into a %d-character summary. The last %d turns were kept verbatim.\n"+
			"Context usage resets on your next message.",
		msg.dropped, len(summary), compactKeepTurns))
	m.pushAssistant("**Summary of earlier conversation**\n\n" + summary)
}

// compactSuggestPct is the context-fill level at which atlas.llm suggests
// compacting. Chosen to leave room to actually run the summarization —
// waiting until the window is full means /compact itself may not fit.
const compactSuggestPct = 75

// maybeSuggestCompact nudges once per crossing when the context is filling
// up. It resets when usage drops, so a compaction re-arms the warning
// rather than silencing it for the session.
func (m *chatModel) maybeSuggestCompact() {
	u := GetLastUsage()
	if u.ContextSize == 0 {
		return
	}
	used := u.PromptTokens + u.CompletionTokens
	if u.TotalTokens > used {
		used = u.TotalTokens
	}
	pct := (used * 100) / u.ContextSize
	if pct < compactSuggestPct {
		m.compactSuggested = false
		return
	}
	if m.compactSuggested || m.compactableTurns() <= 0 {
		return
	}
	m.compactSuggested = true
	m.pushSystem(fmt.Sprintf(
		"Context is %d%% full (%d/%d tokens). Run /compact to summarize earlier turns "+
			"and free up room — or /reset to start over.",
		pct, used, u.ContextSize))
}
