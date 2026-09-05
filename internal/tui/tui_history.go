package tui

// Input history — what Up and Down recall.

// maxInputHistory bounds the recall buffer. Generous enough to cover a
// working session, small enough to stay irrelevant to memory.
const maxInputHistory = 200

// recordHistory appends a submitted line to the recall buffer and resets
// the browse position. Consecutive duplicates are collapsed, so holding
// Enter on the same command doesn't bury the rest of the history.
func (m *chatModel) recordHistory(input string) {
	if n := len(m.inputHistory); n == 0 || m.inputHistory[n-1] != input {
		m.inputHistory = append(m.inputHistory, input)
		if len(m.inputHistory) > maxInputHistory {
			m.inputHistory = m.inputHistory[len(m.inputHistory)-maxInputHistory:]
		}
	}
	m.historyIdx = len(m.inputHistory)
	m.historyDraft = ""
}

// recallPrev walks back through submitted input. Reports whether it
// consumed the key — when it returns false the textarea gets the event and
// moves the cursor instead, so multi-line editing still works.
func (m *chatModel) recallPrev() bool {
	// Only recall from the top line; elsewhere Up means "move up a line".
	if m.textarea.Line() != 0 || len(m.inputHistory) == 0 || m.historyIdx == 0 {
		return false
	}
	if m.historyIdx == len(m.inputHistory) {
		m.historyDraft = m.textarea.Value()
	}
	m.historyIdx--
	m.setInput(m.inputHistory[m.historyIdx])
	return true
}

// recallNext walks forward, restoring the parked draft past the newest entry.
func (m *chatModel) recallNext() bool {
	// Only recall from the last line; elsewhere Down means "move down a line".
	if m.textarea.Line() != m.textarea.LineCount()-1 || m.historyIdx >= len(m.inputHistory) {
		return false
	}
	m.historyIdx++
	if m.historyIdx == len(m.inputHistory) {
		m.setInput(m.historyDraft)
		m.historyDraft = ""
		return true
	}
	m.setInput(m.inputHistory[m.historyIdx])
	return true
}

// setInput replaces the input box contents and parks the cursor at the end,
// which is where you want it when editing a recalled command.
func (m *chatModel) setInput(s string) {
	m.textarea.SetValue(s)
	m.textarea.CursorEnd()
}
