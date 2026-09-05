package tui

import (
	"strings"
	"testing"

	"atlas.llm/internal/engine"
	mcpx "atlas.llm/internal/mcp"
	"atlas.llm/internal/tools"
)

func TestParseAMASpec(t *testing.T) {
	cases := []struct {
		name    string
		args    map[string]any
		wantErr bool
		check   func(tools.AmaSpec) bool
	}{
		{
			name: "radio with options",
			args: map[string]any{"kind": "radio", "question": "Pick one",
				"options": []any{"A", "B", "C"}},
			check: func(s tools.AmaSpec) bool {
				return s.Kind == tools.AmaRadio && s.Question == "Pick one" && len(s.Options) == 3
			},
		},
		{
			name: "checkbox",
			args: map[string]any{"kind": "checkbox", "question": "Pick many",
				"options": []any{"X", "Y"}},
			check: func(s tools.AmaSpec) bool { return s.Kind == tools.AmaCheckbox && len(s.Options) == 2 },
		},
		{
			name: "confirm defaults its options",
			args: map[string]any{"kind": "confirm", "question": "Proceed?"},
			check: func(s tools.AmaSpec) bool {
				return s.Kind == tools.AmaConfirm && len(s.Options) == 2 && s.Options[0] == "Yes"
			},
		},
		{
			name: "answer_confirm defaults its options",
			args: map[string]any{"kind": "answer_confirm", "question": "Use this plan?"},
			check: func(s tools.AmaSpec) bool {
				return s.Kind == tools.AmaAnswerConfirm && len(s.Options) == 2
			},
		},
		{
			name:  "kind defaults to radio",
			args:  map[string]any{"question": "Q", "options": []any{"a", "b"}},
			check: func(s tools.AmaSpec) bool { return s.Kind == tools.AmaRadio },
		},
		{
			name: "confirm may override its options",
			args: map[string]any{"kind": "confirm", "question": "Proceed?",
				"options": []any{"Proceed", "Cancel", "Proceed with edits"}},
			check: func(s tools.AmaSpec) bool { return len(s.Options) == 3 },
		},
		{name: "missing question", args: map[string]any{"kind": "radio", "options": []any{"a", "b"}}, wantErr: true},
		{name: "radio needs two options", args: map[string]any{"kind": "radio", "question": "Q", "options": []any{"only"}}, wantErr: true},
		{name: "radio needs options", args: map[string]any{"kind": "radio", "question": "Q"}, wantErr: true},
		{name: "unknown kind", args: map[string]any{"kind": "slider", "question": "Q", "options": []any{"a", "b"}}, wantErr: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			spec, err := tools.ParseAMASpec(tc.args)
			if tc.wantErr {
				if err == nil {
					t.Errorf("expected an error, got %+v", spec)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tc.check != nil && !tc.check(spec) {
				t.Errorf("spec did not match: %+v", spec)
			}
		})
	}
}

func TestFormatAMASelection(t *testing.T) {
	if got := tools.FormatAMASelection(nil); !strings.Contains(got, "nothing") {
		t.Errorf("empty selection = %q, want a 'nothing' phrasing", got)
	}
	if got := tools.FormatAMASelection([]string{"Blue"}); !strings.Contains(got, "Blue") {
		t.Errorf("single = %q, want it to name Blue", got)
	}
	got := tools.FormatAMASelection([]string{"Blue", "Green"})
	if !strings.Contains(got, "Blue") || !strings.Contains(got, "Green") {
		t.Errorf("multi = %q, want both options named", got)
	}
}

// ask_user is advertised to the model only while /ama is on, but is always
// resolvable by name so a stray call can be handled rather than 500'd.
func TestAskUserToolGatedByAMA(t *testing.T) {
	prev := tools.AmaOn.Load()
	defer tools.AmaOn.Store(prev)
	tools.AmaOn.Store(false)
	if toolNamesContain(tools.ActiveTools(), tools.AskUserToolName) {
		t.Error("ask_user advertised while /ama is off")
	}
	tools.AmaOn.Store(true)
	if !toolNamesContain(tools.ActiveTools(), tools.AskUserToolName) {
		t.Error("ask_user not advertised while /ama is on")
	}
	if _, ok := mcpx.LookupTool(tools.AskUserToolName); !ok {
		t.Error("ask_user not resolvable by lookupTool")
	}
}

// Called outside the interactive TUI (a one-shot -c run, say), ask_user can't
// prompt anyone — it must fail loudly rather than hang.
func TestAskUserRunIsNonInteractive(t *testing.T) {
	tool, _ := mcpx.LookupTool(tools.AskUserToolName)
	if _, err := tool.Run(map[string]any{"question": "Q", "options": []any{"a", "b"}}); err == nil {
		t.Error("ask_user.Run should error outside the interactive TUI")
	}
}

func toolNamesContain(tools []tools.Tool, name string) bool {
	for _, t := range tools {
		if t.Name == name {
			return true
		}
	}
	return false
}

// resolveAMA feeds the chosen option(s) back to the agent as the tool result,
// and a dismissal produces a proceed-anyway result rather than stalling.
func TestResolveAMAFeedsAnswerBack(t *testing.T) {
	m := newChatModel()
	call := engine.ToolCall{ID: "c1", Function: engine.ToolCallFunction{Name: tools.AskUserToolName,
		Arguments: `{"kind":"radio","question":"Color?","options":["Blue","Green"]}`}}
	m.amaCall = &call
	m.amaSpec = tools.AmaSpec{Kind: tools.AmaRadio, Question: "Color?", Options: []string{"Blue", "Green"}}
	m.pickerIdx = 1 // Green
	m.picking = "ama"
	m.resolveAMA(true)

	if m.picking != "" {
		t.Error("picker still open after resolve")
	}
	last := m.agentMsgs[len(m.agentMsgs)-1]
	if last.Role != "tool" || last.ToolCallID != "c1" {
		t.Fatalf("last message is not the tool result: %+v", last)
	}
	if !strings.Contains(last.Content, "Green") {
		t.Errorf("answer %q does not carry the choice Green", last.Content)
	}
}

func TestResolveAMACheckbox(t *testing.T) {
	m := newChatModel()
	call := engine.ToolCall{ID: "c2", Function: engine.ToolCallFunction{Name: tools.AskUserToolName}}
	m.amaCall = &call
	m.amaSpec = tools.AmaSpec{Kind: tools.AmaCheckbox, Question: "Which?", Options: []string{"A", "B", "C"}}
	m.amaChecked = []bool{true, false, true}
	m.picking = "ama"
	m.resolveAMA(true)

	last := m.agentMsgs[len(m.agentMsgs)-1]
	if !strings.Contains(last.Content, "A") || !strings.Contains(last.Content, "C") {
		t.Errorf("checkbox answer %q missing A and C", last.Content)
	}
	if strings.Contains(last.Content, `"B"`) {
		t.Errorf("checkbox answer %q wrongly includes unchecked B", last.Content)
	}
}

func TestResolveAMADismiss(t *testing.T) {
	m := newChatModel()
	call := engine.ToolCall{ID: "c3", Function: engine.ToolCallFunction{Name: tools.AskUserToolName}}
	m.amaCall = &call
	m.amaSpec = tools.AmaSpec{Kind: tools.AmaRadio, Question: "Q", Options: []string{"a", "b"}}
	m.picking = "ama"
	m.resolveAMA(false)

	last := m.agentMsgs[len(m.agentMsgs)-1]
	if last.Role != "tool" {
		t.Fatalf("dismiss did not produce a tool result: %+v", last)
	}
	low := strings.ToLower(last.Content)
	if !strings.Contains(low, "dismiss") && !strings.Contains(low, "proceed") {
		t.Errorf("dismiss result %q should tell the model to proceed", last.Content)
	}
}
