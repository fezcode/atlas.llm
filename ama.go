package main

import (
	"fmt"
	"strings"
	"sync/atomic"
)

// This file is /ama ("ask me anything"): a mode that lets the agent turn a
// decision back to the user through an interactive picker instead of guessing.
// The model calls one tool, ask_user; the TUI renders it as a checkbox,
// radio, or confirmation list; the choice is fed back as the tool result.
//
// ask_user is deliberately NOT in toolRegistry: it is a mode-gated pseudo-tool
// advertised only while /ama is on (so the built-in count is unchanged), it is
// intercepted in the agent loop rather than run like a normal tool, and its
// answer comes from the user, not code.

// amaOn is the /ama toggle. Read by activeTools (whether to advertise
// ask_user) and by the agent loop (whether to honour a call). An atomic so
// the config-time reader and the UI-time writer don't need the model lock.
var amaOn atomic.Bool

const askUserToolName = "ask_user"

// amaKind is the shape of question the model is asking.
type amaKind string

const (
	amaRadio         amaKind = "radio"          // pick exactly one
	amaCheckbox      amaKind = "checkbox"       // pick any number
	amaConfirm       amaKind = "confirm"        // yes/no before an action
	amaAnswerConfirm amaKind = "answer_confirm" // approve a drafted answer
)

// amaSpec is a parsed, validated ask_user call.
type amaSpec struct {
	Kind     amaKind
	Question string
	Options  []string
}

// multiSelect reports whether more than one option may be chosen.
func (s amaSpec) multiSelect() bool { return s.Kind == amaCheckbox }

// askUserTool is the tool definition advertised to the model while /ama is on.
var askUserTool = Tool{
	Name: askUserToolName,
	Description: "Ask the user a question and wait for their answer, shown as an interactive list they " +
		"pick from. Use this when a choice is genuinely the user's to make — an ambiguous request, a " +
		"branch you can't decide from the code, or confirming a plan before you act — rather than guessing. " +
		"Do not use it for things you can determine yourself. kind selects the widget:\n" +
		"- radio: the user picks exactly one of `options`.\n" +
		"- checkbox: the user picks any number of `options` (zero or more).\n" +
		"- confirm: a yes/no step confirmation before doing something; options default to Yes/No.\n" +
		"- answer_confirm: present a drafted answer or plan as the question and let the user approve it or " +
		"ask for changes; options default to Approve/Request changes.\n" +
		"The result is the option text the user chose. If they dismiss it, proceed with your best judgment.",
	Parameters: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"kind": map[string]any{
				"type":        "string",
				"enum":        []string{"radio", "checkbox", "confirm", "answer_confirm"},
				"description": "The kind of question. Defaults to radio.",
			},
			"question": map[string]any{
				"type":        "string",
				"description": "The question to show the user. For answer_confirm, put the drafted answer or plan here.",
			},
			"options": map[string]any{
				"type":        "array",
				"items":       map[string]any{"type": "string"},
				"description": "The choices to offer. Required for radio and checkbox (at least two). Optional for confirm and answer_confirm, which default to Yes/No and Approve/Request changes.",
			},
		},
		"required": []string{"question"},
	},
	Run: toolAskUser,
}

// toolAskUser is the fallback Run for contexts without an interactive TUI (a
// one-shot -c run, tests). It can't prompt anyone, so it fails loudly rather
// than hanging. In the TUI the call is intercepted before Run is ever reached.
func toolAskUser(map[string]any) (string, error) {
	return "", fmt.Errorf("ask_user needs the interactive session and can't be used here — answer without it")
}

// parseAMASpec validates a decoded ask_user argument map into an amaSpec.
func parseAMASpec(args map[string]any) (amaSpec, error) {
	kindStr, _ := args["kind"].(string)
	kindStr = strings.ToLower(strings.TrimSpace(kindStr))
	if kindStr == "" {
		kindStr = string(amaRadio)
	}
	kind := amaKind(kindStr)
	switch kind {
	case amaRadio, amaCheckbox, amaConfirm, amaAnswerConfirm:
	default:
		return amaSpec{}, fmt.Errorf("unknown kind %q (expected radio, checkbox, confirm, or answer_confirm)", kindStr)
	}

	question, _ := args["question"].(string)
	question = strings.TrimSpace(question)
	if question == "" {
		return amaSpec{}, fmt.Errorf("question is required")
	}

	opts := decodeStringSlice(args["options"])
	switch kind {
	case amaConfirm:
		if len(opts) == 0 {
			opts = []string{"Yes", "No"}
		}
	case amaAnswerConfirm:
		if len(opts) == 0 {
			opts = []string{"Approve", "Request changes"}
		}
	default: // radio, checkbox
		if len(opts) < 2 {
			return amaSpec{}, fmt.Errorf("%s needs at least two options", kind)
		}
	}
	return amaSpec{Kind: kind, Question: question, Options: opts}, nil
}

// decodeStringSlice turns a JSON-decoded array (or single string) into a
// trimmed, non-empty string slice.
func decodeStringSlice(v any) []string {
	var out []string
	switch t := v.(type) {
	case []any:
		for _, e := range t {
			if s, ok := e.(string); ok {
				if s = strings.TrimSpace(s); s != "" {
					out = append(out, s)
				}
			}
		}
	case []string:
		for _, s := range t {
			if s = strings.TrimSpace(s); s != "" {
				out = append(out, s)
			}
		}
	case string:
		if s := strings.TrimSpace(t); s != "" {
			out = append(out, s)
		}
	}
	return out
}

// formatAMASelection renders the user's choice(s) as the tool result string
// fed back to the model. Quoted so multi-word options read unambiguously.
func formatAMASelection(chosen []string) string {
	if len(chosen) == 0 {
		return "The user selected nothing."
	}
	quoted := make([]string, len(chosen))
	for i, c := range chosen {
		quoted[i] = fmt.Sprintf("%q", c)
	}
	return "The user selected: " + strings.Join(quoted, ", ")
}
