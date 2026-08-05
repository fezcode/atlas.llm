package main

import (
	"os"
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
)

func newYesmanModel() *chatModel {
	return &chatModel{textarea: textarea.New(), viewport: viewport.New(100, 20)}
}

func TestYesmanToggles(t *testing.T) {
	m := newYesmanModel()
	if m.yesman {
		t.Fatal("yesman must default to off")
	}
	m.handleYesman(nil)
	if !m.yesman {
		t.Error("bare /yesman did not turn it on")
	}
	m.handleYesman(nil)
	if m.yesman {
		t.Error("bare /yesman did not turn it back off")
	}
	m.handleYesman([]string{"on"})
	if !m.yesman {
		t.Error("/yesman on failed")
	}
	m.handleYesman([]string{"off"})
	if m.yesman {
		t.Error("/yesman off failed")
	}
	m.handleYesman([]string{"banana"})
	if m.yesman {
		t.Error("an invalid argument changed the setting")
	}
}

// The whole point of the feature being session-only: it must never reach
// config.json, or a toggle flipped once would arm every future run.
func TestYesmanIsNeverPersisted(t *testing.T) {
	withTempHome(t)
	m := newYesmanModel()
	m.handleYesman([]string{"on"})

	cfg, err := loadConfig()
	if err != nil {
		t.Fatal(err)
	}
	// Config has no field for it at all — assert the shape can't grow one
	// silently by checking the marshalled form.
	if got := configContains(t, "yesman"); got {
		t.Error("yesman leaked into config.json")
	}
	_ = cfg
}

func configContains(t *testing.T, key string) bool {
	t.Helper()
	p, err := configPath()
	if err != nil {
		return false
	}
	data, err := os.ReadFile(p)
	if err != nil {
		return false // no config written at all
	}
	return strings.Contains(strings.ToLower(string(data)), key)
}

// Destructive tools must bypass the modal only while yesman is on, and
// safe tools must be unaffected either way.
func TestYesmanBypassesConfirmForDestructiveOnly(t *testing.T) {
	m := newYesmanModel()
	m.agentMsgs = []ChatMsg{{Role: "system", Content: "x"}}

	// Safe tool: never prompts, regardless.
	m.pendingCalls = []ToolCall{{ID: "1", Function: ToolCallFunction{Name: "list_dir", Arguments: `{"path":"."}`}}}
	m.dispatchNextTool()
	if m.picking == "tool_confirm" {
		t.Error("a read-only tool opened the confirm modal")
	}

	// Destructive with yesman off: must prompt.
	m.picking = ""
	m.pendingCalls = []ToolCall{{ID: "2", Function: ToolCallFunction{Name: "run_cmd", Arguments: `{"command":"echo hi"}`}}}
	m.dispatchNextTool()
	if m.picking != "tool_confirm" {
		t.Error("run_cmd did not prompt with yesman off")
	}

	// Destructive with yesman on: must not prompt.
	m.picking = ""
	m.confirmCall = nil
	m.yesman = true
	m.pendingCalls = []ToolCall{{ID: "3", Function: ToolCallFunction{Name: "run_cmd", Arguments: `{"command":"echo hi"}`}}}
	m.dispatchNextTool()
	if m.picking == "tool_confirm" {
		t.Error("run_cmd prompted despite yesman being on")
	}
}

// An auto-approved destructive call must still be visible in the trace.
func TestYesmanCallsAreStillAnnounced(t *testing.T) {
	m := newYesmanModel()
	m.yesman = true
	m.agentMsgs = []ChatMsg{{Role: "system", Content: "x"}}
	m.pendingCalls = []ToolCall{{ID: "1", Function: ToolCallFunction{Name: "write_file", Arguments: `{"path":"x","content":"y"}`}}}
	m.dispatchNextTool()

	joined := strings.Join(m.rendered, "\n")
	if !strings.Contains(joined, "auto-approved") {
		t.Errorf("auto-approved call was not announced in the trace:\n%s", joined)
	}
}

func TestYesmanShowsInHeaderAndFooter(t *testing.T) {
	m := newYesmanModel()
	if strings.Contains(m.renderHeader(120), "yesman") {
		t.Error("header shows a yesman marker while it is off")
	}
	m.yesman = true
	if !strings.Contains(m.renderHeader(120), "yesman") {
		t.Error("header lacks a yesman marker while it is on")
	}
	if !strings.Contains(m.renderFooter(120), "yesman") {
		t.Error("footer lacks a yesman marker while it is on")
	}
}
