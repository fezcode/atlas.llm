package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLiftLeadingCd(t *testing.T) {
	tests := []struct{ cmd, cwd, wantCmd, wantCwd, wantLifted string }{
		{"cd src && go build", "", "go build", "src", "src"},
		{"cd src; ls", "", "ls", "src", "src"},
		{"  cd  src/api  &&  ls -la ", "", "ls -la", "src/api", "src/api"},
		{`cd "my dir" && ls`, "", "ls", "my dir", "my dir"},
		// A bare cd has no lasting effect; report where it landed.
		{"cd src", "", "pwd", "src", "src"},
		// Nothing to lift.
		{"go build", "", "go build", "", ""},
		{"ls && cd src", "", "ls && cd src", "", ""},
		// An explicit cwd wins — combining both would be ambiguous.
		{"cd src && ls", "other", "cd src && ls", "other", ""},
	}
	for _, tt := range tests {
		gotCmd, gotCwd, gotLifted := liftLeadingCd(tt.cmd, tt.cwd)
		if gotCmd != tt.wantCmd || gotCwd != tt.wantCwd || gotLifted != tt.wantLifted {
			t.Errorf("liftLeadingCd(%q,%q) = (%q,%q,%q), want (%q,%q,%q)",
				tt.cmd, tt.cwd, gotCmd, gotCwd, gotLifted, tt.wantCmd, tt.wantCwd, tt.wantLifted)
		}
	}
}

// The lift has to actually work end to end, and say what it did.
func TestRunCmdLiftsCdAndExplains(t *testing.T) {
	root := jailRoot(t)
	sub := filepath.Join(root, "sub")
	if err := os.MkdirAll(sub, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(sub, "marker.txt"), []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}
	out, err := toolRunCmd(map[string]any{"command": "cd sub && ls"})
	if err != nil {
		t.Fatalf("run_cmd: %v", err)
	}
	if !strings.Contains(out, "marker.txt") {
		t.Errorf("command did not run in sub: %q", out)
	}
	if !strings.Contains(out, "cwd") {
		t.Errorf("result should explain the cwd argument: %q", out)
	}
	// Escaping via cd is still refused.
	if _, err := toolRunCmd(map[string]any{"command": "cd .. && ls"}); err == nil {
		t.Error("cd .. escaped the jail")
	}
}

// The agent prompt must name the root and show the real layout, so the model
// stops guessing directory names.
func TestAgentPromptCarriesProjectLayout(t *testing.T) {
	root := jailRoot(t)
	for _, d := range []string{"internal", "cmd"} {
		if err := os.MkdirAll(filepath.Join(root, d), 0755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "main.go"), []byte("package main"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, ".git"), 0755); err != nil {
		t.Fatal(err)
	}

	p := agentSystemPromptNow()
	if !strings.Contains(p, root) {
		t.Errorf("prompt does not name the root %q", root)
	}
	for _, want := range []string{"internal/", "cmd/", "main.go"} {
		if !strings.Contains(p, want) {
			t.Errorf("prompt omits %q", want)
		}
	}
	if strings.Contains(p, ".git") {
		t.Error("prompt should not list dotfiles")
	}
	if !strings.Contains(p, "no need to cd") {
		t.Error("prompt should say the model is already at the root")
	}
}

func TestProjectOverviewTruncates(t *testing.T) {
	root := jailRoot(t)
	for i := 0; i < 30; i++ {
		if err := os.WriteFile(filepath.Join(root, string(rune('a'+i%26))+string(rune('0'+i/26))+".txt"),
			[]byte("x"), 0644); err != nil {
			t.Fatal(err)
		}
	}
	out := projectOverview(5)
	if !strings.HasSuffix(out, "…") {
		t.Errorf("expected truncation marker, got %q", out)
	}
	if n := len(strings.Fields(out)); n > 6 {
		t.Errorf("listed %d entries with a cap of 5", n)
	}
}

func TestReasoningSetting(t *testing.T) {
	// auto: off for chat, on where it pays for itself.
	if reasoningEnabledFor(Config{}, false) {
		t.Error("auto should disable reasoning for plain chat")
	}
	if !reasoningEnabledFor(Config{}, true) {
		t.Error("auto should keep reasoning for agentic turns")
	}
	// Explicit settings override both.
	if !reasoningEnabledFor(Config{Reasoning: "on"}, false) {
		t.Error("on should force reasoning even for chat")
	}
	if reasoningEnabledFor(Config{Reasoning: "off"}, true) {
		t.Error("off should suppress reasoning even for agentic turns")
	}
	for _, v := range []string{"off", "OFF", " off "} {
		if reasoningEnabled(Config{Reasoning: v}) {
			t.Errorf("%q should disable reasoning", v)
		}
	}
	if !strings.Contains(reasoningDisplay(Config{Reasoning: "off"}), "off") {
		t.Error("display should report off")
	}
}

func TestReasoningRoundTrips(t *testing.T) {
	withTempHome(t)
	if err := saveConfig(Config{CurrentModel: defaultModel, Reasoning: "off"}); err != nil {
		t.Fatal(err)
	}
	got, err := loadConfig()
	if err != nil {
		t.Fatal(err)
	}
	if got.Reasoning != "off" || reasoningEnabled(got) {
		t.Errorf("reasoning = %q, enabled=%v", got.Reasoning, reasoningEnabled(got))
	}
}
