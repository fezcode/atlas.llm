package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func jailRoot(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	if resolved, err := filepath.EvalSymlinks(root); err == nil {
		root = resolved
	}
	t.Cleanup(setSessionRootForTest(root))
	return root
}

func TestResolveInRootAllowsRootAndSubdirs(t *testing.T) {
	root := jailRoot(t)
	if err := os.MkdirAll(filepath.Join(root, "a", "b"), 0755); err != nil {
		t.Fatal(err)
	}
	for _, in := range []string{"", ".", "a", "a/b", "./a/b", "a/b/new.txt", root, filepath.Join(root, "a")} {
		got, err := resolveInRoot(in)
		if err != nil {
			t.Errorf("resolveInRoot(%q) errored: %v", in, err)
			continue
		}
		if !strings.HasPrefix(got, root) {
			t.Errorf("resolveInRoot(%q) = %q, outside root %q", in, got, root)
		}
	}
}

// The whole point of the jail: nothing above the root is reachable.
func TestResolveInRootRejectsEscapes(t *testing.T) {
	root := jailRoot(t)
	for _, in := range []string{
		"..", "../..", "../secret", "a/../../secret",
		"/etc/passwd", "/", filepath.Dir(root),
	} {
		if got, err := resolveInRoot(in); err == nil {
			t.Errorf("resolveInRoot(%q) = %q, expected refusal", in, got)
		}
	}
}

// A symlink pointing outside must not be a way around the check.
func TestResolveInRootRejectsSymlinkEscape(t *testing.T) {
	root := jailRoot(t)
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "secret.txt"), []byte("no"), 0644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(root, "escape")
	if err := os.Symlink(outside, link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if got, err := resolveInRoot("escape/secret.txt"); err == nil {
		t.Errorf("symlink escape allowed: %q", got)
	}
	// And the link itself.
	if got, err := resolveInRoot("escape"); err == nil {
		t.Errorf("symlink to an outside dir allowed: %q", got)
	}
}

// Errors must name the root, since "no such file or directory" was exactly
// what sent models into retry loops.
func TestResolveInRootErrorIsInformative(t *testing.T) {
	root := jailRoot(t)
	_, err := resolveInRoot("../nope")
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), root) {
		t.Errorf("error should name the root %q: %v", root, err)
	}
}

func TestToolsAreJailed(t *testing.T) {
	root := jailRoot(t)
	if err := os.WriteFile(filepath.Join(root, "inside.txt"), []byte("hello"), 0644); err != nil {
		t.Fatal(err)
	}
	outside := t.TempDir()
	secret := filepath.Join(outside, "secret.txt")
	if err := os.WriteFile(secret, []byte("classified"), 0644); err != nil {
		t.Fatal(err)
	}

	// Reads inside the root work and are relative to it.
	if out, err := toolReadFile(map[string]any{"path": "inside.txt"}); err != nil || out != "hello" {
		t.Errorf("read inside root: %q, %v", out, err)
	}
	// Every path-taking tool must refuse an outside path.
	cases := []struct {
		name string
		run  func() (string, error)
	}{
		{"read_file", func() (string, error) { return toolReadFile(map[string]any{"path": secret}) }},
		{"list_dir", func() (string, error) { return toolListDir(map[string]any{"path": outside}) }},
		{"grep", func() (string, error) {
			return toolGrep(map[string]any{"pattern": "classified", "path": outside})
		}},
		{"write_file", func() (string, error) {
			return toolWriteFile(map[string]any{"path": filepath.Join(outside, "x.txt"), "content": "x"})
		}},
		{"edit_file", func() (string, error) {
			return toolEditFile(map[string]any{"path": secret, "old_string": "classified", "new_string": "x"})
		}},
		{"multi_edit", func() (string, error) {
			return toolMultiEdit(map[string]any{"path": secret,
				"edits": []any{map[string]any{"old_string": "classified", "new_string": "x"}}})
		}},
		{"run_cmd cwd", func() (string, error) {
			return toolRunCmd(map[string]any{"command": "pwd", "cwd": outside})
		}},
	}
	for _, c := range cases {
		if _, err := c.run(); err == nil {
			t.Errorf("%s reached outside the root", c.name)
		}
	}
	// The outside file must be untouched by the attempted writes.
	if b, _ := os.ReadFile(secret); string(b) != "classified" {
		t.Errorf("outside file was modified: %q", b)
	}
}

// run_cmd runs in the root by default and in a subdir when asked, which is
// what replaces a `cd` that never persisted between calls.
func TestRunCmdUsesJailedWorkingDirectory(t *testing.T) {
	root := jailRoot(t)
	sub := filepath.Join(root, "sub")
	if err := os.MkdirAll(sub, 0755); err != nil {
		t.Fatal(err)
	}
	// `pwd` on Windows resolves to the MSYS one when git-bash is installed,
	// which prints /tmp/... for a path under %TEMP% — a translation of the
	// right directory, but not comparable to it as text. cmd's `cd` prints
	// the native form.
	pwd := "pwd"
	if runtime.GOOS == "windows" {
		pwd = "cd"
	}

	out, err := toolRunCmd(map[string]any{"command": pwd})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, root) {
		t.Errorf("default cwd = %q, want the root %q", out, root)
	}
	out, err = toolRunCmd(map[string]any{"command": pwd, "cwd": "sub"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "sub") {
		t.Errorf("cwd=sub gave %q", out)
	}
	if _, err := toolRunCmd(map[string]any{"command": pwd, "cwd": "nope"}); err == nil {
		t.Error("expected an error for a nonexistent cwd")
	}
}

func TestDisplayPathIsRelative(t *testing.T) {
	root := jailRoot(t)
	if got := displayPath(filepath.Join(root, "a", "b.go")); got != filepath.Join("a", "b.go") {
		t.Errorf("displayPath = %q, want a/b.go", got)
	}
	if got := displayPath(root); got != "." {
		t.Errorf("displayPath(root) = %q, want .", got)
	}
}

// A model repeating the identical call must be stopped early rather than
// left to spend the whole round budget on it.
func TestRepeatedCallDetection(t *testing.T) {
	m := &chatModel{}
	call := ToolCall{Function: ToolCallFunction{Name: "list_dir", Arguments: `{"path":"internal"}`}}
	for i := 1; i <= maxIdenticalCalls; i++ {
		if m.noteRepeatedCall(call) {
			t.Fatalf("flagged as stuck after only %d calls", i)
		}
	}
	if !m.noteRepeatedCall(call) {
		t.Errorf("not flagged after %d identical calls", maxIdenticalCalls+1)
	}
	// Different arguments are a different call.
	other := ToolCall{Function: ToolCallFunction{Name: "list_dir", Arguments: `{"path":"."}`}}
	if m.noteRepeatedCall(other) {
		t.Error("a distinct call was flagged as repeated")
	}
}
