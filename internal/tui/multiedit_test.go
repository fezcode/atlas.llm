package tui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	mcpx "atlas.llm/internal/mcp"
	"atlas.llm/internal/tools"
)

// writeTemp creates a file inside a jailed root, since tools now refuse
// paths outside the directory atlas.llm was started in.
func writeTemp(t *testing.T, body string) string {
	t.Helper()
	root := t.TempDir()
	if resolved, err := filepath.EvalSymlinks(root); err == nil {
		root = resolved
	}
	t.Cleanup(tools.SetSessionRootForTest(root))
	p := filepath.Join(root, "f.go")
	if err := os.WriteFile(p, []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
	return p
}

func edits(pairs ...string) []any {
	var out []any
	for i := 0; i < len(pairs); i += 2 {
		out = append(out, map[string]any{"old_string": pairs[i], "new_string": pairs[i+1]})
	}
	return out
}

func TestMultiEditAppliesAllInOrder(t *testing.T) {
	p := writeTemp(t, "alpha\nbeta\ngamma\n")
	out, err := tools.ToolMultiEdit(map[string]any{
		"path":  p,
		"edits": edits("alpha", "ALPHA", "gamma", "GAMMA"),
	})
	if err != nil {
		t.Fatalf("multi_edit: %v", err)
	}
	t.Log(out)
	got, _ := os.ReadFile(p)
	if string(got) != "ALPHA\nbeta\nGAMMA\n" {
		t.Errorf("got %q", got)
	}
}

// A later edit may legitimately target text an earlier one introduced.
func TestMultiEditSeesEarlierEdits(t *testing.T) {
	p := writeTemp(t, "one\n")
	if _, err := tools.ToolMultiEdit(map[string]any{
		"path":  p,
		"edits": edits("one", "two", "two", "three"),
	}); err != nil {
		t.Fatalf("multi_edit: %v", err)
	}
	got, _ := os.ReadFile(p)
	if string(got) != "three\n" {
		t.Errorf("sequential edits gave %q, want %q", got, "three\n")
	}
}

// The whole point: a failure anywhere leaves the file untouched.
func TestMultiEditIsAtomic(t *testing.T) {
	const original = "alpha\nbeta\n"
	for _, tc := range []struct {
		name  string
		edits []any
		want  string
	}{
		{"missing", edits("alpha", "ALPHA", "nope", "X"), "not found"},
		{"ambiguous", edits("a", "X"), "matches"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := writeTemp(t, original)
			_, err := tools.ToolMultiEdit(map[string]any{"path": p, "edits": tc.edits})
			if err == nil {
				t.Fatal("expected an error")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %q does not mention %q", err, tc.want)
			}
			if !strings.Contains(err.Error(), "No edits were applied") {
				t.Errorf("error should state nothing was applied: %v", err)
			}
			got, _ := os.ReadFile(p)
			if string(got) != original {
				t.Errorf("file was modified despite the failure: %q", got)
			}
		})
	}
}

func TestMultiEditRejectsBadArguments(t *testing.T) {
	p := writeTemp(t, "x\n")
	for _, tc := range []struct {
		name string
		args map[string]any
	}{
		{"no path", map[string]any{"edits": edits("x", "y")}},
		{"no edits", map[string]any{"path": p}},
		{"empty edits", map[string]any{"path": p, "edits": []any{}}},
		{"edits not array", map[string]any{"path": p, "edits": "x"}},
		{"element not object", map[string]any{"path": p, "edits": []any{"x"}}},
		{"empty old_string", map[string]any{"path": p, "edits": edits("", "y")}},
		{"missing new_string", map[string]any{"path": p, "edits": []any{map[string]any{"old_string": "x"}}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := tools.ToolMultiEdit(tc.args); err == nil {
				t.Error("expected an error")
			}
		})
	}
}

// The index makes a failure actionable when a model sends many edits.
func TestMultiEditErrorNamesTheFailingIndex(t *testing.T) {
	p := writeTemp(t, "alpha\nbeta\n")
	_, err := tools.ToolMultiEdit(map[string]any{
		"path":  p,
		"edits": edits("alpha", "A", "missing", "X"),
	})
	if err == nil || !strings.Contains(err.Error(), "edits[1]") {
		t.Errorf("error should identify edits[1]: %v", err)
	}
}

func TestMultiEditNoOpIsReported(t *testing.T) {
	p := writeTemp(t, "same\n")
	out, err := tools.ToolMultiEdit(map[string]any{"path": p, "edits": edits("same", "same")})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "No change") {
		t.Errorf("expected a no-change report, got %q", out)
	}
}

func TestMultiEditIsRegisteredAndDestructive(t *testing.T) {
	tl, ok := mcpx.LookupTool("multi_edit")
	if !ok {
		t.Fatal("multi_edit is not in the registry")
	}
	if !tl.Destructive {
		t.Error("multi_edit must require confirmation")
	}
	props := tl.Parameters["properties"].(map[string]any)
	if _, ok := props["edits"]; !ok {
		t.Error("schema has no edits property")
	}
	req, _ := tl.Parameters["required"].([]string)
	if len(req) != 2 {
		t.Errorf("required = %v, want path and edits", req)
	}
}
