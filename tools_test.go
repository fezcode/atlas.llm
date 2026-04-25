package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestArgString(t *testing.T) {
	args := map[string]any{"path": "main.go", "count": 3}

	got, err := argString(args, "path", true)
	if err != nil || got != "main.go" {
		t.Fatalf("argString(path): got %q err=%v", got, err)
	}

	// Missing + required → error.
	if _, err := argString(args, "missing", true); err == nil {
		t.Fatalf("argString(missing, required=true) should error")
	}

	// Missing + optional → empty, no error.
	got, err = argString(args, "missing", false)
	if err != nil || got != "" {
		t.Fatalf("argString(missing, optional): got %q err=%v", got, err)
	}

	// Wrong type → error.
	if _, err := argString(args, "count", true); err == nil {
		t.Fatalf("argString(count) should error: int is not string")
	}
}

func TestIsLikelyText(t *testing.T) {
	if !isLikelyText([]byte("package main\nfunc main() {}")) {
		t.Fatal("source code should be detected as text")
	}
	if isLikelyText([]byte{0x7f, 'E', 'L', 'F', 0, 0, 0, 0}) {
		t.Fatal("ELF-ish bytes should be detected as binary")
	}
	if !isLikelyText([]byte{}) {
		t.Fatal("empty slice should count as text (no NUL)")
	}
}

func TestTruncateForModel(t *testing.T) {
	small := strings.Repeat("a", 100)
	if got := truncateForModel(small); got != small {
		t.Fatal("short input should pass through unchanged")
	}
	big := strings.Repeat("b", toolResultSizeLimit+500)
	got := truncateForModel(big)
	if !strings.HasPrefix(got, strings.Repeat("b", toolResultSizeLimit)) {
		t.Fatal("truncation should keep the prefix")
	}
	if !strings.Contains(got, "truncated") {
		t.Fatal("truncated output should announce the truncation")
	}
}

func TestToolReadFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "hello.txt")
	want := "hello world\n"
	if err := os.WriteFile(path, []byte(want), 0644); err != nil {
		t.Fatal(err)
	}
	got, err := toolReadFile(map[string]any{"path": path})
	if err != nil || got != want {
		t.Fatalf("read_file: got %q err=%v", got, err)
	}

	if _, err := toolReadFile(map[string]any{"path": filepath.Join(dir, "nope")}); err == nil {
		t.Fatal("read_file on missing path should error")
	}
	if _, err := toolReadFile(map[string]any{}); err == nil {
		t.Fatal("read_file without path should error")
	}
}

func TestToolListDir(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(dir, "sub"), 0755); err != nil {
		t.Fatal(err)
	}
	out, err := toolListDir(map[string]any{"path": dir})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "a.txt") {
		t.Errorf("list_dir missing file entry: %q", out)
	}
	if !strings.Contains(out, "sub/") {
		t.Errorf("list_dir should suffix directories with '/': %q", out)
	}
}

func TestToolGrep(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "a.go"), "package main\nfunc foo() {}\nfunc bar() {}\n")
	mustWrite(t, filepath.Join(dir, "b.go"), "package main\nfunc baz() {}\n")

	out, err := toolGrep(map[string]any{"pattern": "^func ba", "path": dir})
	if err != nil {
		t.Fatal(err)
	}
	// bar and baz both match ^func ba.
	if !strings.Contains(out, "bar()") || !strings.Contains(out, "baz()") {
		t.Errorf("grep missed expected hits: %q", out)
	}
	if strings.Contains(out, "foo()") {
		t.Errorf("grep matched foo() which should not match ^func ba: %q", out)
	}

	// No matches → explicit empty-marker.
	out, err = toolGrep(map[string]any{"pattern": "nothingmatches", "path": dir})
	if err != nil || !strings.Contains(out, "no matches") {
		t.Errorf("grep with no hits should say '(no matches)', got %q err=%v", out, err)
	}

	// Invalid regex → error surfaces to caller.
	if _, err := toolGrep(map[string]any{"pattern": "[", "path": dir}); err == nil {
		t.Error("grep with invalid regex should error")
	}
}

func TestToolGrepSkipsBinary(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "bin.dat"), "needle\x00\x00\x00\x00needle")
	mustWrite(t, filepath.Join(dir, "ok.txt"), "needle here")

	out, err := toolGrep(map[string]any{"pattern": "needle", "path": dir})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out, "bin.dat") {
		t.Errorf("grep should skip binary files, got hit: %q", out)
	}
	if !strings.Contains(out, "ok.txt") {
		t.Errorf("grep should still match text files: %q", out)
	}
}

func TestToolWriteFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "out.txt")
	_, err := toolWriteFile(map[string]any{"path": path, "content": "hello"})
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(path)
	if err != nil || string(got) != "hello" {
		t.Fatalf("write_file: got %q err=%v", got, err)
	}

	// Overwrite.
	if _, err := toolWriteFile(map[string]any{"path": path, "content": "replaced"}); err != nil {
		t.Fatal(err)
	}
	got, _ = os.ReadFile(path)
	if string(got) != "replaced" {
		t.Errorf("write_file should overwrite, got %q", got)
	}
}

func TestToolEditFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "code.go")
	mustWrite(t, path, "package main\nfunc oldName() {}\n")

	_, err := toolEditFile(map[string]any{
		"path":       path,
		"old_string": "oldName",
		"new_string": "newName",
	})
	if err != nil {
		t.Fatalf("edit_file: %v", err)
	}
	got, _ := os.ReadFile(path)
	if !strings.Contains(string(got), "newName") || strings.Contains(string(got), "oldName") {
		t.Errorf("edit_file did not apply replacement, got %q", got)
	}

	// Not found → error.
	if _, err := toolEditFile(map[string]any{
		"path":       path,
		"old_string": "neverWasHere",
		"new_string": "x",
	}); err == nil {
		t.Error("edit_file on missing needle should error")
	}

	// Multiple matches → reject.
	mustWrite(t, path, "dup\ndup\n")
	if _, err := toolEditFile(map[string]any{
		"path":       path,
		"old_string": "dup",
		"new_string": "x",
	}); err == nil {
		t.Error("edit_file with non-unique needle should reject")
	}

	// Empty old_string → reject.
	if _, err := toolEditFile(map[string]any{
		"path":       path,
		"old_string": "",
		"new_string": "x",
	}); err == nil {
		t.Error("edit_file with empty old_string should reject")
	}
}

func TestToolRunCmd(t *testing.T) {
	cmd := "echo hello"
	if runtime.GOOS == "windows" {
		cmd = "echo hello"
	}
	out, err := toolRunCmd(map[string]any{"command": cmd})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "exit=0") {
		t.Errorf("expected exit=0 line, got %q", out)
	}
	if !strings.Contains(out, "hello") {
		t.Errorf("expected stdout in result, got %q", out)
	}
}

func TestToolRunCmdNonZeroExit(t *testing.T) {
	cmd := "exit 3"
	if runtime.GOOS == "windows" {
		cmd = "exit /B 3"
	}
	out, err := toolRunCmd(map[string]any{"command": cmd})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "exit=3") {
		t.Errorf("expected exit=3, got %q", out)
	}
}

func TestToolDefsJSONShape(t *testing.T) {
	defs := toolDefsJSON()
	if len(defs) != len(toolRegistry) {
		t.Fatalf("expected %d tool defs, got %d", len(toolRegistry), len(defs))
	}
	for _, d := range defs {
		if d["type"] != "function" {
			t.Errorf("tool def should have type=function: %v", d)
		}
		fn, ok := d["function"].(map[string]any)
		if !ok {
			t.Fatalf("tool def.function missing: %v", d)
		}
		if _, ok := fn["name"].(string); !ok {
			t.Errorf("tool def.function.name must be a string: %v", fn)
		}
		params, ok := fn["parameters"].(map[string]any)
		if !ok {
			t.Fatalf("tool def.function.parameters must be an object: %v", fn)
		}
		if params["type"] != "object" {
			t.Errorf("tool params.type must be 'object', got %v", params["type"])
		}
	}
}

func TestToolNamesStable(t *testing.T) {
	got := toolNames()
	// Expect at least the core six; sorted.
	want := []string{"edit_file", "grep", "list_dir", "read_file", "run_cmd", "write_file"}
	for _, w := range want {
		found := false
		for _, g := range got {
			if g == w {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("toolNames missing %q (got %v)", w, got)
		}
	}
	for i := 1; i < len(got); i++ {
		if got[i-1] > got[i] {
			t.Errorf("toolNames not sorted: %v", got)
			break
		}
	}
}

func TestSummarizeToolCallArgs(t *testing.T) {
	got := summarizeToolCallArgs(map[string]any{"path": "main.go"})
	if !strings.Contains(got, "main.go") {
		t.Errorf("summary should include arg value: %q", got)
	}
	big := summarizeToolCallArgs(map[string]any{"content": strings.Repeat("x", 500)})
	if len(big) > 125 {
		t.Errorf("summary should be truncated, got %d chars", len(big))
	}
}

func mustWrite(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}
