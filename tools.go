package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"time"
)

// Tool is one agentic capability the model can invoke. Name maps 1:1 to the
// function name advertised in the /v1/chat/completions "tools" param.
// Destructive tools (writes, edits, shell execution) route through the TUI
// confirm flow before Run is called.
type Tool struct {
	Name        string
	Description string
	// Parameters is a JSON Schema object describing the tool's arguments.
	// Encoded as-is into the tool definition sent to llama-server.
	Parameters  map[string]any
	Destructive bool
	// Run executes the tool with decoded args and returns the string that
	// will be fed back to the model as the `role: tool` reply. Errors are
	// stringified and returned the same way so the model can recover.
	Run func(args map[string]any) (string, error)
}

// toolResultSizeLimit caps the bytes we feed back to the model per tool
// call. Keeps runaway directory listings, large files, and chatty MCP
// servers from blowing the context window.
//
// Sized against the context, not picked arbitrarily: at roughly 4 bytes per
// token this is ~1.5K tokens, about 9% of the default 16K ctx. The previous
// 32KB was ~8K tokens — half the entire window from a single call, so two
// tool calls exhausted the context before any real work happened.
const toolResultSizeLimit = 6 * 1024

// maxAgentSteps bounds how many tool-call rounds a single user prompt can
// trigger. Stops runaway loops if the model keeps asking for one more read.
const maxAgentSteps = 20

var toolRegistry = map[string]Tool{
	"read_file": {
		Name:        "read_file",
		Description: "Read a UTF-8 text file from the project and return its contents. Use for small-to-medium source files.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{
					"type":        "string",
					"description": "Path relative to the project root (the directory atlas.llm was started in). Paths outside it are refused.",
				},
			},
			"required": []string{"path"},
		},
		Run: toolReadFile,
	},
	"list_dir": {
		Name:        "list_dir",
		Description: "List entries in a directory. Returns one line per entry, with a trailing '/' on directories.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{
					"type":        "string",
					"description": "Directory path relative to the project root. Defaults to '.' (the root itself).",
				},
			},
		},
		Run: toolListDir,
	},
	"grep": {
		Name:        "grep",
		Description: "Regex search across files under a directory. Returns 'path:line: snippet' lines for each match. Skips binaries and files larger than 256KB.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"pattern": map[string]any{
					"type":        "string",
					"description": "Go/RE2 regular expression. Case-sensitive by default; prefix with (?i) for case-insensitive.",
				},
				"path": map[string]any{
					"type":        "string",
					"description": "Directory to search, relative to the project root. Defaults to '.'.",
				},
			},
			"required": []string{"pattern"},
		},
		Run: toolGrep,
	},
	"write_file": {
		Name:        "write_file",
		Description: "Overwrite a file with new contents. Creates parent directories if needed. Confirmation required.",
		Destructive: true,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{
					"type":        "string",
					"description": "Destination path relative to the project root.",
				},
				"content": map[string]any{
					"type":        "string",
					"description": "Full new contents of the file (UTF-8).",
				},
			},
			"required": []string{"path", "content"},
		},
		Run: toolWriteFile,
	},
	"edit_file": {
		Name:        "edit_file",
		Description: "Replace exactly one occurrence of old_string with new_string in the given file. old_string must appear exactly once; otherwise the edit is rejected. Confirmation required.",
		Destructive: true,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{
					"type":        "string",
					"description": "File to edit, relative to the project root.",
				},
				"old_string": map[string]any{
					"type":        "string",
					"description": "Exact string to find. Include enough surrounding context for it to be unique in the file.",
				},
				"new_string": map[string]any{
					"type":        "string",
					"description": "Replacement string.",
				},
			},
			"required": []string{"path", "old_string", "new_string"},
		},
		Run: toolEditFile,
	},
	"multi_edit": {
		Name: "multi_edit",
		Description: "Apply several find/replace edits to one file in a single atomic operation. " +
			"Edits are applied in order, and each old_string must match exactly once at the point it is applied " +
			"(so a later edit may target text an earlier one introduced). " +
			"If any edit fails, the file is left completely untouched. " +
			"Prefer this over repeated edit_file calls when changing one file in several places. Confirmation required.",
		Destructive: true,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{
					"type":        "string",
					"description": "File to edit, relative to the project root.",
				},
				"edits": map[string]any{
					"type":        "array",
					"description": "Edits to apply, in order.",
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"old_string": map[string]any{
								"type":        "string",
								"description": "Exact string to find. Include enough surrounding context for it to be unique in the file.",
							},
							"new_string": map[string]any{
								"type":        "string",
								"description": "Replacement string.",
							},
						},
						"required": []string{"old_string", "new_string"},
					},
				},
			},
			"required": []string{"path", "edits"},
		},
		Run: toolMultiEdit,
	},
	"run_cmd": {
		Name:        "run_cmd",
		Description: "Execute a shell command and return combined stdout+stderr. Runs in the project root unless cwd is given. Each call is a fresh shell, so use cwd rather than a leading 'cd'. 30s timeout. Confirmation required.",
		Destructive: true,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"command": map[string]any{
					"type":        "string",
					"description": "Shell command. Runs via 'cmd /C' on Windows and 'sh -c' elsewhere.",
				},
				"cwd": map[string]any{
					"type": "string",
					"description": "Directory to run in, relative to the project root. " +
						"Use this instead of a leading 'cd' — each call is a fresh shell, " +
						"so a 'cd' does not carry over to later calls.",
				},
			},
			"required": []string{"command"},
		},
		Run: toolRunCmd,
	},
}

// toolNames returns the built-in registry keys in stable order — used to
// render status output.
func toolNames() []string {
	names := make([]string, 0, len(toolRegistry))
	for n := range toolRegistry {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// activeTools returns every tool the model can call this turn: the built-ins
// plus whatever the connected MCP servers are contributing. Built-ins sort
// first so the namespaced MCP tools don't bury them.
func activeTools() []Tool {
	out := make([]Tool, 0, len(toolRegistry))
	for _, n := range toolNames() {
		out = append(out, toolRegistry[n])
	}
	return append(out, mcpToolSnapshot()...)
}

// toolDefsJSON builds the OpenAI-compatible `tools` payload once per call.
// Shape: [{"type":"function","function":{"name":...,"description":...,"parameters":{...}}}]
func toolDefsJSON() []map[string]any {
	tools := activeTools()
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		out = append(out, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        t.Name,
				"description": t.Description,
				"parameters":  t.Parameters,
			},
		})
	}
	return out
}

// argString pulls a required string arg from the decoded map, returning a
// user-friendly error for missing/wrong-typed values so the model can correct.
func argString(args map[string]any, key string, required bool) (string, error) {
	v, ok := args[key]
	if !ok || v == nil {
		if required {
			return "", fmt.Errorf("missing required argument %q", key)
		}
		return "", nil
	}
	s, ok := v.(string)
	if !ok {
		return "", fmt.Errorf("argument %q must be a string (got %T)", key, v)
	}
	return s, nil
}

func truncateForModel(s string) string {
	if len(s) <= toolResultSizeLimit {
		return s
	}
	return s[:toolResultSizeLimit] + fmt.Sprintf("\n\n... (truncated, %d more bytes)", len(s)-toolResultSizeLimit)
}

func toolReadFile(args map[string]any) (string, error) {
	path, err := argString(args, "path", true)
	if err != nil {
		return "", err
	}
	abs, err := resolveInRoot(path)
	if err != nil {
		return "", err
	}
	b, err := os.ReadFile(abs)
	if err != nil {
		return "", err
	}
	return truncateForModel(string(b)), nil
}

func toolListDir(args map[string]any) (string, error) {
	path, _ := argString(args, "path", false)
	if path == "" {
		path = "."
	}
	abs, err := resolveInRoot(path)
	if err != nil {
		return "", err
	}
	entries, err := os.ReadDir(abs)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() {
			name += "/"
		}
		b.WriteString(name)
		b.WriteByte('\n')
	}
	return truncateForModel(b.String()), nil
}

func toolGrep(args map[string]any) (string, error) {
	pattern, err := argString(args, "pattern", true)
	if err != nil {
		return "", err
	}
	rootArg, _ := argString(args, "path", false)
	if rootArg == "" {
		rootArg = "."
	}
	root, err := resolveInRoot(rootArg)
	if err != nil {
		return "", err
	}
	re, err := regexp.Compile(pattern)
	if err != nil {
		return "", fmt.Errorf("invalid regex: %w", err)
	}
	const maxFileBytes = 256 * 1024
	const maxHits = 200
	var b strings.Builder
	hits := 0
	err = filepath.WalkDir(root, func(path string, d os.DirEntry, werr error) error {
		if werr != nil {
			return nil
		}
		if d.IsDir() {
			n := d.Name()
			if n == ".git" || n == "node_modules" || n == "vendor" || n == ".atlas" {
				return filepath.SkipDir
			}
			return nil
		}
		info, err := d.Info()
		if err != nil || info.Size() > maxFileBytes {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil || !isLikelyText(data) {
			return nil
		}
		lines := bytes.Split(data, []byte("\n"))
		for i, line := range lines {
			if re.Match(line) {
				snippet := strings.TrimSpace(string(line))
				if len(snippet) > 200 {
					snippet = snippet[:200] + "…"
				}
				fmt.Fprintf(&b, "%s:%d: %s\n", displayPath(path), i+1, snippet)
				hits++
				if hits >= maxHits {
					return filepath.SkipAll
				}
			}
		}
		return nil
	})
	if err != nil {
		return "", err
	}
	if hits == 0 {
		return "(no matches)", nil
	}
	return truncateForModel(b.String()), nil
}

// isLikelyText is a crude binary sniff: treats content with a NUL byte in
// the first 8KB as binary.
func isLikelyText(data []byte) bool {
	n := len(data)
	if n > 8192 {
		n = 8192
	}
	for i := 0; i < n; i++ {
		if data[i] == 0 {
			return false
		}
	}
	return true
}

func toolWriteFile(args map[string]any) (string, error) {
	path, err := argString(args, "path", true)
	if err != nil {
		return "", err
	}
	content, err := argString(args, "content", true)
	if err != nil {
		return "", err
	}
	abs, err := resolveInRoot(path)
	if err != nil {
		return "", err
	}
	if dir := filepath.Dir(abs); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return "", err
		}
	}
	if err := os.WriteFile(abs, []byte(content), 0644); err != nil {
		return "", err
	}
	return fmt.Sprintf("Wrote %d bytes to %s", len(content), displayPath(abs)), nil
}

func toolEditFile(args map[string]any) (string, error) {
	path, err := argString(args, "path", true)
	if err != nil {
		return "", err
	}
	oldStr, err := argString(args, "old_string", true)
	if err != nil {
		return "", err
	}
	newStr, err := argString(args, "new_string", true)
	if err != nil {
		return "", err
	}
	if oldStr == "" {
		return "", fmt.Errorf("old_string must not be empty")
	}
	abs, err := resolveInRoot(path)
	if err != nil {
		return "", err
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return "", err
	}
	count := strings.Count(string(data), oldStr)
	if count == 0 {
		return "", fmt.Errorf("old_string not found in %s", displayPath(abs))
	}
	if count > 1 {
		return "", fmt.Errorf("old_string matches %d times in %s — needs to be unique; include more surrounding context", count, displayPath(abs))
	}
	updated := strings.Replace(string(data), oldStr, newStr, 1)
	if err := os.WriteFile(abs, []byte(updated), 0644); err != nil {
		return "", err
	}
	return fmt.Sprintf("Replaced 1 occurrence in %s (%d bytes → %d bytes)", displayPath(abs), len(data), len(updated)), nil
}

func toolRunCmd(args map[string]any) (string, error) {
	command, err := argString(args, "command", true)
	if err != nil {
		return "", err
	}
	cwdArg, _ := argString(args, "cwd", false)
	dir, err := resolveInRoot(cwdArg)
	if err != nil {
		return "", err
	}
	if info, err := os.Stat(dir); err != nil || !info.IsDir() {
		return "", fmt.Errorf("cwd %q is not a directory under the project root", cwdArg)
	}

	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("cmd", "/C", command)
	} else {
		cmd = exec.Command("sh", "-c", command)
	}
	// Each call is a fresh shell, so `cd` never persists. Anchoring here is
	// what makes a working directory meaningful across calls.
	cmd.Dir = dir
	done := make(chan error, 1)
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	if err := cmd.Start(); err != nil {
		return "", err
	}
	go func() { done <- cmd.Wait() }()
	select {
	case werr := <-done:
		exitCode := 0
		if werr != nil {
			if ee, ok := werr.(*exec.ExitError); ok {
				exitCode = ee.ExitCode()
			} else {
				return "", werr
			}
		}
		return fmt.Sprintf("exit=%d\n%s", exitCode, truncateForModel(out.String())), nil
	case <-time.After(30 * time.Second):
		_ = cmd.Process.Kill()
		<-done
		return "", fmt.Errorf("command timed out after 30s")
	}
}

// summarizeToolCallArgs returns a single-line, truncated JSON rendering of
// the decoded args, for inline display in the TUI trace.
func summarizeToolCallArgs(args map[string]any) string {
	b, err := json.Marshal(args)
	if err != nil {
		return "{}"
	}
	s := string(b)
	if len(s) > 120 {
		s = s[:120] + "…"
	}
	return s
}

// fileEdit is one find/replace pair within a multi_edit call.
type fileEdit struct {
	oldStr string
	newStr string
}

// argEdits decodes the `edits` array. Models get this shape wrong in
// predictable ways, so each failure names the offending index.
func argEdits(args map[string]any) ([]fileEdit, error) {
	raw, ok := args["edits"]
	if !ok || raw == nil {
		return nil, fmt.Errorf("missing required argument %q", "edits")
	}
	list, ok := raw.([]any)
	if !ok {
		return nil, fmt.Errorf("argument %q must be an array of {old_string, new_string} objects (got %T)", "edits", raw)
	}
	if len(list) == 0 {
		return nil, fmt.Errorf("%q is empty — nothing to do", "edits")
	}
	out := make([]fileEdit, 0, len(list))
	for i, item := range list {
		obj, ok := item.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("edits[%d] must be an object with old_string and new_string (got %T)", i, item)
		}
		oldStr, err := argString(obj, "old_string", true)
		if err != nil {
			return nil, fmt.Errorf("edits[%d]: %w", i, err)
		}
		newStr, err := argString(obj, "new_string", true)
		if err != nil {
			return nil, fmt.Errorf("edits[%d]: %w", i, err)
		}
		if oldStr == "" {
			return nil, fmt.Errorf("edits[%d]: old_string must not be empty", i)
		}
		out = append(out, fileEdit{oldStr: oldStr, newStr: newStr})
	}
	return out, nil
}

// toolMultiEdit applies every edit to an in-memory copy and writes once at
// the end. A failure anywhere means nothing is written — a partially
// applied batch would leave the file in a state neither the user nor the
// model expects.
func toolMultiEdit(args map[string]any) (string, error) {
	path, err := argString(args, "path", true)
	if err != nil {
		return "", err
	}
	edits, err := argEdits(args)
	if err != nil {
		return "", err
	}
	abs, err := resolveInRoot(path)
	if err != nil {
		return "", err
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return "", err
	}

	updated := string(data)
	for i, e := range edits {
		// Counted against the running content, not the original, so an edit
		// may legitimately target text an earlier edit introduced.
		switch strings.Count(updated, e.oldStr) {
		case 0:
			return "", fmt.Errorf("edits[%d]: old_string not found in %s. No edits were applied", i, displayPath(abs))
		case 1:
		default:
			return "", fmt.Errorf("edits[%d]: old_string matches %d times in %s — needs to be unique; "+
				"include more surrounding context. No edits were applied",
				i, strings.Count(updated, e.oldStr), displayPath(abs))
		}
		updated = strings.Replace(updated, e.oldStr, e.newStr, 1)
	}

	if updated == string(data) {
		return fmt.Sprintf("No change: the %d edits left %s byte-identical", len(edits), displayPath(abs)), nil
	}
	if err := os.WriteFile(abs, []byte(updated), 0644); err != nil {
		return "", err
	}
	return fmt.Sprintf("Applied %d edits to %s (%d bytes → %d bytes)",
		len(edits), displayPath(abs), len(data), len(updated)), nil
}
