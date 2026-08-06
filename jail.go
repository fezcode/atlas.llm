package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// Tools operate inside a jail rooted at the directory atlas.llm was started
// in. Paths may point anywhere at or below that root; anything above it is
// refused.
//
// This exists for two reasons. The obvious one is blast radius: the model
// can edit files and run commands, and a mistaken "../.." should not reach
// the rest of the filesystem. The subtler one is that relative paths were
// resolved against the process working directory with no root concept at
// all, so a model that reasoned about paths from the project root got
// confusing "no such file or directory" errors and tended to retry
// variations until it exhausted its tool-call budget.

var (
	rootOnce sync.Once
	rootDir  string
)

// sessionRoot is the directory tools are confined to: where atlas.llm was
// launched. Resolved once, with symlinks expanded so the containment check
// compares like with like.
func sessionRoot() string {
	rootOnce.Do(func() {
		wd, err := os.Getwd()
		if err != nil {
			rootDir = "."
			return
		}
		if resolved, err := filepath.EvalSymlinks(wd); err == nil {
			wd = resolved
		}
		rootDir = filepath.Clean(wd)
	})
	return rootDir
}

// setSessionRootForTest overrides the jail root and returns a restore func.
// Tests only.
//
// sessionRoot() is called first so the real initializer runs: consuming
// rootOnce with an empty function would leave the root permanently blank
// for anything that restores afterwards.
func setSessionRootForTest(dir string) func() {
	prev := sessionRoot()
	if resolved, err := filepath.EvalSymlinks(dir); err == nil {
		dir = resolved
	}
	rootDir = filepath.Clean(dir)
	return func() { rootDir = prev }
}

// withinRoot reports whether an already-absolute, cleaned path is the root
// or sits underneath it.
func withinRoot(abs, root string) bool {
	if abs == root {
		return true
	}
	return strings.HasPrefix(abs, root+string(os.PathSeparator))
}

// resolveInRoot turns a tool-supplied path into an absolute path inside the
// jail, or returns an error naming the violation.
//
// Symlinks are expanded on the deepest existing ancestor, so a link pointing
// outside the root is caught even when the final component doesn't exist yet
// (the write_file / multi_edit case).
func resolveInRoot(p string) (string, error) {
	root := sessionRoot()
	if strings.TrimSpace(p) == "" {
		return root, nil
	}
	abs := p
	if !filepath.IsAbs(abs) {
		abs = filepath.Join(root, abs)
	}
	abs = filepath.Clean(abs)

	// Walk up to the nearest existing ancestor and resolve symlinks there,
	// then re-attach the remainder.
	probe, rest := abs, ""
	for {
		if resolved, err := filepath.EvalSymlinks(probe); err == nil {
			abs = filepath.Clean(filepath.Join(resolved, rest))
			break
		}
		parent := filepath.Dir(probe)
		if parent == probe {
			break // reached the filesystem root without finding anything
		}
		rest = filepath.Join(filepath.Base(probe), rest)
		probe = parent
	}

	if !withinRoot(abs, root) {
		rel := p
		return "", fmt.Errorf("path %q is outside the working directory (%s). "+
			"Tools may only read and write at or below it — use a relative path", rel, root)
	}
	return abs, nil
}

// displayPath renders a jailed absolute path relative to the root, so tool
// output and confirmation prompts stay readable.
func displayPath(abs string) string {
	rel, err := filepath.Rel(sessionRoot(), abs)
	if err != nil || strings.HasPrefix(rel, "..") {
		return abs
	}
	if rel == "." {
		return "."
	}
	return rel
}

// projectOverview lists the root's top-level entries, for the agent system
// prompt.
//
// Models were guessing directory names — asking for "internal" in a project
// that has no such directory, getting an error, and retrying variations
// until they exhausted their tool-call budget. Showing the actual layout up
// front removes the guesswork, and costs one directory read per turn.
func projectOverview(maxEntries int) string {
	root := sessionRoot()
	entries, err := os.ReadDir(root)
	if err != nil {
		return ""
	}
	var dirs, files []string
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, ".") {
			continue // .git and friends are noise here
		}
		if e.IsDir() {
			dirs = append(dirs, name+"/")
		} else {
			files = append(files, name)
		}
	}
	all := append(dirs, files...)
	truncated := false
	if len(all) > maxEntries {
		all = all[:maxEntries]
		truncated = true
	}
	if len(all) == 0 {
		return ""
	}
	out := strings.Join(all, "  ")
	if truncated {
		out += "  …"
	}
	return out
}
