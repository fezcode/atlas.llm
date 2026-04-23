package main

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	ignore "github.com/sabhiram/go-gitignore"
)

type DumpOptions struct {
	TargetDir string
	Output    string
	Exclude   []string
	Summarize bool
}

func runDump(opts DumpOptions) error {
	outAbs, _ := filepath.Abs(opts.Output)

	extraExcludes := make(map[string]bool)
	for _, ext := range opts.Exclude {
		ext = strings.TrimSpace(ext)
		if ext == "" {
			continue
		}
		if !strings.HasPrefix(ext, ".") {
			ext = "." + ext
		}
		extraExcludes[strings.ToLower(ext)] = true
	}

	ignorer := loadIgnorer(opts.TargetDir)

	out, err := os.Create(opts.Output)
	if err != nil {
		return fmt.Errorf("creating output: %w", err)
	}
	defer out.Close()

	fmt.Fprintf(out, "# Project Context\n\n")

	return filepath.WalkDir(opts.TargetDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		absPath, _ := filepath.Abs(path)
		if absPath == outAbs {
			return nil
		}

		relPath, err := filepath.Rel(opts.TargetDir, path)
		if err != nil {
			relPath = path
		}
		if relPath == "." {
			return nil
		}

		if d.IsDir() && d.Name() == ".git" {
			return filepath.SkipDir
		}

		if ignorer != nil && ignorer.MatchesPath(relPath) {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		if d.IsDir() {
			return nil
		}

		ext := strings.ToLower(filepath.Ext(path))
		if extraExcludes[ext] {
			return nil
		}

		content, err := os.ReadFile(path)
		if err != nil {
			return nil
		}
		if strings.Contains(string(content), "\x00") {
			return nil
		}

		lang := strings.TrimPrefix(ext, ".")
		if lang == "txt" || lang == "" {
			lang = "text"
		}

		fmt.Fprintf(out, "## %s\n", relPath)

		if opts.Summarize {
			fmt.Printf("Summarizing %s...\n", relPath)
			summary, err := summarizeContent(string(content))
			if err != nil {
				fmt.Fprintf(os.Stderr, "Warning: failed to summarize %s: %v\n", relPath, err)
			} else {
				fmt.Fprintf(out, "> **AI Summary:** %s\n\n", strings.ReplaceAll(summary, "\n", "\n> "))
			}
		}

		fmt.Fprintf(out, "```%s\n", lang)
		out.Write(content)
		if !strings.HasSuffix(string(content), "\n") {
			out.WriteString("\n")
		}
		fmt.Fprintf(out, "```\n\n")
		return nil
	})
}

func loadIgnorer(targetDir string) *ignore.GitIgnore {
	gitignorePath := filepath.Join(targetDir, ".gitignore")
	if _, err := os.Stat(gitignorePath); err == nil {
		ig, _ := ignore.CompileIgnoreFile(gitignorePath)
		return ig
	}
	return ignore.CompileIgnoreLines(".git", "node_modules", "vendor", "build", "dist", "*.exe", "*.dll", "*.so")
}
