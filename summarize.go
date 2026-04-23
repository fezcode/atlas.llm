package main

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// DefaultSummarizeMaxSize is the per-file size cap for /summarize. Files
// bigger than this are skipped — summarizeContent already truncates the
// prompt to ~10KB, but walking megabyte-scale generated/minified files
// wastes time. Override via --max-size.
const DefaultSummarizeMaxSize int64 = 256 * 1024

// SummarizeOptions configures a single /summarize run.
type SummarizeOptions struct {
	TargetDir string
	Output    string
	MaxSize   int64
	Exclude   []string
}

// summarizeDirectory walks opts.TargetDir and writes per-file summaries to
// opts.Output (usually SUMMARY.md). progress is called with status strings;
// pass nil to write to stdout.
func summarizeDirectory(opts SummarizeOptions, progress func(string)) error {
	if opts.Output == "" {
		opts.Output = "SUMMARY.md"
	}
	if opts.MaxSize <= 0 {
		opts.MaxSize = DefaultSummarizeMaxSize
	}
	targetDir := opts.TargetDir
	if targetDir == "" {
		targetDir = "."
	}

	log := func(s string) {
		if progress != nil {
			progress(s)
		} else {
			fmt.Println(s)
		}
	}

	excludes := make(map[string]bool)
	for _, ext := range opts.Exclude {
		ext = strings.TrimSpace(ext)
		if ext == "" {
			continue
		}
		if !strings.HasPrefix(ext, ".") {
			ext = "." + ext
		}
		excludes[strings.ToLower(ext)] = true
	}

	outAbs, _ := filepath.Abs(opts.Output)
	ignorer := loadIgnorer(targetDir)

	out, err := os.Create(opts.Output)
	if err != nil {
		return fmt.Errorf("creating output: %w", err)
	}
	defer out.Close()

	m, err := currentModel()
	if err == nil {
		fmt.Fprintf(out, "# Project Summary\n\n_Generated with model: %s_\n\n", m.Name)
	} else {
		fmt.Fprintf(out, "# Project Summary\n\n")
	}

	return filepath.WalkDir(targetDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		absPath, _ := filepath.Abs(path)
		if absPath == outAbs {
			return nil
		}

		relPath, err := filepath.Rel(targetDir, path)
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
		if excludes[ext] {
			log(fmt.Sprintf("Skipping %s (excluded by --exclude)", relPath))
			return nil
		}
		if info, ierr := d.Info(); ierr == nil && info.Size() > opts.MaxSize {
			log(fmt.Sprintf("Skipping %s (%d bytes > max-size %d)", relPath, info.Size(), opts.MaxSize))
			return nil
		}

		content, err := os.ReadFile(path)
		if err != nil {
			return nil
		}
		if strings.Contains(string(content), "\x00") {
			return nil
		}

		log(fmt.Sprintf("Summarizing %s...", relPath))
		summary, err := summarizeContent(string(content))
		if err != nil {
			log(fmt.Sprintf("  warning: %v", err))
			return nil
		}
		fmt.Fprintf(out, "## %s\n\n%s\n\n", relPath, summary)
		return nil
	})
}
