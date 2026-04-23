package main

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// summarizeDirectory walks targetDir and writes per-file summaries to outFile (SUMMARY.md).
// progress is called with status strings; pass nil to write to stdout.
func summarizeDirectory(targetDir, outFile string, progress func(string)) error {
	log := func(s string) {
		if progress != nil {
			progress(s)
		} else {
			fmt.Println(s)
		}
	}

	outAbs, _ := filepath.Abs(outFile)
	ignorer := loadIgnorer(targetDir)

	out, err := os.Create(outFile)
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
