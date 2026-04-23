package main

import (
	"bufio"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

type GrepHit struct {
	Path    string
	Line    int
	Snippet string
}

// DefaultGrepMaxSize is the per-file size cap for --grep. Files larger than
// this are skipped: the file content is embedded in the prompt as a -p arg,
// and Windows caps CreateProcess command-lines at ~32K. Minified/generated
// files above this threshold are rarely useful for semantic search anyway.
const DefaultGrepMaxSize = 32 * 1024

// grepDirectory walks targetDir and asks the local model to identify lines
// semantically matching query in each text file. Files larger than maxSize
// bytes are skipped (0 = use DefaultGrepMaxSize). progress is called with
// status strings; pass nil to write to stdout.
func grepDirectory(targetDir, query string, maxSize int64, progress func(string)) ([]GrepHit, error) {
	if maxSize <= 0 {
		maxSize = DefaultGrepMaxSize
	}
	log := func(s string) {
		if progress != nil {
			progress(s)
		} else {
			fmt.Println(s)
		}
	}

	ignorer := loadIgnorer(targetDir)
	var hits []GrepHit

	err := filepath.WalkDir(targetDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		relPath, rerr := filepath.Rel(targetDir, path)
		if rerr != nil {
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

		if info, ierr := d.Info(); ierr == nil && info.Size() > maxSize {
			log(fmt.Sprintf("Skipping %s (%d bytes > max-size %d)", relPath, info.Size(), maxSize))
			return nil
		}

		content, err := os.ReadFile(path)
		if err != nil {
			return nil
		}
		if strings.Contains(string(content), "\x00") {
			return nil
		}

		log(fmt.Sprintf("Searching %s...", relPath))
		fileHits, err := grepFile(relPath, string(content), query)
		if err != nil {
			log(fmt.Sprintf("  warning: %v", err))
			return nil
		}
		hits = append(hits, fileHits...)
		return nil
	})
	return hits, err
}

func grepFile(relPath, content, query string) ([]GrepHit, error) {
	numbered := numberLines(content)
	prompt := fmt.Sprintf(`You are a semantic code search tool. The user is looking for:

%q

Below is a file with line numbers. Identify lines that MATCH the query — exact
matches, close paraphrases, or clearly related code. Be strict: skip lines that
are only tangentially related.

Respond with ONE match per line in the format:
LINE:<number>

If nothing matches, respond with exactly:
NONE

Do not include explanations or any other text.

FILE: %s
----
%s
----`, query, relPath, numbered)

	raw, err := runInference(prompt, 256)
	if err != nil {
		return nil, err
	}
	return parseGrepResponse(raw, relPath, content), nil
}

func numberLines(content string) string {
	var b strings.Builder
	sc := bufio.NewScanner(strings.NewReader(content))
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	i := 1
	for sc.Scan() {
		fmt.Fprintf(&b, "%d: %s\n", i, sc.Text())
		i++
	}
	return b.String()
}

var lineRe = regexp.MustCompile(`(?i)LINE\s*:\s*(\d+)`)

func parseGrepResponse(raw, relPath, content string) []GrepHit {
	if strings.Contains(strings.ToUpper(raw), "NONE") &&
		!strings.Contains(strings.ToUpper(raw), "LINE") {
		return nil
	}
	lines := strings.Split(content, "\n")
	seen := make(map[int]bool)
	var hits []GrepHit
	for _, m := range lineRe.FindAllStringSubmatch(raw, -1) {
		n, err := strconv.Atoi(m[1])
		if err != nil || n < 1 || n > len(lines) || seen[n] {
			continue
		}
		seen[n] = true
		hits = append(hits, GrepHit{
			Path:    relPath,
			Line:    n,
			Snippet: strings.TrimRight(lines[n-1], "\r"),
		})
	}
	return hits
}

func formatGrepHits(hits []GrepHit) string {
	if len(hits) == 0 {
		return "No matches."
	}
	var b strings.Builder
	for _, h := range hits {
		fmt.Fprintf(&b, "%s:%d: %s\n", h.Path, h.Line, h.Snippet)
	}
	return strings.TrimRight(b.String(), "\n")
}
