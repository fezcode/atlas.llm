package main

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// ProgressFn is called as bytes stream in. total may be -1 if unknown.
type ProgressFn func(written, total int64)

type countingWriter struct {
	written int64
	total   int64
	onWrite ProgressFn
}

func (cw *countingWriter) Write(p []byte) (int, error) {
	n := len(p)
	cw.written += int64(n)
	if cw.onWrite != nil {
		cw.onWrite(cw.written, cw.total)
	}
	return n, nil
}

func downloadFile(dest, url string, onProgress ProgressFn) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer out.Close()

	cw := &countingWriter{total: resp.ContentLength, onWrite: onProgress}
	if _, err := io.Copy(io.MultiWriter(out, cw), resp.Body); err != nil {
		return err
	}
	return nil
}

// githubAsset is the subset of GitHub's release-asset JSON we care about.
type githubAsset struct {
	Name               string `json:"name"`
	BrowserDownloadURL string `json:"browser_download_url"`
}

type githubRelease struct {
	TagName string        `json:"tag_name"`
	Assets  []githubAsset `json:"assets"`
}

// latestLlamacppAsset resolves the correct llama.cpp release asset URL for
// this OS/arch by querying GitHub for the latest release.
func latestLlamacppAsset() (string, string, error) {
	key := runtime.GOOS + "/" + runtime.GOARCH
	suffix, ok := llamacppAssetSuffix[key]
	if !ok {
		return "", "", fmt.Errorf("no llama.cpp prebuilt available for %s", key)
	}

	resp, err := http.Get(llamacppLatestURL)
	if err != nil {
		return "", "", fmt.Errorf("fetch latest release: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", "", fmt.Errorf("GitHub API returned %s", resp.Status)
	}

	var rel githubRelease
	if err := json.NewDecoder(resp.Body).Decode(&rel); err != nil {
		return "", "", fmt.Errorf("decode release JSON: %w", err)
	}
	for _, a := range rel.Assets {
		if strings.HasSuffix(a.Name, suffix) {
			return a.BrowserDownloadURL, rel.TagName, nil
		}
	}
	return "", "", fmt.Errorf("no asset ending in %q in release %s", suffix, rel.TagName)
}

// downloadEngine fetches the latest llama.cpp prebuilt archive for the
// current platform, extracts it into the engine dir, and removes any legacy
// llamafile binary left over from older atlas.llm versions.
func downloadEngine(onProgress ProgressFn) error {
	if isEngineDownloaded() {
		return nil
	}
	url, _, err := latestLlamacppAsset()
	if err != nil {
		return err
	}
	dir, err := engineDir()
	if err != nil {
		return err
	}

	archiveName := "llamacpp" + filepath.Ext(url)
	if strings.HasSuffix(url, ".tar.gz") {
		archiveName = "llamacpp.tar.gz"
	}
	archivePath := filepath.Join(dir, archiveName)

	if err := downloadFile(archivePath, url, onProgress); err != nil {
		return err
	}
	defer os.Remove(archivePath)

	if strings.HasSuffix(archiveName, ".zip") {
		if err := extractZip(archivePath, dir); err != nil {
			return fmt.Errorf("extract zip: %w", err)
		}
	} else {
		if err := extractTarGz(archivePath, dir); err != nil {
			return fmt.Errorf("extract tar.gz: %w", err)
		}
	}

	if runtime.GOOS != "windows" {
		if bin, err := findEngineBinary(); err == nil {
			_ = os.Chmod(bin, 0755)
		}
	}

	// Best-effort cleanup of the old llamafile binary from pre-0.4 installs.
	if base, err := atlasDir(); err == nil {
		for _, name := range []string{"llamafile", "llamafile.exe"} {
			_ = os.Remove(filepath.Join(base, name))
		}
	}

	return nil
}

func extractZip(src, destDir string) error {
	r, err := zip.OpenReader(src)
	if err != nil {
		return err
	}
	defer r.Close()
	cleanDest := filepath.Clean(destDir) + string(os.PathSeparator)
	for _, f := range r.File {
		target := filepath.Join(destDir, f.Name)
		if !strings.HasPrefix(target, cleanDest) {
			return fmt.Errorf("zip slip: %s", f.Name)
		}
		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(target, 0755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
			return err
		}
		if err := writeZipEntry(f, target); err != nil {
			return err
		}
	}
	return nil
}

func writeZipEntry(f *zip.File, target string) error {
	rc, err := f.Open()
	if err != nil {
		return err
	}
	defer rc.Close()
	out, err := os.OpenFile(target, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, rc)
	return err
}

func extractTarGz(src, destDir string) error {
	f, err := os.Open(src)
	if err != nil {
		return err
	}
	defer f.Close()
	gzr, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gzr.Close()
	tr := tar.NewReader(gzr)
	cleanDest := filepath.Clean(destDir) + string(os.PathSeparator)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		target := filepath.Join(destDir, hdr.Name)
		if !strings.HasPrefix(target, cleanDest) {
			return fmt.Errorf("tar slip: %s", hdr.Name)
		}
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
				return err
			}
			if err := writeTarEntry(tr, target, os.FileMode(hdr.Mode)); err != nil {
				return err
			}
		}
	}
}

func writeTarEntry(tr *tar.Reader, target string, mode os.FileMode) error {
	out, err := os.OpenFile(target, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, mode)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, tr)
	return err
}

// downloadModel fetches a model into models/. No-op if already present.
func downloadModel(m Model, onProgress ProgressFn) error {
	p, err := modelPath(m)
	if err != nil {
		return err
	}
	if isModelDownloaded(m) {
		return nil
	}
	return downloadFile(p, m.URL, onProgress)
}

// requireEngine returns the path to llama-cli[.exe] or an error asking the
// user to /download. Does NOT download automatically.
func requireEngine() (string, error) {
	if !isEngineDownloaded() {
		return "", fmt.Errorf("inference engine is not downloaded — run /download engine (or /download) in chat")
	}
	return findEngineBinary()
}

// requireModel returns the model path or an error asking the user to /download.
// Does NOT download automatically.
func requireModel(m Model) (string, error) {
	p, err := modelPath(m)
	if err != nil {
		return "", err
	}
	if !isModelDownloaded(m) {
		return "", fmt.Errorf("model %q is not downloaded — run /download %s in chat", m.Name, m.Name)
	}
	return p, nil
}

// runChat drives a /v1/chat/completions call against the persistent
// llama-server. The server is lazy-started on the first call per process
// (or whenever the active model changes) so the GGUF mmap + warmup cost is
// paid once per session, not once per turn.
//
// Using chat completions (instead of raw /completion) means llama-server
// applies the model's native chat template — Gemma 3's
// <start_of_turn>/<end_of_turn> sentinels, ChatML, etc. — and stops at the
// turn boundary. Raw completion with "User:/Assistant:" markers was causing
// the model to hallucinate additional fake turns after its real answer.
func runChat(msgs []ChatMsg, maxTokens int) (string, error) {
	if _, err := requireEngine(); err != nil {
		return "", err
	}
	if m, err := currentModel(); err == nil {
		if _, err := requireModel(m); err != nil {
			return "", err
		}
	}

	s, err := ensureServer()
	if err != nil {
		return "", fmt.Errorf("server: %w", err)
	}
	out, err := s.ChatComplete(msgs, maxTokens)
	if err != nil {
		return "", fmt.Errorf("inference failed: %w", err)
	}
	return strings.TrimSpace(out), nil
}

// runSingleUser is a convenience wrapper for one-shot tasks (summarize,
// grep) that have no conversational history — just a single user prompt.
func runSingleUser(system, user string, maxTokens int) (string, error) {
	msgs := []ChatMsg{}
	if system != "" {
		msgs = append(msgs, ChatMsg{Role: "system", Content: system})
	}
	msgs = append(msgs, ChatMsg{Role: "user", Content: user})
	return runChat(msgs, maxTokens)
}

func summarizeContent(content string) (string, error) {
	return runSingleUser(
		"You are a concise code summarizer. Respond with only 1-3 plain sentences describing the file's purpose. Do not use markdown, code blocks, or lists.",
		"Summarize this file:\n\n"+content,
		150,
	)
}

type ChatMessage struct {
	Role    string // "user" or "assistant"
	Content string
}

func chat(history []ChatMessage, userInput string) (string, error) {
	msgs := []ChatMsg{
		{Role: "system", Content: "You are a concise, helpful coding assistant. Keep replies under three short paragraphs unless more detail is explicitly requested."},
	}
	for _, m := range history {
		msgs = append(msgs, ChatMsg{Role: m.Role, Content: m.Content})
	}
	msgs = append(msgs, ChatMsg{Role: "user", Content: userInput})
	return runChat(msgs, 192)
}

func formatBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for n2 := n / unit; n2 >= unit; n2 /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "KMGTPE"[exp])
}
