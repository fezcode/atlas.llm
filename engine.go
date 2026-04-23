package main

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

// ProgressFn is called as bytes stream in. total may be -1 if unknown.
type ProgressFn func(written, total int64)

type countingWriter struct {
	written  int64
	total    int64
	onWrite  ProgressFn
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

// downloadEngine fetches llamafile into the data dir. No-op if already present.
func downloadEngine(onProgress ProgressFn) error {
	p, err := enginePath()
	if err != nil {
		return err
	}
	if isEngineDownloaded() {
		return nil
	}
	if err := downloadFile(p, llamafileURL, onProgress); err != nil {
		return err
	}
	if runtime.GOOS != "windows" {
		_ = os.Chmod(p, 0755)
	}
	return nil
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

// requireEngine returns the engine path or an error asking the user to /download.
// Does NOT download automatically.
func requireEngine() (string, error) {
	p, err := enginePath()
	if err != nil {
		return "", err
	}
	if !isEngineDownloaded() {
		return "", fmt.Errorf("inference engine is not downloaded — run /download engine (or /download) in chat")
	}
	return p, nil
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

func runInference(prompt string, maxTokens int) (string, error) {
	eng, err := requireEngine()
	if err != nil {
		return "", err
	}
	m, err := currentModel()
	if err != nil {
		return "", err
	}
	mp, err := requireModel(m)
	if err != nil {
		return "", err
	}

	cmd := exec.Command(eng, "-m", mp, "--cli", "-p", prompt, "-n", fmt.Sprintf("%d", maxTokens), "--temp", "0.2")
	var out, stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("inference failed: %v (stderr: %s)", err, stderr.String())
	}
	return strings.TrimSpace(out.String()), nil
}

func summarizeContent(content string) (string, error) {
	prompt := "Summarize the following code file concisely in 1-3 sentences:\n\n" + content
	return runInference(prompt, 150)
}

type ChatMessage struct {
	Role    string // "user" or "assistant"
	Content string
}

func buildChatPrompt(history []ChatMessage, userInput string) string {
	var b strings.Builder
	b.WriteString("You are a concise, helpful coding assistant.\n\n")
	for _, m := range history {
		switch m.Role {
		case "user":
			b.WriteString("User: ")
		case "assistant":
			b.WriteString("Assistant: ")
		}
		b.WriteString(m.Content)
		b.WriteString("\n")
	}
	b.WriteString("User: ")
	b.WriteString(userInput)
	b.WriteString("\nAssistant:")
	return b.String()
}

func chat(history []ChatMessage, userInput string) (string, error) {
	return runInference(buildChatPrompt(history, userInput), 512)
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
