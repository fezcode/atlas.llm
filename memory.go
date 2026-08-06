package main

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

// Reporting memory for the model server is done by measuring the running
// process, not by predicting from model shape.
//
// The obvious calculation — 2 tensors * layers * ctx * kv_heads * head_dim *
// 2 bytes for an f16 cache — overestimates badly on the models atlas.llm
// ships. Measured against llama-server holding Qwen3.5-4B: 0.91 GB resident
// at ctx 16384 and 2.36 GB at 65536, about 31 KB per token, where that
// formula predicts 128 KB per token. The gap is roughly 4x, consistent with
// hybrid attention where only a fraction of layers keep a full-context
// cache. Rather than show a number that is four times too large, atlas.llm
// reports what the process is actually using and what the weights cost on
// disk.

// processRSS returns the resident set size of a process in bytes, or ok=false
// when it can't be determined.
func processRSS(pid int) (int64, bool) {
	if pid <= 0 {
		return 0, false
	}
	if runtime.GOOS == "windows" {
		return windowsProcessRSS(pid)
	}
	// ps reports RSS in kilobytes on both macOS and Linux.
	out, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(pid)).Output()
	if err != nil {
		return 0, false
	}
	kb, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	if err != nil || kb <= 0 {
		return 0, false
	}
	return kb * 1024, true
}

// windowsProcessRSS parses tasklist's CSV output, whose memory column looks
// like "\"12,345 K\"".
func windowsProcessRSS(pid int) (int64, bool) {
	out, err := exec.Command("tasklist", "/FI", "PID eq "+strconv.Itoa(pid),
		"/FO", "CSV", "/NH").Output()
	if err != nil {
		return 0, false
	}
	fields := strings.Split(strings.TrimSpace(string(out)), ",")
	if len(fields) < 5 {
		return 0, false
	}
	mem := strings.Trim(fields[len(fields)-1], "\" \r\n")
	mem = strings.TrimSuffix(mem, " K")
	mem = strings.ReplaceAll(mem, ",", "")
	mem = strings.ReplaceAll(mem, " ", "")
	kb, err := strconv.ParseInt(strings.TrimSpace(mem), 10, 64)
	if err != nil || kb <= 0 {
		return 0, false
	}
	return kb * 1024, true
}

// serverMemory reports the running model server's resident memory. Returns
// ok=false when no server is running, which is the normal state before the
// first message.
func serverMemory() (int64, bool) {
	serverMu.Lock()
	s := activeServer
	serverMu.Unlock()
	if s == nil || s.cmd == nil || s.cmd.Process == nil {
		return 0, false
	}
	return processRSS(s.cmd.Process.Pid)
}

// memoryDisplay renders the memory line for `/config`: what the model server
// is actually using right now, plus the weights on disk as the floor any
// session has to pay.
func memoryDisplay(cfg Config) string {
	var parts []string
	if rss, ok := serverMemory(); ok {
		parts = append(parts, fmt.Sprintf("model server using %s now", formatBytes(rss)))
	} else {
		parts = append(parts, "model server not running")
	}
	if m, err := currentModel(); err == nil && isModelDownloaded(m) {
		if p, err := modelPath(m); err == nil {
			if info, err := statSize(p); err == nil {
				parts = append(parts, fmt.Sprintf("weights %s on disk", formatBytes(info)))
			}
		}
	}
	parts = append(parts, fmt.Sprintf("grows with ctx_size (%d)", resolveCtxSize(cfg)))
	return strings.Join(parts, " · ")
}

// statSize returns a file's size in bytes.
func statSize(path string) (int64, error) {
	info, err := os.Stat(path)
	if err != nil {
		return 0, err
	}
	return info.Size(), nil
}
