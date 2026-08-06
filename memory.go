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

// systemRAM returns total physical memory in bytes, or ok=false when it
// can't be determined. Used to say whether a model will actually fit rather
// than just how large it is.
func systemRAM() (int64, bool) {
	switch runtime.GOOS {
	case "darwin":
		out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
		if err != nil {
			return 0, false
		}
		n, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
		return n, err == nil && n > 0
	case "linux":
		data, err := os.ReadFile("/proc/meminfo")
		if err != nil {
			return 0, false
		}
		for _, line := range strings.Split(string(data), "\n") {
			if !strings.HasPrefix(line, "MemTotal:") {
				continue
			}
			f := strings.Fields(line)
			if len(f) < 2 {
				return 0, false
			}
			kb, err := strconv.ParseInt(f[1], 10, 64)
			return kb * 1024, err == nil && kb > 0
		}
		return 0, false
	case "windows":
		out, err := exec.Command("wmic", "ComputerSystem", "get", "TotalPhysicalMemory").Output()
		if err != nil {
			return 0, false
		}
		for _, line := range strings.Split(string(out), "\n") {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "TotalPhysical") {
				continue
			}
			n, err := strconv.ParseInt(line, 10, 64)
			if err == nil && n > 0 {
				return n, true
			}
		}
		return 0, false
	}
	return 0, false
}

// Fit describes how comfortably a model's weights sit in system memory.
type Fit int

const (
	FitUnknown Fit = iota
	FitComfortable
	FitOK
	FitTight
	FitTooBig
)

func (f Fit) String() string {
	switch f {
	case FitComfortable:
		return "comfortable"
	case FitOK:
		return "ok"
	case FitTight:
		return "tight"
	case FitTooBig:
		return "too big"
	}
	return ""
}

// modelFit judges a model's weights against total RAM.
//
// Only the weights are counted. The KV cache adds to this and grows with
// ctx_size, but predicting it from model shape proved unreliable (see the
// note at the top of this file), so the figure here is the floor rather
// than a total — deliberately, since an honest floor beats a wrong total.
func modelFit(m Model) (share float64, fit Fit) {
	total, ok := systemRAM()
	if !ok || total <= 0 {
		return 0, FitUnknown
	}
	weights := parseModelSize(m.Size)
	if weights <= 0 {
		return 0, FitUnknown
	}
	share = float64(weights) / float64(total)
	switch {
	case share < 0.25:
		return share, FitComfortable
	case share < 0.5:
		return share, FitOK
	case share < 0.75:
		return share, FitTight
	default:
		return share, FitTooBig
	}
}

// modelResourceNote is the per-model column in `/list` and the picker.
func modelResourceNote(m Model) string {
	share, fit := modelFit(m)
	if fit == FitUnknown {
		return ""
	}
	return fmt.Sprintf("%.0f%% RAM, %s", share*100, fit)
}
