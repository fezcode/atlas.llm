package main

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/charmbracelet/lipgloss"
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
// like "\"12,345 K\"" — or "\"8.174.328 K\"", depending on the locale.
func windowsProcessRSS(pid int) (int64, bool) {
	out, err := exec.Command("tasklist", "/FI", "PID eq "+strconv.Itoa(pid),
		"/FO", "CSV", "/NH").Output()
	if err != nil {
		return 0, false
	}
	// Parse as real CSV rather than splitting on commas: where the locale
	// groups with commas the memory column contains one, so splitting left
	// half of "12,345 K" in its own field, which then parsed as 345.
	rec, err := csv.NewReader(bytes.NewReader(out)).Read()
	if err != nil || len(rec) < 5 {
		return 0, false
	}
	return parseTasklistKB(rec[len(rec)-1])
}

// parseTasklistKB reads tasklist's memory column into bytes. The grouping
// separator follows the system locale — "12,345 K" on en-US, "8.174.328 K"
// where it is a dot — so this keeps the digits and discards everything else
// rather than stripping one specific separator. A non-numeric column
// ("N/A", which tasklist emits for processes it can't inspect) is ok=false.
func parseTasklistKB(field string) (int64, bool) {
	var digits strings.Builder
	for _, r := range field {
		if r >= '0' && r <= '9' {
			digits.WriteRune(r)
		}
	}
	kb, err := strconv.ParseInt(digits.String(), 10, 64)
	if err != nil || kb <= 0 {
		return 0, false
	}
	return kb * 1024, true
}

// processRSS spawns a process on every platform — ps or tasklist — which is
// far too expensive for the header, because that re-renders on every
// keystroke. Measured at ~180ms per tasklist call on Windows, it was the
// entire input lag.
//
// Resident memory moves slowly, so the reading is cached and refreshed off
// the render path: callers get the last known value immediately and never
// block on a probe. The gauge stays blank until the first refresh lands
// rather than stalling the first frame.
const rssCacheTTL = 2 * time.Second

var (
	rssMu       sync.Mutex
	rssPID      int
	rssValue    int64
	rssOK       bool
	rssAt       time.Time
	rssInflight bool
)

func processRSSCached(pid int) (int64, bool) {
	rssMu.Lock()
	defer rssMu.Unlock()

	if pid != rssPID || time.Since(rssAt) >= rssCacheTTL {
		if !rssInflight {
			rssInflight = true
			go func() {
				v, ok := processRSS(pid)
				rssMu.Lock()
				rssPID, rssValue, rssOK = pid, v, ok
				rssAt, rssInflight = time.Now(), false
				rssMu.Unlock()
			}()
		}
	}
	if pid != rssPID {
		// A different process than the cached one — report nothing rather
		// than the previous server's memory.
		return 0, false
	}
	return rssValue, rssOK
}

// resetRSSCache drops the cached reading so a test can force a fresh probe.
func resetRSSCache() {
	rssMu.Lock()
	rssPID, rssValue, rssOK, rssAt, rssInflight = 0, 0, false, time.Time{}, false
	rssMu.Unlock()
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
	return processRSSCached(s.cmd.Process.Pid)
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

// Total RAM does not change while the process runs, and reading it shells
// out on every platform, so it is resolved once.
var (
	ramOnce  sync.Once
	ramBytes int64
	ramOK    bool
)

// systemRAM returns total physical memory in bytes, or ok=false when it
// can't be determined. Used to say whether a model will actually fit rather
// than just how large it is.
func systemRAM() (int64, bool) {
	ramOnce.Do(func() { ramBytes, ramOK = readSystemRAM() })
	return ramBytes, ramOK
}

func readSystemRAM() (int64, bool) {
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

// modelFit judges a model's total footprint — weights plus the KV cache at
// the current ctx_size — against system RAM.
//
// The KV term only appears for models already downloaded, since it needs the
// GGUF shape metadata. For anything not yet fetched this reports weights
// alone, which understates a large context.
func modelFit(m Model) (share float64, fit Fit) {
	total, ok := systemRAM()
	if !ok || total <= 0 {
		return 0, FitUnknown
	}
	cfg, _ := loadConfig()
	weights, kv, _ := modelMemoryEstimate(m, resolveCtxSize(cfg))
	if weights <= 0 {
		return 0, FitUnknown
	}
	share = float64(weights+kv) / float64(total)
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
	cfg, _ := loadConfig()
	if _, kv, ok := modelMemoryEstimate(m, resolveCtxSize(cfg)); ok && kv > 0 {
		return fmt.Sprintf("%.0f%% RAM (+%s ctx), %s", share*100, formatBytes(kv), fit)
	}
	return fmt.Sprintf("%.0f%% RAM, %s", share*100, fit)
}

// modelSizeBytes returns a model's weight size. The real file is used when
// it's on disk; the registry string is only a fallback for models not yet
// downloaded, and it is approximate (gemma-4-e2b declares ~2.9GB and is
// actually 3.11GB).
func modelSizeBytes(m Model) int64 {
	if p, err := modelPath(m); err == nil {
		if size, err := statSize(p); err == nil && size > 0 {
			return size
		}
	}
	return parseModelSize(m.Size)
}

// modelMemoryEstimate returns weights plus the KV cache at the given context
// size, for a model that may or may not be downloaded. ok=false when the
// shape metadata isn't readable, which is the case before download.
func modelMemoryEstimate(m Model, ctx int) (weights, kv int64, ok bool) {
	weights = modelSizeBytes(m)
	if weights <= 0 {
		return 0, 0, false
	}
	p, err := modelPath(m)
	if err != nil {
		return weights, 0, false
	}
	meta, err := readGGUFMetaCached(p)
	if err != nil || !meta.complete() {
		return weights, 0, false
	}
	return weights, meta.kvCacheBytes(ctx), true
}

// renderMemSegment formats the model server's resident memory for the
// header. Empty until a server is running, so the header stays quiet before
// the first message.
func renderMemSegment() string {
	rss, ok := serverMemory()
	if !ok || rss <= 0 {
		return ""
	}
	label := metaLabelStyle.Render("mem ")
	value := metaValueStyle.Render(formatBytes(rss))
	total, haveTotal := systemRAM()
	if !haveTotal || total <= 0 {
		return label + value
	}
	pct := int(float64(rss) / float64(total) * 100)
	style := sysStyle
	switch {
	case pct >= 80:
		style = lipgloss.NewStyle().Foreground(colErr).Bold(true)
	case pct >= 60:
		style = lipgloss.NewStyle().Foreground(colBusy).Bold(true)
	}
	return label + value + " " + style.Render(fmt.Sprintf("(%d%%)", pct))
}
