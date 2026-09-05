package tui

import (
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"atlas.llm/internal/catalog"
	"atlas.llm/internal/config"
	"atlas.llm/internal/engine"
)

// Download progress: resolving what to fetch and metering it on the footer.

// progressToSysMsg returns a progress callback that forwards each status
// line into the bubbletea event loop as a sysMsg — so per-file progress
// renders as muted log lines in the viewport instead of leaking through
// as raw stdout writes and corrupting the alt-screen.
func progressToSysMsg() func(string) {
	return func(s string) {
		if program != nil {
			program.Send(sysMsg{content: s})
		}
	}
}

type downloadTargets struct {
	engine bool
	models []catalog.Model
}

// resolveDownloadTargets parses /download args:
//
//	(none)       -> engine + current model
//	engine       -> engine only
//	all          -> engine + every registered model
//	<model-name> -> engine + that specific model
func resolveDownloadTargets(args []string) (downloadTargets, error) {
	if len(args) == 0 {
		cfg, _ := config.LoadConfig()
		cur, ok := config.FindModel(cfg.CurrentModel)
		if !ok {
			return downloadTargets{}, fmt.Errorf("current model %q not in registry", cfg.CurrentModel)
		}
		return downloadTargets{engine: true, models: []catalog.Model{cur}}, nil
	}
	switch strings.ToLower(args[0]) {
	case "engine":
		return downloadTargets{engine: true}, nil
	case "all":
		return downloadTargets{engine: true, models: append([]catalog.Model(nil), catalog.AvailableModels...)}, nil
	default:
		m, ok := config.FindModel(args[0])
		if !ok {
			return downloadTargets{}, fmt.Errorf("unknown model: %s (try /list)", args[0])
		}
		return downloadTargets{engine: true, models: []catalog.Model{m}}, nil
	}
}

// applyDownloadProgress records a progress message into the download state
// the footer renders. A message for a different file restarts the meter —
// chained downloads (engine, then a model) must not inherit the previous
// file's start time or rate. Returns the progress-bar animation cmd, or nil.
func (m *chatModel) applyDownloadProgress(msg downloadProgressMsg) tea.Cmd {
	if msg.name != m.dlName {
		m.dlMeter = engine.DownloadMeter{}
	}
	m.dlMeter.Observe(time.Now(), msg.written)
	m.dlName = msg.name
	m.dlWritten = msg.written
	m.dlTotal = msg.total
	if msg.total > 0 {
		return m.progress.SetPercent(float64(msg.written) / float64(msg.total))
	}
	return nil
}

// throttledProgress returns a ProgressFn that forwards updates to the bubbletea
// program at most every 100ms (plus one final update at completion).
func throttledProgress(name string) engine.ProgressFn {
	var last time.Time
	return func(written, total int64) {
		done := total > 0 && written >= total
		if !done && time.Since(last) < 100*time.Millisecond {
			return
		}
		last = time.Now()
		if program != nil {
			program.Send(downloadProgressMsg{name: name, written: written, total: total})
		}
	}
}

func runDownloadAllCmd(t downloadTargets) tea.Cmd {
	return func() tea.Msg {
		var done []string
		if t.engine && engine.EngineNeedsDownload() {
			if err := engine.DownloadEngine(throttledProgress("engine")); err != nil {
				return downloadDoneMsg{what: strings.Join(done, ", "), err: fmt.Errorf("engine: %w", err)}
			}
			done = append(done, "engine")
		}
		for _, mm := range t.models {
			if config.IsModelDownloaded(mm) {
				continue
			}
			if err := engine.DownloadModel(mm, throttledProgress(mm.Name)); err != nil {
				return downloadDoneMsg{what: strings.Join(done, ", "), err: fmt.Errorf("%s: %w", mm.Name, err)}
			}
			done = append(done, mm.Name)
		}
		if len(done) == 0 {
			return downloadDoneMsg{what: "nothing (already present)"}
		}
		return downloadDoneMsg{what: strings.Join(done, ", ")}
	}
}
