package tui

import (
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"

	"atlas.llm/internal/engine"
)

// The meter's speed is a smoothed bytes-per-second: steady input reads
// steady, and it must be fed by explicit timestamps so it can be tested
// without sleeping.
func TestDownloadMeterSpeed(t *testing.T) {
	t0 := time.Date(2026, 8, 16, 12, 0, 0, 0, time.UTC)
	const mib = 1024 * 1024

	var d engine.DownloadMeter
	if d.Speed() != 0 {
		t.Errorf("fresh meter reports speed %v, want 0", d.Speed())
	}
	d.Observe(t0, 0)
	d.Observe(t0.Add(1*time.Second), 10*mib)
	if got := d.Speed(); got != 10*mib {
		t.Errorf("first measured speed = %.0f B/s, want %d", got, 10*mib)
	}
	// A steady stream keeps a steady reading.
	d.Observe(t0.Add(2*time.Second), 20*mib)
	if got := d.Speed(); got != 10*mib {
		t.Errorf("steady speed drifted to %.0f B/s, want %d", got, 10*mib)
	}
	// A stalled second pulls the reading down — the display must not stay
	// pinned at the last good rate while nothing arrives.
	d.Observe(t0.Add(3*time.Second), 20*mib)
	if got := d.Speed(); got >= 10*mib || got <= 0 {
		t.Errorf("stalled speed = %.0f B/s, want between 0 and %d", got, 10*mib)
	}
}

func TestDownloadMeterElapsed(t *testing.T) {
	t0 := time.Date(2026, 8, 16, 12, 0, 0, 0, time.UTC)
	var d engine.DownloadMeter
	if d.Elapsed(t0) != 0 {
		t.Error("fresh meter reports nonzero elapsed")
	}
	d.Observe(t0, 0)
	if got := d.Elapsed(t0.Add(90 * time.Second)); got != 90*time.Second {
		t.Errorf("elapsed = %v, want 90s", got)
	}
}

func TestFormatSpeed(t *testing.T) {
	cases := []struct {
		bps  float64
		want string
	}{
		{42.5 * 1024 * 1024, "42.5 MB/s"},
		{500, "500 B/s"},
		{0, "0 B/s"},
	}
	for _, c := range cases {
		if got := engine.FormatSpeed(c.bps); got != c.want {
			t.Errorf("formatSpeed(%v) = %q, want %q", c.bps, got, c.want)
		}
	}
}

func TestFormatElapsed(t *testing.T) {
	cases := []struct {
		d    time.Duration
		want string
	}{
		{0, "0:00"},
		{42 * time.Second, "0:42"},
		{3*time.Minute + 7*time.Second, "3:07"},
		{1*time.Hour + 2*time.Minute + 33*time.Second, "1:02:33"},
	}
	for _, c := range cases {
		if got := engine.FormatElapsed(c.d); got != c.want {
			t.Errorf("formatElapsed(%v) = %q, want %q", c.d, got, c.want)
		}
	}
}

// The download footer must show how fast bytes arrive and for how long the
// download has been running, alongside the existing bar and byte counts.
func TestDownloadFooterShowsSpeedAndElapsed(t *testing.T) {
	m := &chatModel{
		textarea: textarea.New(),
		viewport: viewport.New(80, 20),
		progress: progress.New(),
		busy:     true, busyReason: "downloading",
		dlName: "qwen3.8-27b-heretic", dlWritten: 4 << 30, dlTotal: 13 << 30,
	}
	m.dlMeter = engine.DownloadMeter{
		Start: time.Now().Add(-90200 * time.Millisecond),
		Rate:  42.5 * 1024 * 1024,
	}
	got := m.renderFooter(120)
	for _, want := range []string{"42.5 MB/s", "1:30"} {
		if !strings.Contains(got, want) {
			t.Errorf("download footer missing %q:\n%s", want, got)
		}
	}

	// Unknown total (no Content-Length): no bar, but speed and elapsed stay.
	m.dlTotal = 0
	got = m.renderFooter(120)
	if !strings.Contains(got, "42.5 MB/s") || !strings.Contains(got, "1:30") {
		t.Errorf("no-total footer lost speed or elapsed:\n%s", got)
	}
}

// A progress message for a new file must restart the meter — chained
// downloads (engine then model) would otherwise inherit the previous file's
// start time and rate.
func TestDownloadMeterResetsPerFile(t *testing.T) {
	m := &chatModel{
		textarea: textarea.New(),
		viewport: viewport.New(80, 20),
		progress: progress.New(),
	}
	m.dlName = "engine"
	m.dlMeter = engine.DownloadMeter{Start: time.Now().Add(-time.Hour), Rate: 1e9}
	m.applyDownloadProgress(downloadProgressMsg{name: "gemma-3-1b-it", written: 1024, total: 4096})
	if m.dlMeter.Rate >= 1e9 {
		t.Error("meter kept the previous file's rate after the name changed")
	}
	if age := time.Since(m.dlMeter.Start); age > time.Minute {
		t.Errorf("meter kept the previous file's start time (%v ago)", age)
	}
}
