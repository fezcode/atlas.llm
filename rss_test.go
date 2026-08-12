package main

import (
	"os"
	"testing"
	"time"
)

// tasklist formats its memory column with the system locale's grouping
// separator. Stripping one specific separator worked on exactly one locale
// and mis-parsed the other: splitting "12,345 K" on commas left "345 K" as
// the last field, which read as 345 KB instead of 12 MB — wrong by 1000x
// and silent about it.
func TestParseTasklistKB(t *testing.T) {
	cases := []struct {
		field string
		want  int64
	}{
		{"12,345 K", 12345 * 1024},      // en-US grouping
		{"8.174.328 K", 8174328 * 1024}, // dot grouping
		{"8 174 328 K", 8174328 * 1024}, // space grouping
		{"5746304 K", 5746304 * 1024},   // no grouping
		{" 1 234 K ", 1234 * 1024},
	}
	for _, tc := range cases {
		got, ok := parseTasklistKB(tc.field)
		if !ok {
			t.Errorf("parseTasklistKB(%q) failed", tc.field)
			continue
		}
		if got != tc.want {
			t.Errorf("parseTasklistKB(%q) = %d, want %d", tc.field, got, tc.want)
		}
	}
	// tasklist reports N/A for processes it can't inspect; that must not
	// become a zero-byte reading the header would render as "0 B".
	for _, field := range []string{"N/A", "", "   ", "K"} {
		if got, ok := parseTasklistKB(field); ok {
			t.Errorf("parseTasklistKB(%q) = %d, true; want failure", field, got)
		}
	}
}

// The header re-renders on every keystroke, so reading memory must never
// spawn a process on the render path — that cost ~180ms per frame and was
// the whole of the input lag.
func TestProcessRSSCachedDoesNotBlockRenders(t *testing.T) {
	resetRSSCache()
	t.Cleanup(resetRSSCache)

	pid := os.Getpid()

	// The first call may miss, but must return immediately either way.
	start := time.Now()
	for i := 0; i < 200; i++ {
		_, _ = processRSSCached(pid)
	}
	elapsed := time.Since(start)
	if elapsed > 50*time.Millisecond {
		t.Errorf("200 cached reads took %s — the render path is still probing", elapsed)
	}
	t.Logf("200 cached reads in %s", elapsed)
}

// Once the background refresh lands, the value must actually be served —
// a cache that never populates is just a silent removal of the feature.
func TestProcessRSSCachedPopulates(t *testing.T) {
	resetRSSCache()
	t.Cleanup(resetRSSCache)

	pid := os.Getpid()
	_, _ = processRSSCached(pid) // kicks off the async probe

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if rss, ok := processRSSCached(pid); ok {
			if rss <= 0 {
				t.Fatalf("cached RSS = %d, want a positive size", rss)
			}
			t.Logf("own RSS resolved to %d bytes", rss)
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Error("background RSS probe never populated the cache")
}

// Switching models restarts llama-server under a new PID. Until the new
// reading lands the gauge must report nothing rather than the dead
// process's memory.
func TestProcessRSSCachedRejectsStalePID(t *testing.T) {
	resetRSSCache()
	t.Cleanup(resetRSSCache)

	pid := os.Getpid()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if _, ok := processRSSCached(pid); ok {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if _, ok := processRSSCached(pid); !ok {
		t.Skip("could not resolve own RSS on this platform")
	}
	if _, ok := processRSSCached(pid + 999999); ok {
		t.Error("a different PID served the cached process's memory")
	}
}
