package main

import (
	"testing"
	"time"
)

// The model picker repaints on every arrow keypress, and each repaint asks
// for every model's resource note. When that read the GGUF header and
// shelled out for system RAM each time, one repaint took over four seconds
// and the picker was unusable.
//
// Thresholds are deliberately loose — this guards against a return to
// per-keypress file parsing, not against millisecond drift.
func TestPickerRepaintIsFast(t *testing.T) {
	repaint := func() time.Duration {
		start := time.Now()
		for _, m := range availableModels {
			_ = modelResourceNote(m)
		}
		return time.Since(start)
	}

	first := repaint() // may parse GGUF headers
	if first > 2*time.Second {
		t.Errorf("first repaint took %s — headers are being parsed too slowly", first)
	}

	// Everything after should be served from cache.
	var worst time.Duration
	for i := 0; i < 20; i++ {
		if d := repaint(); d > worst {
			worst = d
		}
	}
	if worst > 50*time.Millisecond {
		t.Errorf("cached repaint took %s — results are not being reused", worst)
	}
	t.Logf("first repaint %s, worst cached repaint %s", first.Round(time.Millisecond), worst.Round(time.Microsecond))
}

// The header redraws constantly, so its segments must not do real work.
func TestHeaderSegmentsAreCheap(t *testing.T) {
	start := time.Now()
	for i := 0; i < 100; i++ {
		_ = renderMemSegment()
		_ = renderCtxSegment()
	}
	if d := time.Since(start); d > 200*time.Millisecond {
		t.Errorf("100 header renders took %s — too slow for a per-frame path", d)
	}
}

// Parsing a header must not walk the tokenizer vocabulary.
func TestGGUFParseSkipsBulkMetadata(t *testing.T) {
	m, err := currentModel()
	if err != nil {
		t.Skip("no current model")
	}
	p, err := modelPath(m)
	if err != nil || !isModelDownloaded(m) {
		t.Skip("model not downloaded")
	}
	start := time.Now()
	if _, err := readGGUFMeta(p); err != nil {
		t.Fatal(err)
	}
	if d := time.Since(start); d > 300*time.Millisecond {
		t.Errorf("uncached header parse took %s — large arrays are being decoded", d)
	}
}

// A cached read must be dramatically cheaper than a cold one.
func TestGGUFCacheIsUsed(t *testing.T) {
	m, err := currentModel()
	if err != nil {
		t.Skip("no current model")
	}
	p, err := modelPath(m)
	if err != nil || !isModelDownloaded(m) {
		t.Skip("model not downloaded")
	}
	if _, err := readGGUFMetaCached(p); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	for i := 0; i < 500; i++ {
		if _, err := readGGUFMetaCached(p); err != nil {
			t.Fatal(err)
		}
	}
	if d := time.Since(start); d > 500*time.Millisecond {
		t.Errorf("500 cached reads took %s — the cache is not working", d)
	}
}
