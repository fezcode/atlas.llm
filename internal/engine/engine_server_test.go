package engine

import (
	"errors"
	"runtime"
	"testing"

	"atlas.llm/internal/catalog"
)

// TestLaunchClassifiesEarlyExit ensures a llama-server that dies before its
// /health endpoint comes up — the signature of a rejected flag — surfaces
// errServerExitedEarly, since startLlamaServer's retry with base flags keys
// on exactly that classification. A binary rejecting `-fa on` is simulated
// with /usr/bin/false: starts fine, exits nonzero immediately.
func TestLaunchClassifiesEarlyExit(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("needs a portable exits-immediately binary; the classification logic is platform-independent")
	}
	m := catalog.Model{Name: "fake", Filename: "fake.gguf"}
	_, err := launchLlamaServer("/usr/bin/false", "/nonexistent/fake.gguf", m, 0, 4096, true)
	if err == nil {
		t.Fatal("expected an error from a server that exits immediately")
	}
	if !errors.Is(err, errServerExitedEarly) {
		t.Fatalf("early exit not classified as errServerExitedEarly: %v", err)
	}
}

// Dropping the KV cache only means something when a server is already up.
// EnsureServer does not mean "the server that is running" — it starts one —
// so /reset and compaction go through this instead. Getting that wrong booted
// the engine and loaded a model purely to erase a cache that did not exist,
// and under `go test` left the subprocess behind with nothing to stop it.
func TestDropKVCacheIfRunningStartsNothing(t *testing.T) {
	ShutdownServer() // whatever an earlier test may have left behind
	DropKVCacheIfRunning()
	serverMu.Lock()
	defer serverMu.Unlock()
	if activeServer != nil {
		t.Fatal("DropKVCacheIfRunning started a server; with none running it must do nothing")
	}
}
