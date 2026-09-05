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
