package tui

import (
	"os"
	"strings"
	"testing"

	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/glamour/styles"
)

// The style has to be a real glamour style, or every markdown render silently
// falls back to unstyled text.
//
// Under `go test` stdout is not a terminal, so notty is the correct answer
// here and the coloured styles are what a real session resolves to.
func TestMarkdownStyleIsUsable(t *testing.T) {
	name := markdownStyleName()
	switch name {
	case styles.DarkStyle, styles.LightStyle, styles.NoTTYStyle:
	default:
		t.Fatalf("markdownStyleName() = %q, want dark, light or notty", name)
	}
	// Whatever it resolved to must never be "auto": that is the value that
	// defers the background query to render time, which is the bug.
	if name == styles.AutoStyle {
		t.Fatal("style resolved to auto, which queries the terminal at render time")
	}
	r, err := glamour.NewTermRenderer(
		glamour.WithStandardStyle(name),
		glamour.WithWordWrap(60),
	)
	if err != nil {
		t.Fatalf("renderer rejected style %q: %v", name, err)
	}
	out, err := r.Render("# hi\n\nsome **bold** text\n")
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}
	if !strings.Contains(out, "hi") {
		t.Errorf("rendered output lost the content: %q", out)
	}
}

// Without detectMarkdownStyle() having run, the getter must still answer
// rather than block or query — a renderer built on a code path that skipped
// startup detection must not reach for the terminal.
func TestMarkdownStyleDefaultsWithoutDetection(t *testing.T) {
	if got := markdownStyleName(); got == "" {
		t.Error("markdownStyleName() returned empty; dark is the safe default")
	}
}

// glamour.WithAutoStyle asks termenv for the background colour, and termenv
// answers by writing an OSC 11 query and reading the reply off stdin. Inside
// the TUI bubbletea owns stdin, so the terminal's answer —
// "\x1b]11;rgb:158e/193a/1e75\x1b\\" — raced the two readers and got rendered
// into the transcript as text.
//
// This is a source check because the failure only appears against a real
// terminal: with stdin not a TTY, as under `go test`, termenv answers from a
// default and nothing leaks. A runtime test would pass while the bug shipped.
func TestTUIDoesNotAutoStyleMarkdown(t *testing.T) {
	src, err := os.ReadFile("tui.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, line := range strings.Split(string(src), "\n") {
		code, _, _ := strings.Cut(line, "//")
		if strings.Contains(code, "WithAutoStyle") {
			t.Errorf("tui.go calls WithAutoStyle: %s\n"+
				"It queries the terminal over stdin, which bubbletea owns. Resolve the "+
				"style once via detectMarkdownStyle() before the program starts and pass "+
				"WithStandardStyle(markdownStyleName()).", strings.TrimSpace(line))
		}
	}
}
