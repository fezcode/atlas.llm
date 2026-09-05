package tui

import (
	"log"
	"os"
	"strings"
	"sync"

	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/glamour/styles"
	"github.com/muesli/termenv"
	"golang.org/x/term"
)

// renderMarkdown runs the assistant's reply through glamour so headings,
// lists, code fences, and inline styling render as ANSI. Falls back to the
// raw text if the renderer fails or the content is empty. The renderer is
// cached and rebuilt only when the viewport width changes.
func (m *chatModel) renderMarkdown(s string) string {
	if strings.TrimSpace(s) == "" {
		return s
	}
	width := m.viewport.Width
	if width <= 0 {
		width = 80
	}
	wrap := width - 4
	if wrap < 20 {
		wrap = 20
	}
	if m.mdRenderer == nil || m.mdWidth != wrap {
		r, err := glamour.NewTermRenderer(
			glamour.WithStandardStyle(markdownStyleName()),
			glamour.WithWordWrap(wrap),
		)
		if err != nil {
			return s
		}
		m.mdRenderer = r
		m.mdWidth = wrap
	}
	out, err := m.mdRenderer.Render(s)
	if err != nil {
		return s
	}
	return strings.Trim(out, "\n")
}

// Markdown style detection happens once, before bubbletea takes over stdin.
//
// glamour.WithAutoStyle() asks termenv whether the background is dark, and
// termenv answers by writing an OSC 11 query to the terminal and reading the
// reply back off stdin. Inside a running TUI, bubbletea owns stdin — so the
// terminal's answer ("\x1b]11;rgb:158e/193a/1e75\x1b\\") races between the two
// readers, and when bubbletea wins it lands in the transcript as text.
//
// The renderer is rebuilt whenever the wrap width changes, and markdown is
// rendered when a reply finishes, which is why the escape sequence appeared
// after a response rather than at startup.
var (
	mdStyleOnce sync.Once
	mdStyle     = styles.DarkStyle
)

// detectMarkdownStyle resolves the style while stdin is still ours. Call it
// before starting the bubbletea program.
//
// This mirrors what WithAutoStyle does internally — notty when stdout isn't a
// terminal, otherwise dark or light by background — with the one difference
// that matters: it runs at a moment when reading the terminal's reply is safe.
func detectMarkdownStyle() {
	mdStyleOnce.Do(func() {
		if !term.IsTerminal(int(os.Stdout.Fd())) {
			mdStyle = styles.NoTTYStyle
			return
		}
		if !termenv.HasDarkBackground() {
			mdStyle = styles.LightStyle
		}
		log.Printf("markdown style: %s", mdStyle)
	})
}

// markdownStyleName returns the resolved style.
//
// If detection never ran — a caller that renders markdown outside the TUI —
// it settles the question without a terminal round trip: IsTerminal is a
// local syscall, while the background query is the thing that must never
// happen off the startup path.
func markdownStyleName() string {
	mdStyleOnce.Do(func() {
		if !term.IsTerminal(int(os.Stdout.Fd())) {
			mdStyle = styles.NoTTYStyle
		}
	})
	return mdStyle
}
