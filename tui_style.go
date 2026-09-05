package main

import (
	"log"
	"os"
	"strings"
	"sync"

	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/glamour/styles"
	"github.com/charmbracelet/lipgloss"
	"github.com/muesli/termenv"
	"golang.org/x/term"
)

// The palette, the pill badges, and markdown rendering — every visual
// constant the rest of the TUI draws with.

// Palette. Hex values degrade gracefully to the nearest 256-color on
// terminals without truecolor support.
var (
	colAccent    = lipgloss.Color("#A78BFA") // violet — brand accent
	colUser      = lipgloss.Color("#38BDF8") // sky — user messages
	colAssistant = lipgloss.Color("#34D399") // emerald — assistant messages
	colMuted     = lipgloss.Color("#9CA3AF") // gray — system/footer
	colDim       = lipgloss.Color("#4B5563") // slate — rules, separators
	colErr       = lipgloss.Color("#F87171") // red
	colBusy      = lipgloss.Color("#FBBF24") // amber
)

var (
	// Pill-style role badges — colored-on-dark backgrounds with one-char padding.
	userPillStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#0B1220")).
			Background(colUser).
			Bold(true).
			Padding(0, 1)
	assistantPillStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#0B1220")).
				Background(colAssistant).
				Bold(true).
				Padding(0, 1)
	errPillStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#0B1220")).
			Background(colErr).
			Bold(true).
			Padding(0, 1)

	sysStyle = lipgloss.NewStyle().Foreground(colMuted).Italic(true)
	// thinkStyle dims the reasoning text a show_thinking transcript carries,
	// so the eye separates it from the reply without a border.
	thinkStyle = lipgloss.NewStyle().Foreground(colMuted).Faint(true)
	errTextStyle = lipgloss.NewStyle().Foreground(colErr)

	// Top bar: accent-colored brand + muted meta, with a thin underline rule.
	brandStyle     = lipgloss.NewStyle().Foreground(colAccent).Bold(true)
	metaLabelStyle = lipgloss.NewStyle().Foreground(colDim)
	metaValueStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#E5E7EB"))
	sepStyle       = lipgloss.NewStyle().Foreground(colDim)
	busyStyle      = lipgloss.NewStyle().Foreground(colBusy).Bold(true)

	footerStyle    = lipgloss.NewStyle().Foreground(colMuted)
	footerKeyStyle = lipgloss.NewStyle().Foreground(colAccent).Bold(true)

	ruleStyle = lipgloss.NewStyle().Foreground(colDim)
)

// ruleMarker is a placeholder stored in m.rendered wherever a full-width
// separator belongs. It is expanded at render time rather than baked in, so
// the rules re-fit when the terminal is resized.
const ruleMarker = "\x00rule\x00"

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
