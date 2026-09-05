package ui

import (
	"github.com/charmbracelet/lipgloss"
)

// The palette, the pill badges, and markdown rendering — every visual
// constant the rest of the TUI draws with.

// Palette. Hex values degrade gracefully to the nearest 256-color on
// terminals without truecolor support.
var (
	ColAccent    = lipgloss.Color("#A78BFA") // violet — brand accent
	colUser      = lipgloss.Color("#38BDF8") // sky — user messages
	ColAssistant = lipgloss.Color("#34D399") // emerald — assistant messages
	ColMuted     = lipgloss.Color("#9CA3AF") // gray — system/footer
	ColDim       = lipgloss.Color("#4B5563") // slate — rules, separators
	ColErr       = lipgloss.Color("#F87171") // red
	ColBusy      = lipgloss.Color("#FBBF24") // amber
)

var (
	// Pill-style role badges — colored-on-dark backgrounds with one-char padding.
	UserPillStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#0B1220")).
			Background(colUser).
			Bold(true).
			Padding(0, 1)
	AssistantPillStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#0B1220")).
				Background(ColAssistant).
				Bold(true).
				Padding(0, 1)
	ErrPillStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#0B1220")).
			Background(ColErr).
			Bold(true).
			Padding(0, 1)

	SysStyle = lipgloss.NewStyle().Foreground(ColMuted).Italic(true)
	// thinkStyle dims the reasoning text a show_thinking transcript carries,
	// so the eye separates it from the reply without a border.
	ThinkStyle = lipgloss.NewStyle().Foreground(ColMuted).Faint(true)
	ErrTextStyle = lipgloss.NewStyle().Foreground(ColErr)

	// Top bar: accent-colored brand + muted meta, with a thin underline rule.
	BrandStyle     = lipgloss.NewStyle().Foreground(ColAccent).Bold(true)
	MetaLabelStyle = lipgloss.NewStyle().Foreground(ColDim)
	MetaValueStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#E5E7EB"))
	SepStyle       = lipgloss.NewStyle().Foreground(ColDim)
	BusyStyle      = lipgloss.NewStyle().Foreground(ColBusy).Bold(true)

	FooterStyle    = lipgloss.NewStyle().Foreground(ColMuted)
	FooterKeyStyle = lipgloss.NewStyle().Foreground(ColAccent).Bold(true)

	RuleStyle = lipgloss.NewStyle().Foreground(ColDim)
)

// ruleMarker is a placeholder stored in m.rendered wherever a full-width
// separator belongs. It is expanded at render time rather than baked in, so
// the rules re-fit when the terminal is resized.
const RuleMarker = "\x00rule\x00"
