package tui

import (
	"fmt"
	"strings"
	"time"

	"atlas.llm/internal/browser"
	"atlas.llm/internal/engine"
)

// /browser — the persistent browser profile, which is the only browser state
// that outlives the session. Everything else about the browser tools is the
// model's business; this is the user's: where the profile lives, how much of
// their disk it is using, and how to sign out of everything at once.

// handleBrowser runs /browser. No argument reports; `clear` deletes.
func (m *chatModel) handleBrowser(args []string) {
	if len(args) == 0 {
		m.pushSystem(renderBrowserProfiles())
		return
	}
	switch strings.ToLower(args[0]) {
	case "clear", "reset", "forget":
		families := args[1:]
		if len(families) == 0 {
			for _, p := range browser.PersistentProfiles() {
				families = append(families, p.Family)
			}
		}
		var done, failed []string
		for _, family := range families {
			if err := browser.ClearPersistentProfile(strings.ToLower(family)); err != nil {
				failed = append(failed, err.Error())
				continue
			}
			done = append(done, strings.ToLower(family))
		}
		var b strings.Builder
		if len(done) > 0 {
			fmt.Fprintf(&b, "Cleared the %s profile — every site it was signed into is signed out.",
				strings.Join(done, " and "))
		}
		for _, e := range failed {
			if b.Len() > 0 {
				b.WriteByte('\n')
			}
			b.WriteString(e)
		}
		m.pushSystem(b.String())
	default:
		m.pushError(fmt.Sprintf("unknown /browser subcommand %q — try `/browser` or `/browser clear [chrome|firefox]`", args[0]))
	}
}

// renderBrowserProfiles reports each persistent profile: where it is, how big
// it has grown, when it was last used, and whether a window has it open.
func renderBrowserProfiles() string {
	var b strings.Builder
	b.WriteString("PERSISTENT BROWSER PROFILES  (browser_open profile=\"persist\")\n")
	for _, p := range browser.PersistentProfiles() {
		fmt.Fprintf(&b, "\n  %s\n", p.Family)
		fmt.Fprintf(&b, "    %-10s %s\n", "path", p.Dir)
		if !p.Exists {
			fmt.Fprintf(&b, "    %-10s never used\n", "state")
			continue
		}
		state := "idle"
		if p.InUse {
			state = "open now — close the window before clearing it"
		}
		fmt.Fprintf(&b, "    %-10s %s\n", "state", state)
		fmt.Fprintf(&b, "    %-10s %s\n", "size", engine.FormatBytes(p.Size))
		if !p.Used.IsZero() {
			fmt.Fprintf(&b, "    %-10s %s\n", "last used", humanSince(p.Used))
		}
	}
	b.WriteString("\nCookies and logins accumulate here and are never deleted on close.\n")
	b.WriteString("`/browser clear` signs out of everything; `/browser clear chrome` just one.")
	return b.String()
}

// humanSince renders a timestamp the way a person reads it: how long ago, and
// the date once "ago" stops being useful.
func humanSince(t time.Time) string {
	d := time.Since(t)
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		return fmt.Sprintf("%d min ago", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%d hours ago", int(d.Hours()))
	case d < 7*24*time.Hour:
		return fmt.Sprintf("%d days ago", int(d.Hours()/24))
	}
	return t.Format("2006-01-02")
}
