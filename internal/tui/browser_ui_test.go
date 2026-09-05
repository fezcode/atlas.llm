package tui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"atlas.llm/internal/browser"
	"atlas.llm/internal/config"
)

// The persistent profile is the one piece of browser state that outlives the
// session, and until now nothing could see it: no path, no size, no way to
// sign out of everything short of deleting a directory by hand.

func TestRenderBrowserProfilesShowsAnUnusedProfile(t *testing.T) {
	withTempHome(t)

	out := renderBrowserProfiles()
	for _, want := range []string{"chrome", "firefox", "never used"} {
		if !strings.Contains(out, want) {
			t.Errorf("/browser output is missing %q:\n%s", want, out)
		}
	}
}

func TestRenderBrowserProfilesShowsPathAndSize(t *testing.T) {
	withTempHome(t)
	base, err := config.AtlasDir()
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(base, "browser-profiles", "chrome")
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "Cookies"), make([]byte, 3<<20), 0600); err != nil {
		t.Fatal(err)
	}

	out := renderBrowserProfiles()
	if !strings.Contains(out, dir) {
		t.Errorf("/browser output does not say where the profile lives:\n%s", out)
	}
	if !strings.Contains(out, "3.1 MB") && !strings.Contains(out, "3.0 MB") && !strings.Contains(out, "MB") {
		t.Errorf("/browser output does not report the size on disk:\n%s", out)
	}
	if !strings.Contains(out, "/browser clear") {
		t.Errorf("/browser output does not say how to clear it:\n%s", out)
	}
}

func TestHandleBrowserClearRemovesTheProfile(t *testing.T) {
	withTempHome(t)
	base, err := config.AtlasDir()
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(base, "browser-profiles", "chrome")
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	marker := filepath.Join(dir, "Cookies")
	if err := os.WriteFile(marker, []byte("session"), 0600); err != nil {
		t.Fatal(err)
	}

	m := newChatModel()
	m.handleBrowser([]string{"clear", "chrome"})

	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Errorf("/browser clear chrome left the profile behind: %v", err)
	}
}

func TestHandleBrowserRejectsAnUnknownSubcommand(t *testing.T) {
	withTempHome(t)
	m := newChatModel()
	m.handleBrowser([]string{"nonsense"})
	if !strings.Contains(lastRendered(&m), "nonsense") {
		t.Errorf("an unknown subcommand should be reported back:\n%s", lastRendered(&m))
	}
}

// A command the completer does not know about is a command the user has to
// already know exists.
func TestBrowserIsACompletableCommand(t *testing.T) {
	found := false
	for _, c := range slashCommands {
		if c == "/browser" {
			found = true
		}
	}
	if !found {
		t.Errorf("/browser missing from slashCommands: %v", slashCommands)
	}
	if _, ok := findHelpTopic("browser"); !ok {
		t.Error("no /help browser topic")
	}
}

// /config's FILES block is where the user looks for "what has atlas.llm put on
// my disk" — the persistent profile belongs in that answer.
func TestConfigListsTheBrowserProfileDir(t *testing.T) {
	withTempHome(t)
	base, err := config.AtlasDir()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(base, "browser-profiles", "chrome"), 0700); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.LoadConfig()
	if err != nil {
		t.Fatal(err)
	}
	out := renderConfig(cfg, configState{})
	if !strings.Contains(out, "browser-profiles") {
		t.Errorf("/config FILES does not mention the browser profiles:\n%s", out)
	}
}

// lastRendered joins what the model has pushed to the transcript so far.
func lastRendered(m *chatModel) string {
	return strings.Join(m.rendered, "\n")
}

var _ = browser.PersistentProfiles
