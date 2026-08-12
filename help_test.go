package main

import (
	"strings"
	"testing"
)

// Every slash command the TUI dispatches must have a help topic, or /help
// silently under-documents the tool.
func TestEverySlashCommandIsDocumented(t *testing.T) {
	for _, c := range slashCommands {
		if _, ok := findHelpTopic(c); !ok {
			t.Errorf("%s has no help topic", c)
		}
	}
}

func TestFindHelpTopicAliasesAndSlashes(t *testing.T) {
	for _, in := range []string{"mcp", "/mcp", "MCP", " mcp "} {
		if _, ok := findHelpTopic(in); !ok {
			t.Errorf("findHelpTopic(%q) failed", in)
		}
	}
	// /exit is an alias of /quit.
	q, ok := findHelpTopic("exit")
	if !ok || q.Name != "quit" {
		t.Errorf("alias lookup gave %q, %v; want quit", q.Name, ok)
	}
	if _, ok := findHelpTopic("nope"); ok {
		t.Error("unknown topic resolved")
	}
}

// Topics must be complete enough to be worth printing.
func TestHelpTopicsAreWellFormed(t *testing.T) {
	seen := map[string]bool{}
	for _, tp := range helpTopics {
		if seen[tp.Name] {
			t.Errorf("duplicate help topic %q", tp.Name)
		}
		seen[tp.Name] = true

		if tp.Summary == "" || len(tp.Usage) == 0 || tp.Detail == "" {
			t.Errorf("%s is missing a summary, usage, or detail", tp.Name)
		}
		for _, u := range tp.Usage {
			if !strings.HasPrefix(u, "/") {
				t.Errorf("%s usage %q should start with a slash", tp.Name, u)
			}
		}
		for _, s := range tp.Subcommands {
			if s.Name == "" || s.Usage == "" || s.Detail == "" {
				t.Errorf("%s subcommand %+v is incomplete", tp.Name, s)
			}
			if !strings.HasPrefix(s.Usage, "/"+tp.Name) {
				t.Errorf("%s subcommand usage %q should start with /%s", tp.Name, s.Usage, tp.Name)
			}
		}
		// A cross-reference to a topic that doesn't exist is a dead end.
		for _, ref := range tp.SeeAlso {
			if _, ok := findHelpTopic(ref); !ok {
				t.Errorf("%s references unknown topic %q", tp.Name, ref)
			}
		}
	}
}

// The subcommands the dispatcher accepts should be the ones documented.
func TestSubcommandsMatchImplementation(t *testing.T) {
	cases := map[string][]string{
		"mcp":   {"add", "catalog", "remove", "trust", "env", "connect", "disconnect", "tools", "logout"},
		"tools": {"on", "off", "list"},
		"set":   {"max_tokens", "gpu_layers", "engine_variant"},
	}
	for cmd, want := range cases {
		tp, ok := findHelpTopic(cmd)
		if !ok {
			t.Fatalf("no topic for %s", cmd)
		}
		for _, w := range want {
			if _, ok := tp.findSub(w); !ok {
				t.Errorf("/%s %s is implemented but undocumented", cmd, w)
			}
		}
	}
}

func TestRenderHelpTopicIncludesEverySection(t *testing.T) {
	tp, _ := findHelpTopic("mcp")
	out := renderHelpTopic(tp, 100)
	for _, want := range []string{"USAGE", "SUBCOMMANDS", "EXAMPLES", "NOTES", "SEE ALSO", "/mcp"} {
		if !strings.Contains(out, want) {
			t.Errorf("rendered /help mcp is missing %q", want)
		}
	}
	// A topic with no subcommands must not print an empty section header.
	plain, _ := findHelpTopic("clear")
	if strings.Contains(renderHelpTopic(plain, 100), "SUBCOMMANDS") {
		t.Error("/help clear printed an empty SUBCOMMANDS section")
	}
}

func TestRenderHelpSub(t *testing.T) {
	tp, _ := findHelpTopic("set")
	s, ok := tp.findSub("gpu_layers")
	if !ok {
		t.Fatal("gpu_layers subcommand missing")
	}
	out := renderHelpSub(tp, s, 100)
	if !strings.Contains(out, "gpu_layers") || !strings.Contains(out, "/help set") {
		t.Errorf("rendered subcommand help looks wrong:\n%s", out)
	}
}

// Rendering must not panic or produce runaway lines at extreme widths.
func TestRenderHelpTopicHandlesNarrowTerminals(t *testing.T) {
	for _, w := range []int{0, 1, 20, 40, 200, 500} {
		for _, tp := range helpTopics {
			out := renderHelpTopic(tp, w)
			if out == "" {
				t.Errorf("%s rendered empty at width %d", tp.Name, w)
			}
		}
	}
}

func TestWrapText(t *testing.T) {
	got := wrapText("alpha beta gamma delta", 11)
	if len(got) < 2 {
		t.Errorf("expected wrapping, got %q", got)
	}
	for _, l := range got {
		if len(l) > 20 {
			t.Errorf("line exceeds wrap width: %q", l)
		}
	}
	// Blank lines separate paragraphs and must survive.
	para := wrapText("one\n\ntwo", 40)
	if len(para) != 3 || para[1] != "" {
		t.Errorf("paragraph break lost: %q", para)
	}
}

func TestHelpSubNamesSkipsPlaceholders(t *testing.T) {
	// /download documents a "<model>" placeholder, which isn't completable.
	for _, n := range helpSubNames("download") {
		if strings.HasPrefix(n, "<") {
			t.Errorf("placeholder %q leaked into completions", n)
		}
	}
	if len(helpSubNames("clear")) != 0 {
		t.Error("expected no subcommands for /clear")
	}
}

// /help tools states how many built-ins there are. A number written out in
// prose goes stale the moment one is added, and nobody notices for a year.
func TestToolsHelpCountMatchesRegistry(t *testing.T) {
	tp, ok := findHelpTopic("tools")
	if !ok {
		t.Fatal("no tools help topic")
	}
	words := map[int]string{
		5: "Five", 6: "Six", 7: "Seven", 8: "Eight",
		9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve",
	}
	want, ok := words[len(toolRegistry)]
	if !ok {
		t.Fatalf("no number word for %d tools — extend the table", len(toolRegistry))
	}
	if !strings.Contains(tp.Detail, want+" built-in tools") {
		t.Errorf("tools help should say %q built-in tools; the registry has %d",
			want, len(toolRegistry))
	}
}

// A tool the model can call but the help never mentions is one the user
// cannot find out about.
func TestToolsHelpNamesEveryBuiltin(t *testing.T) {
	tp, ok := findHelpTopic("tools")
	if !ok {
		t.Fatal("no tools help topic")
	}
	for name := range toolRegistry {
		if !strings.Contains(tp.Detail, name) {
			t.Errorf("%s is missing from /help tools", name)
		}
	}
}
