package main

import (
	"fmt"
	"sort"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// helpSub documents one subcommand, reachable as `/help <cmd> <sub>`.
type helpSub struct {
	Name   string
	Usage  string
	Detail string
}

// helpTopic is the long-form documentation for one slash command. The
// compact table in welcomeText() says what a command is; this says how it
// behaves, what it costs, and where it bites.
type helpTopic struct {
	Name    string
	Aliases []string
	Summary string
	Usage   []string
	// Detail is prose. Blank lines separate paragraphs.
	Detail      string
	Subcommands []helpSub
	Examples    [][2]string // command, what it does
	Notes       []string
	SeeAlso     []string
}

var helpTopics = []helpTopic{
	{
		Name:    "help",
		Summary: "Show the command overview, or detailed help for one command.",
		Usage:   []string{"/help", "/help <command>", "/help <command> <subcommand>"},
		Detail: "With no arguments, prints the grouped command overview plus a " +
			"Performance block reflecting what this machine can actually do.\n\n" +
			"With a command name, prints everything about that command: every " +
			"form it accepts, what it changes on disk, and the failure modes " +
			"worth knowing. With a subcommand too, narrows to just that.",
		Examples: [][2]string{
			{"/help mcp", "everything about MCP servers"},
			{"/help mcp add", "just the add subcommand"},
			{"/help set", "all settings and their effects"},
		},
	},
	{
		Name:    "list",
		Summary: "List known models and whether they're downloaded.",
		Usage:   []string{"/list"},
		Detail: "Prints the model registry with a download indicator, the " +
			"approximate on-disk size, and which model is currently selected. " +
			"Also shows whether the inference engine itself has been downloaded.\n\n" +
			"Nothing here is fetched automatically — a model showing " +
			"\"not downloaded\" needs an explicit /download.",
		SeeAlso: []string{"model", "download"},
	},
	{
		Name:    "model",
		Summary: "Switch the active model, via a picker or by name.",
		Usage:   []string{"/model", "/model <name>"},
		Detail: "With no argument, opens a picker: ↑/↓ to move, Enter to select, " +
			"Esc to cancel. The currently active model is marked and preselected.\n\n" +
			"Switching is instant and persists to config.json, but it does not " +
			"download anything. If the model you pick isn't on disk, you'll be " +
			"told to run /download for it.\n\n" +
			"The model server restarts on your next message, not immediately — " +
			"so the first reply after a switch pays the model-load cost.",
		Examples: [][2]string{
			{"/model", "open the picker"},
			{"/model qwen3.5-4b", "lightest model that can drive /tools and /mcp"},
			{"/model qwen3.5-9b", "switch directly by name"},
		},
		Notes: []string{
			"Tool-calling needs a model trained for it. Qwen3.5 (4B and up) and " +
				"Ministral-3 (8B and up) handle it; Gemma 3 will usually ignore " +
				"tools or invent a fake call, which makes /tools and /mcp look " +
				"broken. qwen3.5-4b is the lightest that works; " +
				"ministral-3-8b-instruct sits between it and qwen3.5-9b.",
		},
		SeeAlso: []string{"list", "download", "tools"},
	},
	{
		Name:    "download",
		Summary: "Fetch the inference engine and model weights.",
		Usage: []string{
			"/download", "/download engine", "/download <model>", "/download all",
		},
		Detail: "Nothing is downloaded automatically — this is the only thing " +
			"that pulls the engine or model weights.\n\n" +
			"The engine is the latest llama.cpp release build for your OS and " +
			"architecture, resolved from GitHub at download time, so you get " +
			"current binaries rather than a pinned version. Which archive is " +
			"fetched depends on `/set engine_variant` (CPU vs a GPU build).\n\n" +
			"Re-running is cheap: an engine or model already present is skipped. " +
			"The exception is an engine_variant change, which wipes the engine " +
			"directory and re-downloads, since mixing build variants would leave " +
			"the wrong binaries behind.",
		Subcommands: []helpSub{
			{Name: "engine", Usage: "/download engine",
				Detail: "Just the llama.cpp engine. Also the way to apply an " +
					"engine_variant change."},
			{Name: "<model>", Usage: "/download qwen3.5-9b",
				Detail: "The engine plus that model. Does not switch to it — use /model."},
			{Name: "all", Usage: "/download all",
				Detail: "The engine plus every model in the registry. Tens of gigabytes."},
		},
		Notes: []string{
			"Downloads run in the background with a progress bar; the TUI stays usable.",
			"CUDA is a two-archive install (engine + ~391MB CUDA runtime), so it's " +
				"markedly larger than the other variants.",
		},
		SeeAlso: []string{"list", "model", "set"},
	},
	{
		Name:    "summarize",
		Summary: "Write a per-file summary of a directory to SUMMARY.md.",
		Usage:   []string{"/summarize [dir]", "/summarize [dir] --max-size=N --exclude=.ext,..."},
		Detail: "Walks the directory, asks the local model for a 1–3 sentence " +
			"summary of every text file, and writes SUMMARY.md there. " +
			".gitignore is respected.\n\n" +
			"Output contains only the generated summaries, never raw file " +
			"contents — it's for orientation, not for feeding to another model. " +
			"Use --dump on the command line if you want full contents.\n\n" +
			"Runtime scales with file count: every file is a separate inference " +
			"call, so a large repo on a small model still takes a while.",
		Examples: [][2]string{
			{"/summarize", "the current directory"},
			{"/summarize ./src", "one subtree"},
			{"/summarize --max-size=131072", "include larger files than the default"},
			{"/summarize --exclude=.min.js,.lock", "skip noisy generated files"},
		},
		SeeAlso: []string{"grep"},
	},
	{
		Name:    "grep",
		Summary: "Semantic search: find lines by intent, not by regex.",
		Usage:   []string{"/grep <query>"},
		Detail: "Walks the current directory and asks the model which lines match " +
			"a natural-language description, printing path:line: snippet for each " +
			"hit. .gitignore is respected.\n\n" +
			"The point is queries regex can't express — \"retry logic with " +
			"backoff\", \"where we load the gitignore\". For literal tokens, " +
			"ordinary grep is faster and exact.\n\n" +
			"Accuracy tracks model quality; small models miss hits and invent " +
			"others. This is different from the `grep` *tool* available under " +
			"/tools, which is a real RE2 regex search the model can call.",
		Examples: [][2]string{
			{"/grep where we parse config", "find config parsing"},
			{"/grep retry logic with backoff", "find retry handling"},
		},
		SeeAlso: []string{"summarize", "tools"},
	},
	{
		Name:    "set",
		Summary: "Show or change persistent settings.",
		Usage:   []string{"/set", "/set <key>", "/set <key> <value>"},
		Detail: "With no arguments, lists every setting and its current value. " +
			"With a key, shows just that one. With a key and value, validates and " +
			"persists to config.json.\n\n" +
			"Settings survive restarts.",
		Subcommands: []helpSub{
			{Name: "max_tokens", Usage: "/set max_tokens 4096",
				Detail: "Cap on reply length in tokens. Default 4096. The ceiling is " +
					"three quarters of ctx_size, since the rest of the window is " +
					"needed for the prompt and history — so raising ctx_size raises " +
					"this too.\n\n" +
					"If replies are being cut off mid-sentence, raise this. If a " +
					"reasoning model returns nothing at all, this is usually why: " +
					"the whole budget went on internal thinking."},
			{Name: "reasoning", Usage: "/set reasoning on|off",
				Detail: "Reasoning models such as Qwen3.5 write an internal <think> " +
					"block before answering. You never see it, but you wait for it.\n\n" +
					"Measured on qwen3.5-4b, \"in one sentence, what is a KV cache?\": " +
					"with thinking the first word appeared at 62.1s and the reply " +
					"finished at 63.9s. Without it, 0.3s and 3.2s — a twentyfold " +
					"difference on a question that needed none of it.\n\n" +
					"The default is auto, which splits the two: off for chat, on for " +
					"tool-driven turns, where thinking measurably improves which tool " +
					"gets called. `on` and `off` apply to everything.\n\n" +
					"One-shot commands — /compact, /summarize, /grep — never think " +
					"regardless, because a think there is pure latency. On qwen3.5-4b " +
					"it once consumed the whole token budget and returned nothing.\n\n" +
					"Models without a reasoning mode, such as Gemma, are unaffected."},
			{Name: "max_tool_rounds", Usage: "/set max_tool_rounds N|off",
				Detail: "How many times the model may call tools while answering a " +
					"single message. Default 40; `off` removes the cap entirely.\n\n" +
					"Work that reads a dozen files or chains several MCP calls can " +
					"legitimately need more than the default. Raise it, or switch it " +
					"off, when the agent is stopping mid-task on real work.\n\n" +
					"Before raising it, look at the trace. Hitting the cap usually " +
					"means the model got stuck retrying something that failed — a " +
					"wrong path being the classic case — rather than doing many " +
					"useful things. Raising the limit then just buys more retries.\n\n" +
					"Turning it off is safer than it sounds: a model repeating the " +
					"identical call is stopped after 4 attempts regardless, and esc " +
					"stops a turn at any point."},
			{Name: "ctx_size", Usage: "/set ctx_size auto|N",
				Detail: "The context window, in tokens — how much conversation, tool " +
					"output, and file content the model can see at once. Default " +
					"16384.\n\n" +
					"This is not fixed by the model, but it is capped by it. Every " +
					"GGUF records the context length it was trained for, and going " +
					"beyond that degrades quality rather than extending memory, so " +
					"/set reads that number and refuses anything larger. `/set` with " +
					"no arguments shows both the value in use and the model's " +
					"ceiling.\n\n" +
					"The ceilings differ a lot. Qwen3.5 was trained for 262144 " +
					"tokens; Gemma 3 for 32768. atlas.llm also imposes its own " +
					"131072 limit, because the KV cache at the top of Qwen's range " +
					"would be tens of gigabytes.\n\n" +
					"Cost is memory: the KV cache grows roughly linearly with this, " +
					"so doubling the window roughly doubles what the model server " +
					"holds beyond the weights. Raise it when tool output or long " +
					"files keep filling the window; leave it alone if memory is " +
					"tight — /compact is the cheaper answer to a full context.\n\n" +
					"Changing it restarts the model server, so it applies from your " +
					"next message."},
			{Name: "gpu_layers", Usage: "/set gpu_layers auto|0|N",
				Detail: "How many model layers to offload to the GPU (llama.cpp's " +
					"-ngl). `auto` (the default) offloads everything when the " +
					"installed engine has a GPU backend, and stays on CPU otherwise. " +
					"`0` forces CPU-only. A number offloads that many layers, which " +
					"is how you fit a model that would otherwise exceed VRAM.\n\n" +
					"Changing this restarts the model server, so it applies on your " +
					"next message.\n\n" +
					"On Windows and Linux this does nothing until a GPU build is " +
					"installed — see `/help set engine_variant`."},
			{Name: "engine_variant", Usage: "/set engine_variant auto|cpu|vulkan|cuda|hip",
				Detail: "Which llama.cpp build to download. This is what decides " +
					"whether your GPU can be used at all — gpu_layers only has an " +
					"effect once a GPU-capable build is installed.\n\n" +
					"Changing it takes two steps, because the setting selects an " +
					"archive and the archive still has to be fetched:\n" +
					"  /set engine_variant cuda\n" +
					"  /download engine\n" +
					"The second command replaces the installed engine: the engine " +
					"directory is wiped and the new build extracted, so the two " +
					"variants can't leave mixed binaries behind. `/set` with no " +
					"arguments shows both what you selected and what is actually " +
					"installed, which is how you confirm the swap happened.\n\n" +
					"Which one to pick:\n" +
					"  auto    default; cpu everywhere, which on macOS already\n" +
					"          includes Metal\n" +
					"  cpu     no GPU. Smallest download, works anywhere\n" +
					"  vulkan  NVIDIA, AMD, or Intel on Windows and Linux. ~35MB,\n" +
					"          the easiest GPU option and usually the right first try\n" +
					"  cuda    NVIDIA on Windows. Usually the fastest, but ~640MB\n" +
					"          because the CUDA runtime ships as a second archive\n" +
					"  hip     AMD Radeon on Windows. ~325MB\n\n" +
					"macOS needs none of this. The macOS archive has always carried " +
					"the Metal backend, so `auto` already runs on the GPU and the " +
					"only GPU-capable value here is cpu.\n\n" +
					"Only variants llama.cpp actually publishes for your platform " +
					"are accepted — asking for cuda on macOS is rejected up front " +
					"with the list of what is available, rather than failing later " +
					"during the download.\n\n" +
					"A GPU build needs a working driver. atlas.llm will not pick one " +
					"for you on Windows or Linux, because a Vulkan or CUDA build " +
					"without the matching driver fails at load and there is no " +
					"reliable way to detect one first. If a GPU build misbehaves, " +
					"`/set engine_variant cpu` then `/download engine` puts you back."},
		},
		Notes: []string{
			"`auto` never selects a GPU build on Windows or Linux. A GPU build " +
				"without a matching driver fails at load and there's no reliable way " +
				"to detect one, so that choice is left to you.",
		},
		SeeAlso: []string{"download"},
	},
	{
		Name:    "tools",
		Summary: "Let the model read, edit, and run things in your project.",
		Usage:   []string{"/tools", "/tools on", "/tools off", "/tools list"},
		Detail: "Off by default. When on, the model can call tools instead of " +
			"guessing — reading files, searching, editing, running commands — and " +
			"loops until it has what it needs before answering.\n\n" +
			"Seven built-in tools. read_file, list_dir, and grep are read-only " +
			"and run silently. write_file, edit_file, multi_edit, and run_cmd are " +
			"destructive and open a confirmation modal first: Enter approves, Esc " +
			"denies. A denial is fed back to the model as a tool error so it " +
			"adapts instead of retrying the same call.\n\n" +
			"Edits are partial: edit_file replaces one unique string and " +
			"multi_edit applies a batch of them atomically, both leaving the rest " +
			"of the file untouched. Only write_file rewrites a whole file.\n\n" +
			"This switch also governs MCP tools. /mcp manages connections; /tools " +
			"decides whether the model may call anything at all.",
		Subcommands: []helpSub{
			{Name: "on", Usage: "/tools on", Detail: "Enable tool-use. Persists across restarts."},
			{Name: "off", Usage: "/tools off",
				Detail: "Disable it, and drop the current agent message list."},
			{Name: "list", Usage: "/tools list",
				Detail: "Show the built-in tools and which need confirmation."},
		},
		Notes: []string{
			"Bounded at 20 tool-call rounds per message, so a confused model " +
				"can't loop forever.",
			"/yesman skips every confirmation for the current session if the " +
				"prompting gets in the way — read `/help yesman` first.",
			"Needs a model that can emit tool calls — Qwen3.5-9B or " +
				"Ministral-3-14B. On Gemma 3 the feature will appear broken.",
			"/reset clears the agent's message list along with the conversation.",
		},
		SeeAlso: []string{"mcp", "model", "yesman"},
	},
	{
		Name:    "mcp",
		Summary: "Connect external tool servers — Slack, Confluence, GitHub, and more.",
		Usage: []string{
			"/mcp", "/mcp add [name [KEY=VALUE ...]]", "/mcp catalog",
			"/mcp remove <name>", "/mcp trust <name> [on|off]",
			"/mcp env <name> KEY=VALUE", "/mcp connect [name]",
			"/mcp disconnect <name>", "/mcp tools", "/mcp logout <name>",
		},
		Detail: "atlas.llm is an MCP client: it connects to Model Context Protocol " +
			"servers and hands their tools to the model through the same loop and " +
			"confirmation modal as the built-ins. That's how it reaches Slack, " +
			"Confluence, GitHub, a database — anything with an MCP server.\n\n" +
			"Start with /mcp add and pick from the list. You don't need to write " +
			"a config file; the commands maintain mcp.json for you.\n\n" +
			"Two kinds of server. Local ones run as a subprocess over stdio " +
			"(most published servers, including Slack's). Remote ones are hosted " +
			"and reached over HTTP, usually behind OAuth — /mcp add opens your " +
			"browser to authorize, and the tokens are stored so later runs " +
			"reconnect on their own.\n\n" +
			"Tools are namespaced server__tool, so two servers both exposing " +
			"`search` don't collide.",
		Subcommands: []helpSub{
			{Name: "add", Usage: "/mcp add [name [KEY=VALUE ...]]",
				Detail: "With no arguments, opens a picker of ready-made servers. " +
					"Picking one writes mcp.json and connects it.\n\n" +
					"Servers needing a token or a path can't be finished from a list, " +
					"so those pre-fill the command in the input box — replace the " +
					"placeholder and press Enter.\n\n" +
					"For anything not in the catalog:\n" +
					"  /mcp add NAME -- npx -y some-mcp-package\n" +
					"  /mcp add NAME --url=https://host/mcp --oauth\n\n" +
					"Flags: --oauth (browser authorization), --sse (older 2024-11-05 " +
					"protocol), --trust (skip confirmation)."},
			{Name: "catalog", Usage: "/mcp catalog",
				Detail: "List the built-in servers, what each needs, and the exact " +
					"command to add it."},
			{Name: "remove", Usage: "/mcp remove <name>",
				Detail: "Delete the server from mcp.json, drop its tools, and clear " +
					"any stored OAuth credentials for it."},
			{Name: "trust", Usage: "/mcp trust <name> [on|off]",
				Detail: "Trusted servers run their tools without asking. Untrusted " +
					"ones — the default — confirm every call.\n\n" +
					"Trust is per server and set by you. A server's own readOnlyHint " +
					"annotations are deliberately ignored, since those come from the " +
					"third party being gated.\n\n" +
					"Reconnects the server, because trust is baked into each tool " +
					"when it's registered."},
			{Name: "env", Usage: "/mcp env <name> KEY=VALUE",
				Detail: "Set or rotate a token without reopening the config file. " +
					"Reconnects afterwards."},
			{Name: "connect", Usage: "/mcp connect [name]",
				Detail: "Connect every enabled server, or just one. Needed for OAuth " +
					"servers on first use, since those are skipped at startup rather " +
					"than launching a browser unprompted."},
			{Name: "disconnect", Usage: "/mcp disconnect <name>",
				Detail: "Drop a server and its tools for this session. The config " +
					"entry stays; use remove to delete it."},
			{Name: "tools", Usage: "/mcp tools",
				Detail: "List the tools connected servers are contributing, and " +
					"whether each will ask for confirmation."},
			{Name: "logout", Usage: "/mcp logout <name>",
				Detail: "Forget a server's stored OAuth tokens. The next connect " +
					"re-authorizes."},
		},
		Examples: [][2]string{
			{"/mcp add", "pick a server from the list"},
			{"/mcp add deepwiki", "no-auth remote server — quickest way to test"},
			{"/mcp add slack SLACK_BOT_TOKEN=xoxb-... SLACK_TEAM_ID=T...", "Slack"},
			{"/mcp add atlassian", "Confluence + Jira, opens a browser"},
			{"/mcp trust deepwiki on", "stop confirming every call"},
		},
		Notes: []string{
			"/tools on is also required — /mcp only manages connections.",
			"At startup every enabled server reconnects automatically, except " +
				"OAuth servers with no stored credentials.",
			"OAuth tokens live in mcp-auth.json with 0600 permissions. That's a " +
				"protected file, not an OS keychain — comparable to ~/.aws/credentials.",
		},
		SeeAlso: []string{"tools", "model"},
	},
	{
		Name:    "config",
		Summary: "Show the whole current setup in one place.",
		Usage:   []string{"/config"},
		Detail: "Prints every persisted setting with its current value, the active " +
			"model and whether it's downloaded, the installed engine build, the " +
			"session-only state (tool-use, yesman, MCP connections), how much " +
			"memory the model server is using, and where everything lives on " +
			"disk.\n\n" +
			"Read-only — `/set <key> <value>` changes settings, `/mcp` manages " +
			"servers. This is the place to start when behaviour is surprising, " +
			"since it shows session state that config.json doesn't record.",
		Notes: []string{
			"The memory figure is measured from the running model server, not " +
				"predicted. It reads as \"not running\" until the first message, " +
				"since the server starts lazily.",
		},
		SeeAlso: []string{"set", "mcp", "tools"},
	},
	{
		Name:    "yesman",
		Summary: "Auto-approve destructive tools for this session only.",
		Usage:   []string{"/yesman", "/yesman on", "/yesman off"},
		Detail: "Normally write_file, edit_file, multi_edit, run_cmd, and any tool " +
			"from an untrusted MCP server open a confirmation modal before they " +
			"run. /yesman skips all of those prompts, so an agent turn proceeds " +
			"end to end without stopping for you.\n\n" +
			"With no argument it toggles; `on` and `off` set it explicitly.\n\n" +
			"This is genuinely dangerous. run_cmd executes arbitrary shell " +
			"commands, and MCP tools reach outside your machine entirely — a " +
			"single mistaken call can delete files, force-push, or post to a " +
			"Slack channel, with nothing asking first. The confirm modal is the " +
			"only thing standing between a model's misunderstanding and your " +
			"filesystem.\n\n" +
			"It is deliberately session-scoped and never written to config.json: " +
			"a toggle you flipped days ago should not silently arm tomorrow's " +
			"session. Quitting always resets it to off.",
		Notes: []string{
			"While it's on, a red ⚠ yesman marker sits in the header and the " +
				"footer, and every auto-approved call is still printed in the " +
				"trace as \"(auto-approved by /yesman)\" — it is never silent.",
			"esc still stops a turn mid-flight, which is the way out if a run " +
				"starts doing something you didn't intend.",
			"For a narrower version of the same idea, `/mcp trust NAME on` " +
				"exempts one MCP server you've vetted, and that one does persist.",
		},
		SeeAlso: []string{"tools", "mcp"},
	},
	{
		Name:    "compact",
		Summary: "Summarize older turns to free up the context window.",
		Usage:   []string{"/compact"},
		Detail: "Folds the earlier part of the conversation into a dense factual " +
			"summary and continues from there, keeping the most recent turns " +
			"verbatim. Use it when replies start failing because the context is " +
			"full, instead of /reset — which throws the conversation away " +
			"entirely.\n\n" +
			"The summary is produced by the local model and re-injected as " +
			"working context, so it keeps decisions, file paths, values, and " +
			"open threads rather than reading as prose.\n\n" +
			"Tool results are the usual reason context fills up: a single MCP " +
			"or file-read result can be thousands of tokens, and they accumulate " +
			"across a turn. /compact clears that out while keeping what it " +
			"established.",
		Notes: []string{
			"The most recent turns are always kept exactly as they were — " +
				"compaction never touches what you're actively working on.",
			"The model server's KV cache is dropped afterwards, since the " +
				"rewritten history no longer matches the cached prefix.",
			"Summarizing costs one inference call, so it takes a moment on a " +
				"large model.",
		},
		SeeAlso: []string{"reset", "clear", "set"},
	},
	{
		Name:    "clear",
		Summary: "Clear the screen, keeping the conversation.",
		Usage:   []string{"/clear"},
		Detail: "Wipes the visible scrollback only. The model still remembers the " +
			"conversation — use /reset to actually drop it.",
		SeeAlso: []string{"reset"},
	},
	{
		Name:    "reset",
		Summary: "Drop the conversation context and the server's cache.",
		Usage:   []string{"/reset"},
		Detail: "Clears the conversation, the agent's tool-call history, the " +
			"on-screen scrollback, the token-usage counter, and the model " +
			"server's KV cache.\n\n" +
			"Use it when earlier turns are dragging replies off course. If you " +
			"only need room in the context window, /compact keeps the " +
			"conversation by summarizing it instead of discarding it.\n\n" +
			"Settings, models, and MCP connections are unaffected.",
		SeeAlso: []string{"clear", "compact"},
	},
	{
		Name:    "quit",
		Aliases: []string{"exit"},
		Summary: "Leave chat.",
		Usage:   []string{"/quit", "/exit"},
		Detail: "Shuts down the model server and any local MCP subprocesses, so " +
			"nothing outlives the session. Ctrl+C does the same.",
	},
}

func findHelpTopic(name string) (helpTopic, bool) {
	name = strings.ToLower(strings.TrimPrefix(strings.TrimSpace(name), "/"))
	for _, t := range helpTopics {
		if t.Name == name {
			return t, true
		}
		for _, a := range t.Aliases {
			if a == name {
				return t, true
			}
		}
	}
	return helpTopic{}, false
}

func (t helpTopic) findSub(name string) (helpSub, bool) {
	name = strings.ToLower(strings.TrimSpace(name))
	for _, s := range t.Subcommands {
		if strings.ToLower(s.Name) == name {
			return s, true
		}
	}
	return helpSub{}, false
}

// helpTopicNames lists every documented command, for tab completion and the
// "unknown topic" message.
func helpTopicNames() []string {
	out := make([]string, 0, len(helpTopics))
	for _, t := range helpTopics {
		out = append(out, t.Name)
	}
	sort.Strings(out)
	return out
}

// helpSubNames lists a command's subcommands, for second-level completion.
func helpSubNames(cmd string) []string {
	t, ok := findHelpTopic(cmd)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(t.Subcommands))
	for _, s := range t.Subcommands {
		// Placeholders like "<model>" aren't literal completions.
		if strings.HasPrefix(s.Name, "<") {
			continue
		}
		out = append(out, s.Name)
	}
	sort.Strings(out)
	return out
}

// wrapText hard-wraps a paragraph at width, preserving blank-line breaks.
func wrapText(s string, width int) []string {
	if width < 20 {
		width = 20
	}
	var out []string
	for _, para := range strings.Split(s, "\n\n") {
		var line strings.Builder
		flush := func() {
			if line.Len() > 0 {
				out = append(out, line.String())
				line.Reset()
			}
		}
		// A leading space marks a line as pre-formatted — command examples
		// must not be reflowed into the surrounding prose.
		for _, raw := range strings.Split(para, "\n") {
			if strings.HasPrefix(raw, " ") || strings.HasPrefix(raw, "\t") {
				flush()
				out = append(out, strings.TrimRight(raw, " \t"))
				continue
			}
			for _, word := range strings.Fields(raw) {
				if line.Len() > 0 && line.Len()+1+len(word) > width {
					flush()
				}
				if line.Len() > 0 {
					line.WriteByte(' ')
				}
				line.WriteString(word)
			}
		}
		flush()
		out = append(out, "")
	}
	// Drop the trailing blank.
	if n := len(out); n > 0 && out[n-1] == "" {
		out = out[:n-1]
	}
	return out
}

// renderHelpTopic formats one command's full documentation.
func renderHelpTopic(t helpTopic, width int) string {
	body := helpBodyWidth(width)
	head := lipgloss.NewStyle().Foreground(colDim).Bold(true)
	cmd := lipgloss.NewStyle().Foreground(colAccent).Bold(true)
	dim := lipgloss.NewStyle().Foreground(colMuted)

	title := "/" + t.Name
	if len(t.Aliases) > 0 {
		for _, a := range t.Aliases {
			title += ", /" + a
		}
	}

	lines := []string{brandStyle.Render(title) + "  " + dim.Render(t.Summary), ""}

	if len(t.Usage) > 0 {
		lines = append(lines, head.Render("  USAGE"))
		for _, u := range t.Usage {
			lines = append(lines, "    "+cmd.Render(u))
		}
		lines = append(lines, "")
	}

	if t.Detail != "" {
		for _, l := range wrapText(t.Detail, body) {
			lines = append(lines, "  "+l)
		}
		lines = append(lines, "")
	}

	if len(t.Subcommands) > 0 {
		lines = append(lines, head.Render("  SUBCOMMANDS"))
		for _, s := range t.Subcommands {
			lines = append(lines, "    "+cmd.Render(s.Usage))
			for _, l := range wrapText(s.Detail, body-4) {
				if l == "" {
					lines = append(lines, "")
					continue
				}
				lines = append(lines, "      "+dim.Render(l))
			}
			lines = append(lines, "")
		}
	}

	if len(t.Examples) > 0 {
		lines = append(lines, head.Render("  EXAMPLES"))
		w := 0
		for _, e := range t.Examples {
			if lipgloss.Width(e[0]) > w {
				w = lipgloss.Width(e[0])
			}
		}
		for _, e := range t.Examples {
			pad := strings.Repeat(" ", w-lipgloss.Width(e[0]))
			lines = append(lines, "    "+cmd.Render(e[0])+pad+"   "+dim.Render(e[1]))
		}
		lines = append(lines, "")
	}

	if len(t.Notes) > 0 {
		lines = append(lines, head.Render("  NOTES"))
		for _, n := range t.Notes {
			wrapped := wrapText(n, body-2)
			for i, l := range wrapped {
				marker := "    • "
				if i > 0 {
					marker = "      "
				}
				lines = append(lines, marker+dim.Render(l))
			}
		}
		lines = append(lines, "")
	}

	if len(t.SeeAlso) > 0 {
		var refs []string
		for _, r := range t.SeeAlso {
			refs = append(refs, "/help "+r)
		}
		lines = append(lines, head.Render("  SEE ALSO")+"  "+dim.Render(strings.Join(refs, " · ")))
	}

	return strings.TrimRight(strings.Join(lines, "\n"), "\n")
}

// renderHelpSub formats a single subcommand.
func renderHelpSub(t helpTopic, s helpSub, width int) string {
	body := helpBodyWidth(width)
	cmd := lipgloss.NewStyle().Foreground(colAccent).Bold(true)
	dim := lipgloss.NewStyle().Foreground(colMuted)

	lines := []string{
		brandStyle.Render("/"+t.Name+" "+s.Name) + "  " + dim.Render("subcommand of /"+t.Name),
		"",
		"    " + cmd.Render(s.Usage),
		"",
	}
	for _, l := range wrapText(s.Detail, body) {
		lines = append(lines, "  "+l)
	}
	lines = append(lines, "", lipgloss.NewStyle().Foreground(colDim).Bold(true).
		Render("  SEE ALSO")+"  "+dim.Render("/help "+t.Name))
	return strings.Join(lines, "\n")
}

// helpBodyWidth keeps prose readable on very wide terminals.
func helpBodyWidth(width int) int {
	body := width - 6
	if body > 84 {
		body = 84
	}
	if body < 40 {
		body = 40
	}
	return body
}

// helpIndexText is the "what can I ask about" footer shown under /help.
func helpIndexText() string {
	dim := lipgloss.NewStyle().Foreground(colMuted)
	acc := lipgloss.NewStyle().Foreground(colAccent).Bold(true)
	return dim.Render("  Detailed help: ") + acc.Render("/help <command>") +
		dim.Render("  e.g. ") + acc.Render("/help mcp") + dim.Render(" · ") +
		acc.Render("/help set gpu_layers") + "\n" +
		dim.Render("  Documented: ") + dim.Render(strings.Join(helpTopicNames(), ", "))
}

// handleHelp implements `/help`, `/help <cmd>`, and `/help <cmd> <sub>`.
func (m *chatModel) handleHelp(args []string) {
	if len(args) == 0 {
		m.rendered = append(m.rendered, welcomeText(), helpIndexText())
		m.refresh()
		return
	}
	t, ok := findHelpTopic(args[0])
	if !ok {
		m.pushError(fmt.Sprintf("no help for %q. Documented commands: %s",
			args[0], strings.Join(helpTopicNames(), ", ")))
		return
	}
	if len(args) == 1 {
		m.rendered = append(m.rendered, renderHelpTopic(t, m.width))
		m.refresh()
		return
	}
	s, ok := t.findSub(args[1])
	if !ok {
		subs := helpSubNames(t.Name)
		if len(subs) == 0 {
			m.pushError(fmt.Sprintf("/%s has no subcommands — try `/help %s`", t.Name, t.Name))
			return
		}
		m.pushError(fmt.Sprintf("/%s has no subcommand %q. Try: %s",
			t.Name, args[1], strings.Join(subs, ", ")))
		return
	}
	m.rendered = append(m.rendered, renderHelpSub(t, s, m.width))
	m.refresh()
}
