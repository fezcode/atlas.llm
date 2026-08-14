package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

var Version = "dev"

const helpText = `atlas.llm - local AI chat + project context tooling

USAGE
  atlas.llm                    Launch interactive chat (TUI)
  atlas.llm [flags] [DIR]      Run a one-shot command
  atlas.llm -c PROMPT          One-shot chat — print reply and exit

FLAGS
  -h, --help           Show this help and exit.
  -v, --version        Print version and exit.
  --clear-logs         Delete the persistent TUI log file and exit.
  --reset-model        Switch to the lightest model and exit. atlas.llm loads
                       the selected model at startup, so quitting while a large
                       one is active can leave the next launch stuck loading it
                       with no way to reach /model. This is the way out.

  -c, --chat PROMPT    Send PROMPT to the local model and print the reply
                       to stdout, then exit. Pass "-" to read PROMPT from
                       stdin (e.g. 'git diff | atlas.llm -c -'). No history
                       is persisted between calls. Same dependency
                       requirement as --summarize/--grep.

  --serve              Host this machine's model on the network and block
                       until Ctrl+C. No TUI. Other machines then run
                       '/set endpoint IP:PORT' to use it, and need no engine
                       or model of their own. Prints the addresses to use.
        --bind ADDR    Interface to listen on (default 0.0.0.0, all).
                       Use 127.0.0.1 to keep it to this machine.
        --port N       Port to listen on (default 8080).
        --slots N      Concurrent conversations (default 4). Slots divide
                       the same KV memory, so more slots means less context
                       each, not more VRAM used.
        --api-key KEY  Require this bearer token. Off by default; without
                       it anyone who can reach the port can use the model.
                       A second port (--port + 1) serves /atlas/info, which
                       is how clients report what they are connected to.

  --summarize          Summarize every text file in DIR (default: .) and write
                       the result to SUMMARY.md in the target directory.
                       Uses the currently selected local model (see /model).
                       Skips .gitignored, binary, and oversized files.
                       Honors --exclude (comma-separated extensions) and
                       --max-size (bytes; default 262144 for summarize, which
                       overrides the grep default when --summarize is set).
                       REQUIRES the engine and model to already be present in
                       ~/.atlas/atlas.llm.data/ — start chat and run /download
                       first. Dependencies are never fetched automatically.

  --grep QUERY         Semantic grep — ask the local model to find lines in
                       DIR (default: .) that match QUERY (natural language or
                       code). Prints "path:line: snippet" for each hit. Same
                       dependency requirement as --summarize.
      --max-size       Skip files larger than this many bytes during --grep
                       (default 32768). Prevents Windows command-line overflow
                       on minified/generated files.

  --dump               Compile every text file in DIR (default: .) into a
                       single Markdown document. Respects .gitignore and
                       skips binary files. Good for pasting full project
                       context into a hosted LLM.
      -o, --output     Output path for --dump (default: project_context.md).
      --exclude        Comma-separated extra extensions to exclude from --dump
                       (e.g. .mp4,.exe). Always in addition to .gitignore.
      --with-summaries Include per-file AI summaries inline with --dump.

INTERACTIVE MODE
  Starting with no arguments opens a chat UI against the currently selected
  local model. Available slash commands inside chat:

    /help           Show in-app help.
    /list           List known models and download status.
    /model [name]   Show current model or switch to NAME (does NOT download).
    /download [arg] Download dependencies explicitly.
                      (no arg)         engine + current model
                      engine           engine only
                      <model-name>     engine + that model
                      all              engine + every registered model
    /summarize      Summarize current directory into SUMMARY.md.
    /grep <query>   Semantic grep across current directory.
    /clear          Clear the on-screen chat history.
    /quit, /exit    Leave chat (Ctrl+C also works).

DATA DIRECTORY
  All engine binaries, models, and the config file live under
  ~/.atlas/atlas.llm.data/:
    config.json            current model selection
    engine/                llama.cpp prebuilt binaries (llama-cli + libs)
    models/<name>.gguf     downloaded model weights

EXAMPLES
  atlas.llm
  atlas.llm -c "explain goroutines in one paragraph"
  atlas.llm -c - < prompt.txt
  atlas.llm --summarize
  atlas.llm --summarize ./src
  atlas.llm --dump -o context.md ./src
  atlas.llm --dump --exclude .mp4,.mp3 --with-summaries
  atlas.llm --grep "where we load the gitignore" ./src
`

// prewarm starts llama-server up-front for one-shot CLI commands so the
// user sees a single "loading model..." message instead of the first
// file's summary/search appearing to hang. Without this the warmup cost
// is paid inside the first chat-completion call, with no surrounding context.
func prewarm() error {
	fmt.Fprintln(os.Stderr, "Loading model...")
	start := time.Now()
	s, err := ensureServer()
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "Model %s ready in %s.\n", s.model.Name, time.Since(start).Round(time.Millisecond))
	return nil
}

// installSignalCleanup kills the llama-server subprocess on Ctrl+C /
// SIGTERM so it doesn't outlive the CLI. Covers the case where the TUI
// defer doesn't run (e.g. shell-level kill, Task Manager).
func installSignalCleanup() {
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigs
		closeActiveBrowser()
		shutdownServer()
		os.Exit(130)
	}()
}

func main() {
	installSignalCleanup()
	// One-shot commands (--summarize / --grep) lazily start llama-server on
	// the first inference call. Make sure we kill it before process exit so
	// a crashed or Ctrl+C'd CLI never orphans the backend.
	defer shutdownServer()
	// Same story for a browser the agent launched: never orphan it.
	defer closeActiveBrowser()
	var (
		versionFlag       bool
		helpFlag          bool
		summarizeFlag     bool
		dumpFlag          bool
		withSummariesFlag bool
		outputFlag        string
		excludeFlag       string
		grepFlag          string
		maxSizeFlag       int64
		clearLogsFlag     bool
		resetModelFlag    bool
		chatFlag          string
		chatFlagSet       bool
		serveFlag         bool
		bindFlag          string
		portFlag          int
		slotsFlag         int
		apiKeyFlag        string
	)

	flag.BoolVar(&versionFlag, "v", false, "")
	flag.BoolVar(&versionFlag, "version", false, "")
	flag.BoolVar(&helpFlag, "h", false, "")
	flag.BoolVar(&helpFlag, "help", false, "")
	flag.BoolVar(&summarizeFlag, "summarize", false, "")
	flag.BoolVar(&dumpFlag, "dump", false, "")
	flag.BoolVar(&withSummariesFlag, "with-summaries", false, "")
	flag.StringVar(&outputFlag, "o", "project_context.md", "")
	flag.StringVar(&outputFlag, "output", "project_context.md", "")
	flag.StringVar(&excludeFlag, "exclude", "", "")
	flag.StringVar(&grepFlag, "grep", "", "")
	flag.Int64Var(&maxSizeFlag, "max-size", DefaultGrepMaxSize, "")
	flag.BoolVar(&clearLogsFlag, "clear-logs", false, "")
	flag.BoolVar(&resetModelFlag, "reset-model", false, "")
	flag.StringVar(&chatFlag, "c", "", "")
	flag.StringVar(&chatFlag, "chat", "", "")
	flag.BoolVar(&serveFlag, "serve", false, "")
	flag.StringVar(&bindFlag, "bind", "0.0.0.0", "")
	flag.IntVar(&portFlag, "port", defaultServePort, "")
	flag.IntVar(&slotsFlag, "slots", defaultServeSlots, "")
	flag.StringVar(&apiKeyFlag, "api-key", "", "")

	flag.Usage = func() { fmt.Print(helpText) }
	flag.Parse()
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "c" || f.Name == "chat" {
			chatFlagSet = true
		}
	})

	if helpFlag {
		fmt.Print(helpText)
		return
	}
	if versionFlag {
		fmt.Printf("atlas.llm v%s\n", Version)
		return
	}

	if serveFlag {
		// Serving is a foreground process with no TUI, so its log goes to
		// the same file the TUI uses and errors go to stderr.
		_, closeLog, logErr := setupLogging()
		if logErr == nil {
			defer closeLog()
		}
		if err := runServe(serveOptions{
			Bind:   bindFlag,
			Port:   portFlag,
			Slots:  slotsFlag,
			APIKey: apiKeyFlag,
		}); err != nil {
			fmt.Fprintf(os.Stderr, "serve: %v\n", err)
			os.Exit(1)
		}
		return
	}

	if clearLogsFlag {
		if err := clearLogs(); err != nil {
			fmt.Fprintf(os.Stderr, "clear-logs: %v\n", err)
			os.Exit(1)
		}
		return
	}

	if resetModelFlag {
		if err := resetToLightestModel(); err != nil {
			fmt.Fprintf(os.Stderr, "reset-model: %v\n", err)
			os.Exit(1)
		}
		return
	}

	targetDir := "."
	if flag.NArg() > 0 {
		targetDir = flag.Arg(0)
	}

	modes := 0
	if summarizeFlag {
		modes++
	}
	if dumpFlag {
		modes++
	}
	if grepFlag != "" {
		modes++
	}
	if chatFlagSet {
		modes++
	}
	if modes > 1 {
		fmt.Fprintln(os.Stderr, "--summarize, --dump, --grep, and -c are mutually exclusive.")
		os.Exit(2)
	}

	switch {
	case chatFlagSet:
		prompt := chatFlag
		if prompt == "-" || prompt == "" {
			b, err := io.ReadAll(os.Stdin)
			if err != nil {
				fmt.Fprintf(os.Stderr, "chat: read stdin: %v\n", err)
				os.Exit(1)
			}
			prompt = string(b)
		}
		prompt = strings.TrimSpace(prompt)
		if prompt == "" {
			fmt.Fprintln(os.Stderr, "chat: empty prompt")
			os.Exit(2)
		}
		if err := prewarm(); err != nil {
			fmt.Fprintf(os.Stderr, "chat: %v\n", err)
			os.Exit(1)
		}
		reply, err := chat(context.Background(), nil, prompt)
		if err != nil {
			fmt.Fprintf(os.Stderr, "chat: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(reply)

	case grepFlag != "":
		if err := prewarm(); err != nil {
			fmt.Fprintf(os.Stderr, "grep: %v\n", err)
			os.Exit(1)
		}
		hits, err := grepDirectory(targetDir, grepFlag, maxSizeFlag, nil)
		if err != nil {
			fmt.Fprintf(os.Stderr, "grep: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(formatGrepHits(hits))

	case summarizeFlag:
		if err := prewarm(); err != nil {
			fmt.Fprintf(os.Stderr, "summarize: %v\n", err)
			os.Exit(1)
		}
		var excludes []string
		if excludeFlag != "" {
			excludes = strings.Split(excludeFlag, ",")
		}
		// --max-size on /summarize uses the summarize default if the user
		// didn't explicitly override it (maxSizeFlag defaults to the grep
		// constant, which is too strict for summaries).
		summarizeMax := int64(maxSizeFlag)
		if maxSizeFlag == DefaultGrepMaxSize {
			summarizeMax = DefaultSummarizeMaxSize
		}
		opts := SummarizeOptions{
			TargetDir: targetDir,
			Output:    "SUMMARY.md",
			MaxSize:   summarizeMax,
			Exclude:   excludes,
		}
		if err := summarizeDirectory(opts, nil); err != nil {
			fmt.Fprintf(os.Stderr, "summarize: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Summary written to %s\n", opts.Output)

	case dumpFlag:
		if withSummariesFlag {
			if err := prewarm(); err != nil {
				fmt.Fprintf(os.Stderr, "dump: %v\n", err)
				os.Exit(1)
			}
		}
		var excludes []string
		if excludeFlag != "" {
			excludes = strings.Split(excludeFlag, ",")
		}
		err := runDump(DumpOptions{
			TargetDir: targetDir,
			Output:    outputFlag,
			Exclude:   excludes,
			Summarize: withSummariesFlag,
		})
		if err != nil {
			fmt.Fprintf(os.Stderr, "dump: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Context written to %s\n", outputFlag)

	default:
		if err := startChat(); err != nil {
			fmt.Fprintf(os.Stderr, "chat: %v\n", err)
			os.Exit(1)
		}
	}
}
