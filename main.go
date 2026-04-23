package main

import (
	"flag"
	"fmt"
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

FLAGS
  -h, --help           Show this help and exit.
  -v, --version        Print version and exit.
  --clear-logs         Delete the persistent TUI log file and exit.

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

	flag.Usage = func() { fmt.Print(helpText) }
	flag.Parse()

	if helpFlag {
		fmt.Print(helpText)
		return
	}
	if versionFlag {
		fmt.Printf("atlas.llm v%s\n", Version)
		return
	}

	if clearLogsFlag {
		if err := clearLogs(); err != nil {
			fmt.Fprintf(os.Stderr, "clear-logs: %v\n", err)
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
	if modes > 1 {
		fmt.Fprintln(os.Stderr, "--summarize, --dump, and --grep are mutually exclusive.")
		os.Exit(2)
	}

	switch {
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
