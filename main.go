package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
)

var Version = "dev"

const helpText = `atlas.ai - local AI chat + project context tooling

USAGE
  atlas.ai                    Launch interactive chat (TUI)
  atlas.ai [flags] [DIR]      Run a one-shot command

FLAGS
  -h, --help           Show this help and exit.
  -v, --version        Print version and exit.

  --summarize          Summarize every text file in DIR (default: .) and write
                       the result to SUMMARY.md in the target directory.
                       Uses the currently selected local model (see /model).
                       REQUIRES the engine and model to already be present in
                       ~/.atlas/atlas.ai.data/ — start chat and run /download
                       first. Dependencies are never fetched automatically.

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
    /clear          Clear the on-screen chat history.
    /quit, /exit    Leave chat (Ctrl+C also works).

DATA DIRECTORY
  All engine binaries, models, and the config file live under
  ~/.atlas/atlas.ai.data/:
    config.json            current model selection
    llamafile[.exe]        inference engine
    models/<name>.gguf     downloaded model weights

EXAMPLES
  atlas.ai
  atlas.ai --summarize
  atlas.ai --summarize ./src
  atlas.ai --dump -o context.md ./src
  atlas.ai --dump --exclude .mp4,.mp3 --with-summaries
`

func main() {
	var (
		versionFlag       bool
		helpFlag          bool
		summarizeFlag     bool
		dumpFlag          bool
		withSummariesFlag bool
		outputFlag        string
		excludeFlag       string
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

	flag.Usage = func() { fmt.Print(helpText) }
	flag.Parse()

	if helpFlag {
		fmt.Print(helpText)
		return
	}
	if versionFlag {
		fmt.Printf("atlas.ai v%s\n", Version)
		return
	}

	targetDir := "."
	if flag.NArg() > 0 {
		targetDir = flag.Arg(0)
	}

	switch {
	case summarizeFlag && dumpFlag:
		fmt.Fprintln(os.Stderr, "--summarize and --dump cannot be combined; use --dump --with-summaries to embed summaries in a dump.")
		os.Exit(2)

	case summarizeFlag:
		out := "SUMMARY.md"
		if err := summarizeDirectory(targetDir, out, nil); err != nil {
			fmt.Fprintf(os.Stderr, "summarize: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Summary written to %s\n", out)

	case dumpFlag:
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
