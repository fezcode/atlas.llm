
# atlas.llm

![Banner](banner-image.png)

A local AI coding companion in a single Go binary. Opens an interactive chat
TUI by default — or, in one shot, summarizes a directory or compiles it into
a single Markdown context file for hosted LLMs. Inference runs fully on-device
via [`llamafile`](https://github.com/Mozilla-Ocho/llamafile); model weights
and the engine are fetched on demand via an explicit `/download` command.

## Modes

### 1. Interactive chat (default)

```powershell
atlas.llm
```

Launches a terminal UI (bubbletea) with the currently selected local model.
Dependencies (engine + model) are **not** downloaded automatically — run
`/download` inside chat to fetch them. Sending a message or running
`/summarize` while they are missing returns an error with the command to run.

**Slash commands inside chat:**

| Command           | What it does                                                        |
| ----------------- | ------------------------------------------------------------------- |
| `/help`           | Show in-app help.                                                   |
| `/list`           | List known models and their download status (`*` = current).        |
| `/model`          | Show the current model.                                             |
| `/model <name>`   | Switch to `<name>` (does **not** download — use `/download`).       |
| `/download`       | Download engine + current model.                                    |
| `/download engine`| Download only the inference engine.                                 |
| `/download <name>`| Download engine + the named model (does not switch to it).          |
| `/download all`   | Download engine + every model in the registry.                      |
| `/summarize`      | Summarize the current directory into `SUMMARY.md`.                  |
| `/clear`          | Clear on-screen chat history.                                       |
| `/quit`, `/exit`  | Leave chat (Ctrl+C also works).                                     |

Keys: `Enter` sends, `Shift+Enter` newline, `Ctrl+C` quits.

### 2. `--summarize` — project summary to SUMMARY.md

Walks the target directory (default: `.`), generates a 1-3 sentence summary
for every text file using the currently selected local model, and writes the
result to `SUMMARY.md` in that directory. Respects `.gitignore`.

```powershell
atlas.llm --summarize
atlas.llm --summarize ./src
```

This is the one-shot equivalent of running `/summarize` inside chat. It does
**not** include raw file contents — only the AI-generated summaries.

### 3. `--dump` — full project context to Markdown

Compiles every text file under the target directory into a single Markdown
document, with syntax-highlighted fenced code blocks. Intended for pasting
into hosted LLMs (Claude, Gemini, ChatGPT). Respects `.gitignore` and skips
binary files automatically.

```powershell
atlas.llm --dump
atlas.llm --dump -o context.md ./src
atlas.llm --dump --exclude .mp4,.mp3
atlas.llm --dump --with-summaries        # inline AI summaries per file
```

| Flag              | Default               | Purpose                                              |
| ----------------- | --------------------- | ---------------------------------------------------- |
| `-o`, `--output`  | `project_context.md`  | Output path.                                         |
| `--exclude`       | —                     | Comma-separated extra extensions to exclude.         |
| `--with-summaries`| off                   | Prepend each file's content with an AI summary block.|

## Top-level flags

| Flag               | Purpose                          |
| ------------------ | -------------------------------- |
| `-h`, `--help`     | Show help.                       |
| `-v`, `--version`  | Print version.                   |
| `--summarize`      | Run summary-to-`SUMMARY.md` mode.|
| `--dump`           | Run directory-to-Markdown mode.  |

## Data directory

All downloaded artifacts and the config file live under
`~/.atlas/atlas.llm.data/`:

```
~/.atlas/atlas.llm.data/
├── config.json           # { "current_model": "gemma-4-e2b-it" }
├── llamafile[.exe]       # inference engine (fetched by /download)
└── models/
    └── <model>.gguf      # model weights (fetched by /download)
```

## Available models

Currently ships with one model in the registry:

- `gemma-4-e2b-it` (~1.7GB)

More can be added by extending `availableModels` in `config.go`.

## Conversation context

Within a running chat session, the full turn history is replayed into every
prompt — so multi-turn follow-ups work. Two caveats:

- **Not persisted.** `/clear` or exiting the chat discards history. Nothing
  is written to disk.
- **No compaction.** The prompt grows linearly with the conversation. Once
  you cross the model's context window it will silently truncate.

One-shot commands (`--summarize`, `--dump --with-summaries`) are stateless —
each file is summarized in isolation.

## Building from source

The canonical build uses [gobake](https://github.com/fezcode/gobake) with the
repo's `Recipe.go` + `recipe.piml`:

```powershell
gobake build
```

Plain `go build` also works if you'd rather not install gobake:

```powershell
go build -o build/atlas.llm.exe .
```

## License

MIT — see [LICENSE](LICENSE).
