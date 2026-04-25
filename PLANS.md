# atlas.llm — planning notes

Working doc. Not a spec. Things get added, struck through, or moved into
commits as they land.

## Models to add to the registry

Currently in `availableModels` (config.go): `gemma-3-1b-it`,
`gemma-4-e2b-it`. The first is the safe default; the second crashes
llama.cpp 0.10.0's Haswell CPU backend on some setups.

Candidates to add:

### gemma-3-4b-it  — SHIPPED (v0.11.0)
- ~2.5 GB Q4_K_M.
- Unsloth repo pattern, same shape as the 1B entry we already ship:
  `https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf`.
- Nice middle ground between 1B (too dumb for real work) and 9B+ (too
  slow on laptop CPU).

### Qwen3.5-9B  — SHIPPED (v0.11.0)
- Using `unsloth/Qwen3.5-9B-GGUF`, Q4_K_M = 5.68 GB.
- Not a reasoning model; standard chat template works as-is.

### Ministral-3-14B-Instruct-2512  — SHIPPED (v0.11.0, non-reasoning variant)
- Using `unsloth/Ministral-3-14B-Instruct-2512-GGUF`, Q4_K_M = 8.24 GB.
- Chose the Instruct release (`mistralai/Ministral-3-14B-Instruct-2512`)
  over the Reasoning one — no `reasoning_content` field to handle, and
  we'd need feature #3 before the reasoning variant would be useful.

## Features

Ordered by effect on day-to-day use, not difficulty. Do top-down.

### 1. Streaming replies  — HIGH IMPACT
Biggest UX lift. Right now `chat()` POSTs to `/v1/chat/completions` and
blocks until the whole generation finishes, so a 20-token reply feels
the same as a 200-token reply: dead silence, then a wall of text.

Plan:
- Add `stream: true` to `chatRequest`. llama-server returns a
  Server-Sent Events stream of `data: {...}\n\n` frames terminated by
  `data: [DONE]\n\n`.
- New `server.go` method `ChatCompleteStream(msgs, maxTokens, onDelta)
  (UsageStats, error)`. `onDelta(partial string)` fires per frame.
- `engine.go` gets a streaming variant of `runChat`; `chat()` still
  returns the final string (aggregated), but the tea.Cmd for chat
  sends `assistantDeltaMsg{text}` messages as tokens land.
- `tui.go`: new message type `assistantDeltaMsg`. On the first delta,
  push an empty assistant pill; on subsequent deltas, append to the
  last rendered line instead of pushing a new one. Final msg
  (`assistantReplyMsg`) just flips busy=false.
- Cursor/typing indicator: maybe a `▋` cursor at the tail of the
  in-flight reply, removed when streaming completes.

### 2. Inline file references  — HIGH IMPACT
`@path` / `@dir/` syntax in the user prompt gets expanded into
context before sending to the model. Pairs with `/grep` — user can
type `@main.go @engine.go how does runInference route through the
server?` and get a real answer.

Plan:
- Preprocessor in `engine.go` (new file `references.go` maybe):
  scan user input for tokens matching `@(\S+)`. For each match:
  - If it's a file, read it (respect `summarizeMaxChars` cap).
  - If it's a dir, walk it with `loadIgnorer` + extension filter.
  - If it's a glob (`@src/**/*.ts`), expand.
  - Otherwise leave the literal in place.
- Inject into the prompt as a system or preamble user message:
  `Referenced files:\n---\nFILE: main.go\n<contents>\n---`.
- Budget: cap total reference bytes at some fraction of ctx (e.g.
  50% of `ctxN` in tokens ≈ 32 KB chars). Drop files past the budget
  with a warning.
- TUI: autocomplete for `@` — show a picker of matching paths?
  Probably phase 2.

### 3. Reasoning-model support  — BLOCKING for Ministral
Reasoning models emit two streams: `reasoning_content` (their
thinking) and `content` (the final answer). If we route them the
same way we route Gemma we'll print the thinking as the reply or
get an empty reply.

Plan:
- `chatResponse.Choices[].Message` already only has `content`. Add a
  parallel `ReasoningContent` field parsed from `reasoning_content`.
- On a reply, if `reasoning_content` is non-empty, render it in a
  collapsed dim block (`▸ reasoning (expand with /thoughts)` or
  similar), then the real reply below. Or just always show both,
  dim the thinking.
- For streaming, the SSE delta distinguishes via `delta.content` vs
  `delta.reasoning_content`.
- Wire up `/thoughts` slash command to toggle visibility.

### 4. Persistent sessions  — MEDIUM IMPACT
`/save NAME` writes current `history` to
`~/.atlas/atlas.llm.data/sessions/NAME.json`. `/load NAME` restores.
`/sessions` lists. Survives restarts. Useful once replies are
worth keeping.

### 5. Generation settings  — PARTIALLY SHIPPED (v0.13.0)
`/set max_tokens N` ships and persists to `config.json` (default 4096,
ceiling 12000 to leave headroom in the 16K ctx). `/set` with no args
lists current settings. `temp` / `top_p` not wired yet — add when there's
a reason to tune them.

### 6. Slash autocomplete  — SHIPPED (v0.14.0)
Tab completes slash command names, with a second pass for arg
completion: `/model <prefix>` against the model registry, `/set <prefix>`
against known keys, `/download <prefix>` against `engine` / `all` / model
names. Multiple matches extend to the longest common prefix and list
candidates inline.

### 9. Agentic tool-use  — SHIPPED (v0.17.0)
`/tools on` enables an OpenAI-style tool-call loop against llama-server's
`/v1/chat/completions`. Six tools: `read_file`, `list_dir`, `grep`,
`write_file`, `edit_file`, `run_cmd`. Destructive tools route through a
TUI confirm modal that pauses the agent loop until the user approves or
denies. Max 20 tool-call rounds per user turn. Qwen3.5-9B and
Ministral-3-14B are the realistic targets; Gemma 3 doesn't reliably
emit tool calls. No prompted-JSON fallback yet — add if there's demand.

### 8. Non-interactive `-c` mode  — SHIPPED (v0.15.0)
`atlas.llm -c "prompt"` prints the model reply to stdout and exits;
`-c -` (or piping into `-c ""`) reads the prompt from stdin. No history
across invocations. Lets atlas.llm slot into shell pipelines without
touching the TUI.

### 7. GPU offload
llama-server already accepts `-ngl N` (number of layers to put on
GPU). Default we pass is `0`. For users with NVIDIA GPUs, llama.cpp
needs to be the CUDA build — the one we download is CPU-only.
Bigger job: detect GPU, download the CUDA asset instead, expose
`-ngl` as a setting. Punt until someone asks.

## Non-features (parked)

- **Whisperfile / audio input.** The llama.cpp engine dir ships
  `whisperfile-0.10.0` in older releases; newer ggml-org/llama.cpp
  builds include `llama-tts.exe`. Cool, but out of scope for a chat
  tool.
- **Tool / function calling.** Interesting, but the local models
  we're shipping aren't reliable enough to make it useful.
- **Multimodal (image / vision).** Some candidate models (Ministral-3
  has a vision encoder; the engine ships `llama-mtmd-cli.exe`) could
  support this. Not worth the protocol work unless someone wants it.
