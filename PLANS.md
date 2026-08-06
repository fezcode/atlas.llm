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

### Ministral-3-8B-Instruct-2512  — SHIPPED (v0.19.0)
Fills the 4B -> 9B gap: Q4_K_M is 5.20 GB, between qwen3.5-4b (2.74) and
qwen3.5-9b (5.68). Same family as the 14B below, which already tool-calls
reliably, so /tools and /mcp work.

Qwen3.5 has no dense size between 4B and 9B (0.8, 2, 4, 9, 27, then MoE), so
the gap had to be filled from another family. Llama-3.1-8B-Instruct (4.92 GB)
also exists and would fit, but it is a 2024 model against Ministral-3's
2512 release.

Worth remembering as an alternative lever: a lighter *quant* of the 9B beats
a smaller model at a heavier quant. Qwen3.5-9B ships Q3_K_M at 4.67 GB and
IQ4_XS at 5.17 GB, both under the 9B Q4_K_M we list.

### Ministral-3-14B-Instruct-2512  — SHIPPED (v0.11.0, non-reasoning variant)
- Using `unsloth/Ministral-3-14B-Instruct-2512-GGUF`, Q4_K_M = 8.24 GB.
- Chose the Instruct release (`mistralai/Ministral-3-14B-Instruct-2512`)
  over the Reasoning one — no `reasoning_content` field to handle, and
  we'd need feature #3 before the reasoning variant would be useful.

## Features

Ordered by effect on day-to-day use, not difficulty. Do top-down.

### 1. Streaming replies  — SHIPPED (v0.23.0)
`stream: true` with SSE parsing in `ChatCompleteStream`. Deltas reach the
TUI through `program.Send` (a tea.Cmd can only return one message), and the
reply occupies a single line in the transcript that is rewritten in place.

Measured on qwen3.5-4b, "In one sentence, what is a KV cache?": the screen
used to sit frozen for 214s. First sign of life is now 1.6s.

That measurement also exposed the real problem behind the wait: the model
spent 65 of 66 seconds on internal reasoning for a one-sentence question and
produced 4935 bytes of thinking against 172 bytes of answer. Reasoning
deltas are counted and surfaced as "thinking… (N of reasoning so far)"
rather than being discarded, so a long think is visible instead of looking
like a hang — and esc becomes a real option.

Details worth keeping:
- Rendered as plain text while streaming, markdown only at the end. Running
  glamour per token is slow and unstable, since a half-written code fence
  renders as garbage.
- Cancelling keeps the partial text and drops the cursor; measured 3.0s from
  cancel to return.
- Deltas arriving after a stop are ignored, so an abandoned turn cannot
  write into the next one.

Not done: agent turns still block, because streamed tool_calls arrive
fragmented and need reassembly. That is the natural follow-up.

### 1b. Original notes
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

### 3. Reasoning-model support  — PARTIALLY SHIPPED (v0.19.0)
Qwen3.5 is a reasoning model. One-shot tasks (`/compact`, `/summarize`,
`/grep`) now send `chat_template_kwargs: {"enable_thinking": false}`, because
the <think> block adds nothing there and consumed the entire token budget —
`/compact` on qwen3.5-4b returned `content:""` with 2155 bytes of
`reasoning_content` and `finish_reason:length`. Note `reasoning_budget: 0`
does NOT work: it truncates the thinking rather than preventing it.

`reasoning_content` is now parsed, so an empty answer with a full reasoning
block reports "used its entire N-token budget on internal reasoning" instead
of surfacing as a blank reply.

Still open: thinking is left enabled for chat and agent turns (it helps tool
use), so a long think can still exhaust max_tokens there. Displaying
reasoning in a collapsed block, and a `/thoughts` toggle, are unbuilt.

### 3b. Original notes  — BLOCKING for Ministral
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

### 10. MCP client  — SHIPPED (v0.18.0)
Connects to Model Context Protocol servers and folds their tools into the
same registry, tool-call loop, and confirm modal as the built-ins. Built on
the official `github.com/modelcontextprotocol/go-sdk` rather than a
hand-rolled JSON-RPC client.

- Config at `~/.atlas/atlas.llm.data/mcp.json`, using the standard
  `mcpServers` shape so existing Claude Desktop / VS Code configs paste in.
- Transports: stdio (`command`/`args`/`env`) and remote HTTP (`url`), with
  `"transport": "sse"` for the older 2024-11-05 protocol.
- OAuth 2.1 + PKCE for hosted servers (`"oauth": true`): loopback redirect
  listener, browser launch, Dynamic Client Registration by default. Tokens
  plus the discovered endpoint config persist to `mcp-auth.json` (0600) so
  refresh tokens survive restarts.
- Tools namespaced `server__tool`, sanitized and capped at 64 chars.
- Per-server `"trust"` decides confirmation. Servers' own `readOnlyHint`
  annotations are deliberately ignored — they come from the party being
  gated.
- `/mcp add` opens a picker over a built-in catalog (Atlassian, Slack,
  GitHub, Linear, Sentry, filesystem, memory, plus everything/deepwiki as
  stdio and remote smoke-test servers) and writes
  mcp.json for the user — no hand-editing to get started. Presets needing a
  token or path pre-fill the command in the input box. `/mcp add NAME -- cmd`
  and `--url=...` cover anything not in the catalog.
- `/mcp`, `/mcp add|remove|trust|env|catalog|connect|disconnect|tools|logout|help`.
- Startup auto-connects everything except OAuth servers with no stored
  credentials, so nothing pops a browser unprompted.

Follow-ups worth doing:
- MCP **prompts** and **resources** — only tools are wired up today.
- `tools/list_changed` notifications: we snapshot tools at connect time and
  don't yet react when a server's tool list changes mid-session.
- OS keychain storage instead of the 0600 file (the macOS `security` CLI
  takes the secret through argv, which is why it wasn't used).
- Per-tool trust overrides, rather than trust being all-or-nothing per
  server.
- Catalog presets are hardcoded; a refresh mechanism (or pulling from an
  upstream registry) would keep package names and endpoints from going
  stale.

### 8. Non-interactive `-c` mode  — SHIPPED (v0.15.0)
`atlas.llm -c "prompt"` prints the model reply to stdout and exits;
`-c -` (or piping into `-c ""`) reads the prompt from stdin. No history
across invocations. Lets atlas.llm slot into shell pipelines without
touching the TUI.

### 7. GPU offload  — SHIPPED (v0.18.0)
`-ngl` is now a setting instead of a hardcoded `0`, and the engine archive
is selectable.

- `/set gpu_layers auto|0|N` -> `-ngl`. `auto` (the default) offloads
  everything on macOS, where the standard llama.cpp archive already ships
  the Metal backend, and stays on CPU elsewhere unless a GPU engine is
  installed. Stored as `*int` so an explicit `0` (force CPU) is
  distinguishable from "unset".
- `/set engine_variant auto|cpu|vulkan|cuda|hip` picks the release archive;
  the offered list is filtered to what llama.cpp actually publishes for the
  running platform. Windows x64 gets all four. CUDA downloads the
  `cudart-llama-bin-*` runtime alongside the engine (two-asset install).
  A variant switch wipes the engine dir and re-downloads; the installed
  variant is recorded in `engine/.variant`.
- Asset matching requires a `llama-` prefix as well as the suffix:
  `cudart-llama-bin-win-cuda-12.4-x64.zip` and
  `llama-b10280-bin-win-cuda-12.4-x64.zip` share a suffix, so tail-only
  matching picked whichever came first in the release listing.
- `ensureServer` now treats `-ngl` as part of the server's identity, so
  changing the setting restarts llama-server instead of silently doing
  nothing until the next model switch.

Measured on an M-series Mac with gemma-3-1b-it Q4_K_M, through atlas's own
server path: 12.1 tok/s at `-ngl 0` vs 27.0 tok/s on Metal (2.2x). Raw
llama-bench on the same model shows prompt processing going 208 -> 997 t/s
(4.8x), which matters most for agent loops with large tool results.

Still open:
- No GPU/driver detection: `auto` never picks a GPU build on Windows/Linux,
  because one without a matching driver fails at load. The user opts in.
  Probing for `nvidia-smi` would let us at least *suggest* CUDA.
- CUDA is pinned to the 12.4 archive; 13.3 also ships. No driver-version
  check to choose between them.
- Linux GPU is Vulkan-only — llama.cpp publishes no Linux CUDA binary, and
  ROCm (`ubuntu-rocm-*`) is unhandled.

### 11. Long-form help  — SHIPPED (v0.19.0)
`/help <command>` and `/help <command> <subcommand>` print full
documentation: every accepted form, what it writes to disk, and the failure
modes worth knowing. `/help` alone keeps the compact overview and adds an
index of documented topics.

- Topics live in a `helpTopics` table (help.go) — summary, usage forms,
  prose detail, per-subcommand detail, examples, notes, cross-references.
- Tab completes both levels: `/help mc<Tab>` then `/help mcp tr<Tab>`.
- Tests assert every command in `slashCommands` has a topic, that every
  implemented subcommand of /mcp, /tools, and /set is documented, and that
  no cross-reference points at a missing topic — so the docs can't silently
  drift from the dispatcher.
- Prose is wrapped at render time against terminal width; lines that begin
  with whitespace are treated as pre-formatted so command examples aren't
  reflowed.

### 12. Input history recall  — SHIPPED (v0.19.0)
`↑`/`↓` walk back and forward through submitted input, shell style. A
half-typed line is parked when you start browsing and restored when you come
back past the newest entry.

Recall only takes over at the first line (for `↑`) and the last line (for
`↓`), so the arrows still move the cursor inside a multi-line draft. Bounded
at 200 entries; consecutive duplicates collapse. In-memory only — not
persisted across sessions, which would be the obvious follow-up.

### 13. Context compaction  — SHIPPED (v0.19.0)
`/compact` folds older turns into a dense factual summary and continues from
it, keeping the last 4 turns verbatim. Less destructive than `/reset`, which
discards the conversation outright.

Prompted by hitting the context limit almost immediately once MCP was
connected. Measured cause: `toolResultSizeLimit` was 32KB — roughly 8K
tokens, half the entire 16K window from a *single* tool call, so two calls
exhausted it. Tool definitions were only ~927 tokens (7%), so they were
never the problem. The cap is now 6KB (~1.5K tokens, ~9% of ctx).

- Orphaned `role: tool` turns and `tool_calls` turns are dropped when their
  counterpart is folded away — llama-server rejects a tool result whose
  originating call is no longer visible.
- The summary is re-injected as a user turn plus a short assistant ack,
  which every chat template handles; a second system message is ignored or
  mishandled by some models.
- KV cache is dropped afterwards, since the rewritten history no longer
  matches the cached prefix.
- A one-shot hint fires at 75% context fill, chosen so there's still room to
  run the summarization itself.

Follow-ups: automatic compaction at a threshold rather than a prompt, and a
configurable `-c` context size (currently hardcoded at 16384).

### 14. Stop generation  — SHIPPED (v0.19.0)
`Esc` aborts an in-flight generation. A `context.Context` is now plumbed
through `chatCompleteCore` to `http.NewRequestWithContext`, so cancelling
closes the connection and llama-server stops generating server-side rather
than the UI merely detaching. Measured: aborts in ~1ms against a live model.

- Queued tool calls from the interrupted turn are abandoned, and an open
  confirm modal is dismissed.
- Starting a turn cancels any previous one, so an abandoned generation can't
  outlive the turn that started it.
- The resulting `context.Canceled` is recognised and reported as "Stopped"
  rather than as an error; a genuine failure during a cancelled turn is
  still surfaced.
- The footer switches to `esc stop generating` while busy.

Not covered: an MCP tool call already in flight still runs to its own
timeout — Tool.Run takes no context. Threading one through would let Esc
interrupt a slow server too.

### 15. multi_edit tool  — SHIPPED (v0.19.0)
`multi_edit` applies a batch of find/replace edits to one file atomically.
Previously N changes to one file meant N `edit_file` calls, each with its own
confirmation, and a failure partway left the file half-edited.

- Edits apply in order against the running content, so a later edit may
  legitimately target text an earlier one introduced.
- Each `old_string` must still match exactly once at the point it is applied.
- Any failure means nothing is written, and the error names the failing
  index so a large batch is debuggable.

### 16. /yesman  — SHIPPED (v0.19.0)
Session-only bypass of the destructive-tool confirmation, for runs where
approving every call is the bottleneck.

Deliberately not a Config field. Persisting it would mean a session days
later silently auto-running run_cmd because of a toggle flipped once, so it
lives on chatModel and dies with the process — a test asserts it never
reaches config.json.

Kept loud rather than silent, since it removes the only barrier in front of
arbitrary shell execution: a red marker in both header and footer while
armed, and every auto-approved call still printed in the trace as
"(auto-approved by /yesman)". esc remains the escape hatch mid-turn.

### 17. Typing no longer scrolls the transcript  — FIXED (v0.19.0)
Every message, including tea.KeyMsg, was forwarded to the viewport, whose
bubbles default keymap is vim-style — so typing h/j/k/l/u/d/f/b or space
scrolled the transcript instead of entering characters.

Key events are no longer forwarded to the viewport at all, and its KeyMap is
zeroed so reinstating the forward can't silently bring the bug back. Explicit
PgUp/PgDn bindings replace the removed scrolling. Non-key messages (mouse
wheel, resize) still reach it. No vim mode.

### 18. Configurable context size  — SHIPPED (v0.19.0)
`-c` was hardcoded at 16384. `/set ctx_size auto|N` now sets it.

The ceiling is the model's own trained context length, read from the GGUF
header by a minimal metadata parser in gguf.go — so the limit is the real
one rather than a guess. Measured on the shipped models: Qwen3.5 (4B and 9B)
report 262144, Gemma 3 1B reports 32768, meaning the old hardcoded 16384 was
using 6% of what Qwen supports.

atlas.llm caps at 131072 regardless, since the KV cache at Qwen's full range
would be tens of gigabytes, with a 2048 floor so the system prompt and tool
definitions always fit.

max_tokens now derives its ceiling from ctx_size (three quarters) instead of
the old fixed 12000, which assumed a 16K window.

ctx is part of the server's identity alongside -ngl, so a change restarts
llama-server rather than silently taking effect on the next model switch.

### 19. /config and per-setting help  — SHIPPED (v0.19.0)
Settings information lived in three places — handleSet, help.go, and the
README — and kept drifting. settings.go is now the single source: each entry
carries its key, usage, how to render the current value, and its guidance,
and /set, /config, and the /help cross-check test all read from it.

- `/set` lists keys with values and summaries.
- `/set <key>` explains the setting instead of echoing it: current value,
  limits computed for the active model, cost, usage, and a pointer to
  `/help set <key>`.
- `/config` shows settings, model, engine, session-only state (tools,
  yesman, MCP), memory, and file paths in one place.
- A test asserts every registry key has a matching `/help set` subcommand.

Memory is measured from the running server process, not predicted. The
obvious KV formula (2 * layers * ctx * kv_heads * head_dim * 2 bytes)
overestimates ~4x on the shipped models: measured Qwen3.5-4B at 0.91 GB
resident for ctx 16384 and 2.36 GB for 65536, about 31 KB/token against the
formula's 128 KB/token — consistent with hybrid attention where only a
fraction of layers keep a full-context cache. Showing the formula's number
would have been four times too large, so it was dropped.

### 20. --reset-model and per-model RAM share  — SHIPPED (v0.20.0)
Init() warms the model server at startup, so quitting on a 14B left the next
launch blocked loading it with no way to reach /model. `atlas.llm
--reset-model` switches to the smallest registry entry and exits, leaving
every other setting untouched.

"Smallest" is computed by parsing the registry Size strings rather than
trusting defaultModel or list order, so adding a smaller model later is
picked up automatically.

/list and the model picker now show each model's weights as a share of
system RAM (read from sysctl / /proc/meminfo / wmic) with a verdict.

Only the weights are counted, deliberately. Predicting the KV cache from
model shape overestimates ~4x on these models (see memory.go), so the figure
is an honest floor rather than a wrong total, and the listing says the
context window adds more on top.

### 21. Tool path jail  — SHIPPED (v0.20.0)
Tools resolved paths against the process working directory with no root
concept, so the model could reach anywhere on the filesystem, and a wrong
relative path produced a bare "no such file or directory".

jail.go now resolves every tool path against the directory atlas.llm was
started in. At or below the root is allowed; anything above is refused with
an error that names the root, so the model can correct rather than guess.
Symlinks are resolved on the deepest existing ancestor, which catches a link
pointing outside even when the final component doesn't exist yet (the
write_file case).

run_cmd gained a cwd argument and now sets cmd.Dir. Each call was already a
fresh shell, so a leading `cd` never persisted — the model had no way to work
in a subdirectory across calls.

Related: identical repeated tool calls are detected after 4 attempts and stop
the turn naming the call, and the 20-round message now explains what
happened and where paths are rooted. Both address turns that hit the cap by
retrying a failing call rather than doing useful work.

Follow-up: the jail constrains tool *arguments*, not what a shell command
does once running. Real confinement of run_cmd needs OS sandboxing.

### 22. Dynamic memory estimates  — SHIPPED (v0.21.0)
The RAM figure in /list was half-dynamic: system RAM was read from the OS,
but model size came from the hardcoded registry string and the context cost
was ignored entirely.

Both are now computed. Downloaded models are sized from the file rather than
the declared string, which is approximate (gemma-4-e2b declares ~2.9GB and
is 3.11GB), and the KV cache at the current ctx_size is added.

The earlier 4x overestimate is solved, not worked around. The GGUF metadata
says why: Qwen3.5 reports `full_attention_interval = 4`, meaning it is a
hybrid SSM/attention model where only 8 of its 32 layers keep a KV cache —
the rest are state-space layers with a fixed-size recurrent state. Counting
only the attending layers brings the estimate to within about 11% of
measurement (predicted 1.61 GB growth from ctx 16384 to 65536 against 1.45 GB
measured), and a test pins that against the recorded hardware numbers.

Gemma 3 reports `attention.sliding_window = 512` but no pattern for how many
layers stay global, so the window is deliberately not applied: erring high is
the right direction for a "will this fit" number.

The header gained a `mem` segment next to the context gauge, showing the
model server's measured resident memory and its share of system RAM.

### 23. Gemma 4  — SHIPPED (v0.22.0)
Added gemma-4-e4b-it (~5.0GB) and gemma-4-12b-it (~7.1GB), joining the
gemma-4-e2b-it already in the registry.

The reason to care: Gemma 4 ships a tool-calling chat template. Gemma 3 does
not, which is why the default gemma-3-1b-it makes /tools and /mcp look broken
on a fresh install. The Gemma 4 template (published 2026-07-09) carries
tool_calls, tool_response and tools, and its header comment reads "Fixed
tool-calling loops, turn closures, and thinking content-ordering".

Not yet verified by running one — a tool-capable template is not proof the
model uses it well, and Gemma 3 parsed fine while still fabricating
```tool_code blocks.

Bigger Gemma 4 variants do not fit 16GB: 26B-A4B is 16.95GB and 31B is
18.32GB at Q4_K_M.

Worth revisiting: if gemma-4-e2b-it tool-calls reliably it is a better
default than gemma-3-1b-it, which would fix the out-of-the-box MCP
experience.

### 24. Picker scrolling  — FIXED (v0.22.0)
renderPicker called viewport.GotoTop() on every repaint, so a list taller
than the viewport left everything below the fold unreachable — the cursor
moved but was invisible, making later models look absent from the picker.
The MCP catalog picker had the mirror-image bug, pinned to the bottom.

Both now scroll only as far as needed to keep the selection visible, and
snap to the top near the first row so the title stays on screen. The bug got
worse with every model added, and the registry is now nine.

### 25. Reasoning toggle and cwd hints  — SHIPPED (v0.24.0)
`/set reasoning auto|on|off`. auto is not a single switch: it disables
thinking for chat and keeps it for tool-driven turns, because the cost
differs enormously by task. Measured on qwen3.5-4b, "in one sentence, what is
a KV cache?": 63.9s with thinking against 3.2s without, first word at 62.1s
against 0.3s. Twenty times slower for a question that needed none of it,
while thinking still earns its keep on tool selection.

Separately, agents kept trying to cd into the project. Two fixes:

The agent system prompt is now built per turn and carries the real project
root plus its top-level entries (dotfiles excluded), so the model can see the
layout instead of guessing names and burning tool-call rounds on the retries.

run_cmd lifts a leading `cd` — `cd sub && ls` runs in `sub` — and says so in
the result, pointing at the cwd argument. Each call is a fresh shell, so a
`cd` never survived to the next call and models assumed otherwise. An
explicit cwd wins, and `cd ..` is still refused by the jail.

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
