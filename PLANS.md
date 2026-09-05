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
- New `engine_server.go` method `ChatCompleteStream(msgs, maxTokens, onDelta)
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

#### GPU detection and CUDA archive selection — SHIPPED (v0.29.0)
Closes the three items that were open above, plus a bug that made the
documented two-step variant switch impossible to perform.

- `nvidia-smi --query-gpu=name,compute_cap,memory.total` is the probe
  (engine_gpu.go). Cached behind a `sync.Once`: `resolveEngineVariant` feeds the
  header and settings rendering, which repaint per keystroke, so spawning a
  process per frame would cost more than the feature saves. On Windows it
  reuses the engine's `CREATE_NO_WINDOW` flags so no console flashes.
- CUDA is no longer one pinned archive. `cudaArchives` lists builds
  newest-first with the compute-capability window each supports, because
  neither archive is a superset of the other: CUDA 13 dropped
  Maxwell/Pascal/Volta (floor sm_75) and 12.4 predates Blackwell (ceiling
  sm_90). A GTX 1080 (61) needs 12.4; an RTX 5070 Ti (120) needs 13.3 and
  was previously handed a build with no kernels for it. Undetected
  hardware falls back to the widest archive — too-new fails at load, while
  too-old only costs performance.
- `auto` now selects `cuda` when an NVIDIA GPU is detected *and* an archive
  covers it. Vulkan is still never auto-selected: nothing tells us a usable
  Vulkan driver is installed.
- `auto` never replaces an already-installed engine. Detection answering
  doesn't prove the CUDA build will load, and spending ~510MB on a working
  machine is not a call to make for the user, so detection decides fresh
  installs only; existing ones get a named suggestion in the /help
  Performance block instead (`engineUpgradeHint`).
- `-ngl` follows the *installed* engine rather than the configured variant
  (`effectiveEngineVariant`), so a detected-but-not-downloaded CUDA build
  no longer hands `-ngl 999` to a CPU-only binary.
- Fixed: `/download engine` skipped whenever any engine existed
  (`!isEngineDownloaded()`), so `/set engine_variant cuda` followed by
  `/download engine` reported "nothing (already present)" and stayed on
  CPU — the documented flow could never be completed. It now runs when the
  installed variant differs (`engineNeedsDownload`).
- Fixed: the HIP archive suffix `win-hip-radeon-x64.zip` no longer exists
  upstream; it was renamed to the versioned `win-rocm-7.14-x64.zip`, so the
  variant was selectable but matched no asset. Linux `ubuntu-rocm-*` is now
  registered too.

Still open:
- ROCm suffixes are version-pinned (`7.14`), the same fragility that broke
  the old HIP entry — a ROCm bump upstream silently breaks matching again.
  Tolerant matching (substring + extension rather than exact tail) would
  end the whole bug class for non-CUDA variants, which need no version
  discrimination.
- No VRAM-aware layer clamping. `auto` offloads everything, so a model
  larger than VRAM (muse-glimmer-30b at ~15.9GB on a 16GB card) needs an
  explicit `/set gpu_layers N`. Detection already reports `memory.total`,
  so the estimate is available — pairing it with the GGUF layer count
  would let auto pick a partial offload instead of failing at load.
- Linux CUDA is still absent because llama.cpp publishes no Linux CUDA
  binary; Vulkan and ROCm cover it.

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
header by a minimal metadata parser in engine_gguf.go — so the limit is the real
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
README — and kept drifting. config_set.go is now the single source: each entry
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
model shape overestimates ~4x on these models (see engine_memory.go), so the figure
is an honest floor rather than a wrong total, and the listing says the
context window adds more on top.

### 21. Tool path jail  — SHIPPED (v0.20.0)
Tools resolved paths against the process working directory with no root
concept, so the model could reach anywhere on the filesystem, and a wrong
relative path produced a bare "no such file or directory".

tool_jail.go now resolves every tool path against the directory atlas.llm was
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

#### Fixed: the mem segment made every keystroke laggy — (v0.29.1)
Reading resident memory spawns a process (`ps`, or `tasklist` on Windows),
and the header re-renders on every keystroke, so the spawn sat directly on
the input path. Measured at ~180ms per `tasklist` call. It only bit while a
model server was running, and stayed hidden until GPU offload made
everything else fast enough for the input lag to become the slowest thing
left.

Readings are now cached for 2s and refreshed off the render path: callers
get the last known value immediately and never block. The gauge stays blank
until the first refresh rather than stalling the first frame.

Two parsing bugs went with it, both in the `tasklist` memory column:
- The grouping separator follows the system locale. The parser stripped
  commas, so on a locale that groups with dots — `"8.174.328 K"` —
  `ParseInt` failed, the reading was dropped, and `/config` reported
  "model server not running" while it was plainly running.
- Splitting the CSV on `,` broke the *comma* locale too, just silently:
  `"12,345 K"` split across two fields and the last one parsed as 345,
  under-reporting by 1000x. Parsed with `encoding/csv` now, keeping only
  the digits so any grouping character works.

`TestProcessRSSRejectsBadPID` had been failing on a dot-locale machine for
this reason, and `TestHeaderSegmentsAreCheap` went from 22.8s to 0.00s.

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

### 26. Muse Glimmer 30B  — SHIPPED (v0.27.0)
Added muse-glimmer-30b (~15.9GB, unsloth UD-Q4_K_XL) — the first registry
entry aimed at 24/32GB machines.

Meta released it 2026-08-10 (Apache 2.0) as a local-agent model: built around
tool calling and long-horizon work, 131K context, llama.cpp support on day
one. That last part is what makes it a one-struct change here — the engine
downloader resolves the latest ggml-org release, so there is no pin to move.
Installs with an older cached engine need `/download engine` before the model
loads; the README says so.

The 16GB line from the Gemma 4 note (#23) still holds — this reads "too big"
in the picker's fit column on 16GB machines, and no quant changes that (the
smallest, UD-IQ2_XXS, is 10.7GB of weights before KV). The fit column doing
its job is the reason the registry can now carry a big-iron entry instead of
excluding them.

unsloth ships only UD dynamic quants for this model, so the filename breaks
the registry's Q4_K_M convention. URL verified to resolve at the size claimed
(15,878,222,368 bytes).

Text-only: vision needs the separate mmproj GGUF, which we don't download.
Not yet verified by running one — it's a 15.9GB download.

### 27. KV-cache optimizations, sized for 16GB — SHIPPED (v0.28.0)
llama-server now launches with an optimized flag set, and falls back to the
old flags automatically if the process dies before /health comes up (the
signature of a rejected flag — engines older than the `-fa on` syntax from
Sep 2025, or a backend that can't force flash attention). Timeouts and
missing binaries don't trigger the retry; only early exit does.

The set: `-fa on`, `--cache-type-k/v q8_0`, `--parallel 2` with `-c` doubled,
`--cache-reuse 256`.

The sizing invariant that makes this safe on 16GB machines: q8_0 halves KV
memory, and that saving pays for the second slot — two q8_0 slots at the
same per-slot context cost what one f16 slot did. Per-slot ctx is unchanged
(s.ctxN stays the per-slot value, so the TUI meter and the picker's fit
column stay honest), and total server memory is unchanged.

Why two slots: llama-server routes each request to the idle slot with the
longest matching prefix, so one-shot calls (/summarize, /grep, /compact's
summarizer) land in the second slot instead of evicting the conversation's
cached prefix from the only slot — previously any such call forced a full
re-prefill of the entire history on the next turn. --cache-reuse salvages
still-matching chunks after /compact rewrites history (runtime no-op on SWA
models like Gemma). /reset and /compact now erase all slots, not just 0.

README gained a KV-cache section and an MCP note (connect servers before a
long agent session — tools render into the prompt prefix, so a mid-session
join busts the cache once).

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

### 28. LAN inference: --serve and /set endpoint — SHIPPED (v0.30.0)
One atlas.llm hosts its model on the network; another uses it and needs no
engine and no weights of its own. A laptop drives an agentic session against
a desktop's GPU.

This was mostly already true and unexposed. Inference has always gone over
HTTP to llama-server, and llama-server already takes `--host`, `--port` and
`--api-key`, so no new protocol or transport was written — the work was
removing the assumption that the other end is a subprocess we own.

- `llamaServer` gained a `base` URL, and `cmd == nil` now means remote. The
  five hardcoded `http://127.0.0.1:%d` strings became `s.url(path)`, and
  every request goes through `s.do` so a bearer token is attached in one
  place. `stopLocked` and `DropKVCache` became no-ops for a remote: killing
  a process we didn't start, or erasing slots other clients are using, are
  the two ways a client could ruin a shared server.
- `requireEngine`/`requireModel` were duplicated at four call sites and are
  now one `requireInference()`, which is a no-op in remote mode. A client
  with no engine and no GGUF is the supported setup, not a broken install.
- `--serve` reuses the existing config (model, variant, ngl, ctx), binds to
  the LAN, prints the real interface addresses to paste into `/set endpoint`,
  and blocks until Ctrl+C. It refuses to run if the same install is
  configured as someone else's client.
- Slots default to 4 when serving so several clients keep their own KV
  prefix instead of evicting each other every turn. The budget does not grow
  with the slot count — `serveCapacity` divides a fixed `ctx_size * 2` — so
  raising slots costs context per client, not VRAM. Multiplying it would
  have OOM'd the 16GB card the default was sized for.
- The client learns what it's talking to from `/props`, since there's no
  local GGUF to read.

What deliberately does not move: tools run on the client, against the
directory it was started in. Only token generation is remote.

Settings that llama-server takes at spawn — ctx_size, gpu_layers,
engine_variant — belong to the server. Setting them on a client saves the
value but changes nothing, and says so; `/model` is refused outright. The
alternative was accepting them silently, which is the failure mode three
separate bugs in v0.29.x already came from.

Measured through the client path: 75.9 tok/s generating on ministral-3-14b
against the RTX 5070 Ti, attach in 60ms.

Known limitations:
- No auth by default. `--api-key` is plumbed through, but a bare `--serve`
  is open to anyone who can reach the port, and `/slots` means clients are
  not isolated from each other's KV state. Fine on a home LAN, not on an
  untrusted one.
- No discovery. Clients need the address typed in; there's no mDNS.
- The client can't switch the server's model, so a shared server is
  single-model until whoever runs it restarts it.
- Nothing reconnects. If the server goes away mid-session the next message
  errors, and `/set endpoint` again is the way back.

### 29. Remote session status and server identity — SHIPPED (v0.31.0)
A client could use a remote but couldn't tell you anything about it: no
indication in the header that inference had left the machine, and `/config`
still described the local engine that wasn't running.

The gap is that llama-server knows nothing about atlas. `/props` gives the
model path, `n_ctx`, `total_slots` and the llama.cpp build; `/slots` gives
live occupancy. None of it covers `gpu_layers`, `engine_variant`, or the
atlas version — which is most of what "show me the remote's config" means.

So a served instance now publishes `/atlas/info` on a sidecar port
(inference port + 1), carrying version, registry model name, engine variant,
-ngl, per-slot context, slot count, auth, and uptime. A sidecar rather than
a reverse proxy: nothing sits in the inference path, so streaming is
untouched and a bug here cannot break generation. The cost is a second port
through the firewall.

- Setting an endpoint probes it immediately and prints what answered rather
  than deferring the failure to the first message. Unreachable addresses are
  still saved — the usual cause is a server that isn't up yet — and the
  error names the cause: refused, no such host, or timed out, instead of
  Go's "connectex: No connection could be made because...".
- The header carries `⇅ REMOTE host:port`, coloured by a 15s background
  heartbeat. Two consecutive misses are required before it reads
  unreachable, so one dropped packet doesn't produce an indicator people
  learn to ignore. The badge reads cached state only — a status light that
  polls from the render path is precisely the v0.29.1 keystroke-lag bug, and
  a test pins 500 renders under 100ms.
- `/config` swaps ENGINE and MEMORY for a REMOTE section: connection state,
  server version and uptime, and the server's model, ctx_size, engine and
  gpu_layers under a heading that says they belong to that machine.
- A reachable server without the sidecar — a plain llama-server, or an
  atlas older than 0.31.0 — is reported as connected with less detail, not
  as an error. Only a payload identifying itself as `atlas.llm` is believed,
  since anything can be listening on port+1.

Known limitations:
- The sidecar is unauthenticated even when `--api-key` is set: the key
  guards inference, and the info endpoint exposes model and config details
  to anyone who can reach the port.
- Port+1 is a convention with no negotiation. If it's taken, serving
  continues without the sidecar and clients silently see less.
- The heartbeat only detects a server that stops answering /health. A server
  that answers but has swapped models is not noticed until reconnect.

### 30. Engine diagnostics, VRAM-aware offload, remote ceilings — SHIPPED (v0.32.0)
Closes the three items left open by the GPU and LAN work, plus a ceiling bug
the remote support introduced.

**llama.cpp's own output is no longer discarded.** `--log-disable` threw away
the model load, the slot layout, and the reason behind a failed start,
leaving a bare `exit status 1` as the only symptom. It is gone; stdout and
stderr were already piped into the atlas log, so about a dozen lines per
start now land there and nothing reaches the terminal.

Two findings while doing it:
- The first attempt passed `--no-log-colors`, which this build rejects. The
  change immediately diagnosed its own bug — previously that would have been
  `exit status 1` with nothing else. `--log-colors off` is the real flag.
- `--log-verbosity 0` suppresses everything; the default threshold is 3, and
  that is the level the useful lines sit at. So no verbosity flag is passed.
- `logWriter` now buffers to line boundaries. llama.cpp writes a line's
  timestamp and its text as separate `write` calls, so logging each chunk
  split every message across two entries.

Current builds do *not* print "offloaded N/N layers" at default verbosity, so
that is not available as offload evidence. VRAM in use is.

**`auto` no longer offloads a model that cannot fit.** `fitGPULayers` charges
the KV cache in full, holds back 1GiB for compute buffers and the CUDA
context, and divides what is left by the per-layer share of the weights.
muse-glimmer-30b on a 16GB card comes out at 39 of 48 layers rather than
attempting all 48 and failing at load. When anything needed is unknown it
declines to clamp: a guessed clamp is a permanent silent slowdown, which is
worse than a loud failure. The arithmetic is `layersThatFit`, split out so it
is testable without a GPU or a GGUF.

**`/config` gained a GPU section** — device, compute capability, VRAM in use,
and what is actually offloaded, with a partial offload labelled as either
"set explicitly" or "auto — model exceeds free VRAM". A silent estimate that
halves throughput should not look like a deliberate setting.

**`max_tokens` ceilings follow the remote.** The ceiling was
`resolveCtxSize(cfg) * 3/4` against the *client's* ctx_size, which says
nothing about the server's: a client at the default 16384 talking to a server
serving 8192 computed a ceiling of 12288, larger than the remote's entire
context, accepted it, and failed at request time. `effectiveCtxSize` prefers
the context the server reports.

Known limitations:
- The offload estimate assumes every layer costs the same share of the
  weights. Mixed-quant GGUFs (unsloth's UD dynamic quants among them) have
  uneven layers, so the estimate is approximate.
- Free VRAM is sampled once before launch. Another process claiming memory
  between the estimate and the load still fails, now with a real error.
- Per-request llama.cpp lines are included at default verbosity, so the log
  grows faster than before on a busy session.

#### Fixed: three tests that had been failing on Windows — (v0.32.0)
All three were platform assumptions, not flakes. They had been red long
enough to become background noise, which is how a real regression would have
hidden among them.

- `resolveInRoot("/etc/passwd")` was allowed. `filepath.IsAbs` is false for a
  slash-rooted path on Windows, so it was joined onto the jail root and
  resolved to `<root>/etc/passwd`. Never an escape, but it silently rewrote
  what was asked for, which could convince a model it had read the real file.
  `isRootedPath` now treats a leading `/` or `\` as absolute on every
  platform — Windows treats it as drive-relative anyway. This is a behaviour
  fix, not a test fix.
- `TestRunCmdUsesJailedWorkingDirectory` ran `pwd`, which resolves to MSYS's
  when git-bash is installed and prints `/tmp/...` for a path under `%TEMP%`.
  The working directory was correct all along; the comparison wasn't. The
  test now uses `cd` on Windows, which prints the native form.
- `TestMCPAuthStoreRoundTrip` asserted `perm&0077 == 0` on a file written
  0600. Go's Chmod on Windows only toggles the read-only attribute, so a
  writable file always reads back 0666 and the mode in `writeAuthStore` is a
  no-op there. What actually protects the file is NTFS inheritance from the
  profile directory, so on Windows the test asserts containment there
  instead.

Not done: a real DACL on the credential file. `golang.org/x/sys/windows`
could set one, but it is ~60 lines of security-sensitive code to strip
Administrators and SYSTEM from a file already restricted by profile
inheritance — a poor trade against the chance of getting an ACL subtly wrong.

#### Fixed: OSC colour-query escape leaking into the transcript — (v0.32.2)
A reply would sometimes be followed by `;rgb:158e/193a/1e75\` in the
transcript. That is a terminal answering an OSC 11 background-colour query.

`glamour.WithAutoStyle()` resolves dark-vs-light by asking termenv, and
termenv asks by writing the query to the terminal and reading the reply off
stdin. Inside the TUI bubbletea owns stdin, so the answer raced two readers
and when bubbletea won it landed in the transcript as text.

The timing explains the "sometimes": the renderer is only rebuilt when it
does not exist or the wrap width changed, and markdown is rendered when a
reply finishes — so it fired on the first reply of a session and again after
a resize, never at startup.

Resolution now happens once in `detectMarkdownStyle`, before the program
starts and while stdin is still ours, and renderers are built with
`WithStandardStyle`. It mirrors AutoStyle exactly, including the TTY check
that picks the notty style when stdout is redirected — missing that check
made rendered output carry ANSI codes under `go test` and broke
TestStoppingMidStreamKeepsPartialText.

Guarded by a source-level test asserting tui.go never calls WithAutoStyle.
The failure only appears against a real terminal — with stdin not a TTY, as
under `go test`, termenv answers from a default and nothing leaks — so a
runtime test would pass while the bug shipped. The guard was verified by
reintroducing the call and watching it fail.

Also fixed while here: `renderRemoteBadge` called `remoteEndpoint()`, which
reads and parses config.json, putting a disk read on every keystroke — a
milder repeat of the tasklist-per-frame bug from v0.29.1. It now reads only
the cached status, and the test deletes config.json before rendering to prove
it never looks.

### 31. Qwen3.8-27B — SHIPPED (v0.39.0)
Added qwen3.8-27b (~13.4GB, unsloth UD-Q3_K_XL) — the first dense ~27B
sized to sit entirely in 16GB, and on its benchmarks the strongest /tools
driver in the registry.

Qwen released it 2026-08-14 (Apache 2.0): dense 27B, 64 layers of the
Qwen3.5-style hybrid attention (Gated DeltaNet + Gated Attention), built for
agentic coding and computer use, 262K native context. GGUF quants and
llama.cpp support landed day one, so — like muse-glimmer before it — it is a
one-struct change here.

Quant choice is the interesting part. Unlike the two MoE entries, where an
oversized quant only spills cheap expert tensors, a dense model pays for
every spilled layer on every token — so Q4_K_M (17.1GB) on a 16GB card would
be a permanent slowdown. UD-Q3_K_XL (13.4GB) fits outright with room for the
KV cache, which the hybrid attention keeps small: only a minority of layers
carry a real KV cache, a layout the estimator in engine_gguf.go already models for
Qwen3.5.

Known limitations:
- Text-only. The model is a native vision-language model, but vision needs
  the separate mmproj GGUF, which we don't download (same stance as
  muse-glimmer-30b).
- Needs a llama.cpp build recent enough to know the architecture;
  `/download engine` refreshes an older install.

### 32. Engine tuning: fifteen llama-server knobs — SHIPPED (v0.40.0)
The most useful llama-server options as first-class settings: kv_offload,
flash_attn, cache_type_k, cache_type_v, threads, batch_size, ubatch_size,
parallel, cache_reuse, mmap, mlock, seed, temperature, override_tensor,
context_shift. Chosen over a raw pass-through escape hatch deliberately —
curated settings can validate conflicts and teach the estimators; a raw
string can't.

The zero value of every new Config field means "auto", pinned by a test that
asserts a zero config produces exactly the flag set that shipped before —
these defaults are measured and documented, and silent drift here changes
every user's launch.

**One registry entry per setting.** The `setting` struct gained `Apply` and
`Restart`; new settings live entirely in config_tuning.go and handleSet gained one
generic branch, instead of fifteen more bespoke switch cases. The older
settings keep theirs for their side effects (probes, shutdowns).

**The estimators follow the settings.** kv_offload=off zeroes the KV charge
in the VRAM fit math (`vramKVCharge`), so `gpu_layers auto` offloads more
layers — that is the setting's whole point. cache_type_k/v feed per-element
bytes into `kvCacheBytesTyped`; on a zero config the average is exactly the
q8_0 constant the estimate always used. override_tensor is explicitly *not*
modelled, and its help says so.

**Restart identity extended.** `tuningFingerprint` — raw settings, never
resolved autos, temperature excluded — is compared by `serverMatches`. Two
traps avoided: comparing resolved values re-creates the v0.28 restart loop,
and comparing the per-slot context share against resolveCtxSize would loop
the same way once `parallel` shrinks the share, so identity now compares
`askedCtx`.

**Conflicts are refused at /set time.** A quantized cache_type_v needs flash
attention; both orderings of that mistake error immediately instead of at
launch. When flash_attn is explicitly off and V is on auto, the launch
degrades V to f16 itself.

**Explicit tuning rides the fallback launch.** The optimized/fallback split
exists for flags the launch chooses on its own; a user's /set is an
instruction, and dropping it silently would make the setting a lie. An old
engine that rejects one fails loudly.

Verified against the real engine: all fourteen flags exist in the installed
llama-server's --help, and a live launch with a tuned set (--no-kv-offload,
q8_0 caches, --seed, -ub 256) reached /health OK.

Known limitations:
- temperature is the only sampling knob; top_p/top_k/min_p would need
  request-struct fields and were left for when someone asks.
- batch_size/ubatch_size don't feed the compute-buffer estimate — the
  estimator's 1GiB headroom absorbs the difference.

#### Fixed: Enter on an empty prompt inserted a newline — (v0.40.1)
The KeyEnter handler `break`s on empty input, and the event then falls
through to the shared `m.textarea.Update(msg)` at the bottom of Update —
which inserts a newline. The cursor dropped a line and the placeholder tip
vanished until the stray "\n" was backspaced away.

The first fix attempt swallowed the key only on the idle path and the test
still failed — which is how the real shape surfaced: `newChatModel` starts
in the busy state while the server warms up, and the busy guard sat *before*
the empty check, so the warm-up window (and any streaming reply) hit the
fall-through first. The empty check now runs before the busy check and
returns early in both states.

Deliberately kept: Enter on a busy *draft* still inserts a newline — typing
continues while a reply streams, and that is how a multi-line message is
composed. The test pins all three behaviours, and runs the empty case under
both busy states so it can't silently depend on what happens to be
downloaded on the machine running it.

### 33. Persistent browser profile — SHIPPED (v0.41.0)
A third profile mode for browser_open: `persist`, alongside `fresh` and
`default`. It launches on a stable directory atlas.llm owns —
`~/.atlas/atlas.llm.data/browser-profiles/<chrome|firefox>/` — that is *not*
deleted on close, so cookies survive across runs.

The motivating case is staying signed in. A fresh throwaway profile carries
no cookies, so a site the user returns to has to be logged into again on
every launch. A persistent profile keeps the session cookies, so signing in
once carries over. Sites that gate access on a profile's reputation (aged
cookies, real history) also fare better on a profile that accumulates that
state than on a blank one — though a live automation check is a separate
surface (the CDP attachment) that a profile alone doesn't change.

Design:
- `profileMode` enum (fresh/default/persist) replaces the `useDefault bool`
  threaded through launchChrome/launchFirefox/launchBrowser. `browserProfileMode`
  parses the tool argument.
- `resolveBrowserProfile` returns (dir, persist): fresh/default get a temp
  dir, persist gets the stable per-family dir. Family matters — the two
  profile formats are incompatible, so chrome and firefox can't share one.
- `killAndCleanup` gained a `remove bool`; every call passes `!persist`, so
  a persistent dir survives Close and every failure path, while throwaways
  are still cleaned up exactly as before.
- `prepPersistentProfile` clears the previous run's transient files before
  relaunch — the browser's own port announcement (DevToolsActivePort /
  WebDriverBiDiServer.json, which waitForBrowserFile would otherwise read
  stale and connect to a dead port) and the Singleton/lock files an unclean
  exit leaves behind. Real profile data (cookies, Local State) is preserved;
  that data is the feature.

Remote debugging runs against our own directory either way, so Chrome's
since-136 refusal to debug the *real* default data dir never applies.

Verified live on Chrome: launch persist → close → dir survives → relaunch on
the leftover dir comes up clean (proving the stale-state prep).

#### Fixed: a submitted prompt left a stray newline in the input — (v0.41.1)
The same fall-through as the empty-prompt fix (v0.40.1), on the other branch.
The send path resets the textarea, but KeyEnter then fell through to the
shared `m.textarea.Update(msg)` at the bottom of Update, which typed the Enter
into the now-empty box — so after every send the cursor sat on a blank second
line. The v0.40.1 fix only returned early on the *empty* path; the submit path
still fell through. It now returns right after queueing the send. Pinned by a
test asserting the textarea is empty (not "\n") after an Enter that starts a
turn.

#### Fixed: Enter added a newline while the model was generating — (v0.41.2)
The third and last face of the same fall-through. Enter on a busy session
(a reply streaming) hit `break` and fell through to the textarea, typing a
newline and dropping the cursor a line. The earlier rationale — "compose a
multi-line message while a reply streams" — was not worth the everyday
surprise of Enter moving the cursor mid-generation, so Enter is now swallowed
outright while busy. Other keys still reach the textarea, so a next message
can be drafted while the reply streams and sent once it ends. The busy-draft
test flipped to assert the draft is unchanged.

### 34. Ask me anything (/ama) — SHIPPED (v0.42.0)
A mode that lets the agent turn a decision back to the user through an
interactive picker instead of guessing. `/ama on|off`, persisted like /tools.

One tool, `ask_user`, with four `kind`s: radio (pick one), checkbox (pick
any — space toggles), confirm (yes/no step), answer_confirm (approve a
drafted answer/plan). The picker mirrors the existing model/confirm modals;
the choice is fed back as the tool result and steers the rest of the turn.

Design notes:
- `ask_user` is deliberately NOT in toolRegistry, so the "sixteen built-in
  tools" count and its help/destructive lists are untouched. It's a mode-gated
  pseudo-tool: `activeTools` appends it only while `amaOn` is set, `lookupTool`
  resolves it by name so a stray call is still handled, and its Run is a
  non-interactive fallback that errors (it's intercepted in the agent loop, so
  Run is never reached in the TUI).
- `amaOn` is an `atomic.Bool` — the tool layer (activeTools, called at config
  time on any goroutine) reads it, the UI writes it, no model lock needed. It
  mirrors the persisted cfg.AMAEnabled, initialised in newChatModel.
- dispatchNextTool intercepts `ask_user` before the destructive/run path and
  hands it to startAMA, which parses+validates the spec and opens the picker.
  resolveAMA feeds the selection (or, on Esc, a proceed-anyway note) back and
  continues the loop — the same shape as resolveConfirm.
- A dismissal never stalls the turn: the model is told to proceed with its
  best judgment. Malformed args or a call while /ama is off likewise get a
  synthesized result rather than hanging.

Only effective while /tools is on, since ask_user is only offered during an
agent turn; /config and the enable message both say so when tools are off.

#### Added: Ctrl+J / Alt+Enter compose a multi-line prompt — (v0.42.1)
Enter submits, so once the empty/submit/busy Enter fixes landed there was no
way left to type a newline. bubbletea v1.3.10 has no Shift key field and most
terminals collapse Shift+Enter to Enter, so the portable newline keys are
Ctrl+J (a reliable distinct key everywhere) and Alt+Enter (detected via the
Alt bit on the Enter key). Both insert a newline via textarea.InsertRune and
swallow the key so nothing submits. Surfaced in the welcome key list.

### 35. Named config profiles — SHIPPED (v0.43.0)
`/config` grew subcommands — save, load, list, delete — for named snapshots
of config.json. The motivating case is a fast/quality split: one profile with
a small model + small context + reasoning off, another with the big model and
a long context, flipped with a single `/config load`.

Design: a profile is a full copy of config.json under
`profiles/<name>.json`, so nothing about how settings are read changes —
config.json stays the one active config, and load/save just copy the whole
file. loadProfile normalises the same way loadConfig does, so a hand-edited or
older profile still comes back sane.

- `matchingProfile` compares the current config to each saved one (through
  their JSON encoding, so pointer fields match by value) and marks the one
  that matches exactly in `/config list` and at the top of bare `/config`.
  No "active profile" field to go stale after a manual /set — a match is
  computed, not tracked.
- Loading overwrites config.json, shuts the server down so the next message
  starts one for the new model/context/tuning, and re-syncs the session state
  that shadows config: agentEnabled, amaOn, and the remote-endpoint badge +
  heartbeat (mirroring the /set endpoint path when the endpoint changes).
- Names are validated as `[A-Za-z0-9_-]{1,32}` — they double as filenames, so
  no separators or dots. Tab-completes profile names after load/delete.

The load integration test backs up and restores the real config.json around
the overwrite, since it exercises the true saveConfig path.

#### Added: /config show <name> — (v0.43.1)
Inspect a profile's settings without loading it. renderProfile prints the
persisted values only (the settings registry plus model, tools, ama) — not
live session state or memory, which belong to the running session, not the
snapshot — and marks the profile ● active when it matches the current config.
Tab-completes profile names like load/delete.
