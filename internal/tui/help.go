package tui

import (
	"fmt"
	"sort"
	"strings"

	"github.com/charmbracelet/lipgloss"

	"atlas.llm/internal/ui"
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
			"Every form includes the engine, so plain /download is enough after " +
			"an engine_variant change; /download engine is just the narrower " +
			"version that skips model weights.\n\n" +
			"The engine is the latest llama.cpp release build for your OS and " +
			"architecture, resolved from GitHub at download time, so you get " +
			"current binaries rather than a pinned version. Which archive is " +
			"fetched depends on `/set engine_variant` (CPU vs a GPU build), and " +
			"for CUDA also on your GPU's compute capability — CUDA 13 dropped " +
			"the older architectures that 12.4 still supports, so the archive is " +
			"chosen to match the card.\n\n" +
			"Re-running is cheap: an engine or model already present is skipped. " +
			"The exception is an engine_variant change, which wipes the engine " +
			"directory and re-downloads, since mixing build variants would leave " +
			"the wrong binaries behind.",
		Subcommands: []helpSub{
			{Name: "engine", Usage: "/download engine",
				Detail: "Just the llama.cpp engine, no model weights. Applies a " +
					"pending engine_variant change, as every /download form does."},
			{Name: "<model>", Usage: "/download qwen3.5-9b",
				Detail: "The engine plus that model. Does not switch to it — use /model."},
			{Name: "all", Usage: "/download all",
				Detail: "The engine plus every model in the registry. Tens of gigabytes."},
		},
		Notes: []string{
			"Downloads run in the background with a progress bar, transfer " +
				"speed, and elapsed time; the TUI stays usable.",
			"CUDA is a two-archive install — the engine plus a ~372MB CUDA runtime — " +
				"so it's markedly larger than the other variants.",
			"On a first install with no engine present, an NVIDIA GPU is detected " +
				"and the CUDA archive picked automatically. An engine that is " +
				"already installed is never replaced without you asking.",
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
			{Name: "show_thinking", Usage: "/set show_thinking on|off",
				Detail: "Stream the <think> block's actual text into the transcript, " +
					"dimmed, above each reply — instead of the byte counter shown " +
					"by default while a model thinks.\n\n" +
					"Display-only: the thinking never joins the history sent back " +
					"to the model, and ^Y copies only the reply. A turn that burns " +
					"its whole token budget thinking keeps its think block on " +
					"screen, which is exactly the turn worth reading.\n\n" +
					"Nothing appears unless the model thinks at all — that is " +
					"`/set reasoning`'s job. Tool-driven turns show the tool trace " +
					"instead of their thinking."},
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
					"tokens; Gemma 3 for 32768. atlas.llm caps the setting at " +
					"262144 — big windows are affordable with quantized cache " +
					"types (q4_0 KV at 150K fits beside a ~9GB model on a 12GB " +
					"card), and the offload planner checks the real fit before " +
					"every launch.\n\n" +
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
					"Under `auto`, a mixture-of-experts model that does not fit is " +
					"handled differently: every layer's attention stays on the GPU " +
					"and the experts of as many layers as needed go to system RAM " +
					"instead. Attention runs for every token and expert weights " +
					"largely do not, so this is much faster than dropping whole " +
					"layers. /config shows it on a `moe` line. Setting a number here " +
					"turns that off — an explicit layer count is taken literally.\n\n" +
					"Changing this restarts the model server, so it applies on your " +
					"next message.\n\n" +
					"On Windows and Linux this does nothing until a GPU build is " +
					"installed — see `/help set engine_variant`."},
			{Name: "endpoint", Usage: "/set endpoint IP:PORT | local",
				Detail: "Runs inference on another machine instead of this one. Point " +
					"it at an atlas.llm started with --serve:\n" +
					"  /set endpoint 192.168.1.50:8080\n\n" +
					"A bare address, an address with a port, or a full URL all work; " +
					"port 8080 is assumed if you leave it off. `local` clears it.\n\n" +
					"Setting it checks the address immediately and reports what " +
					"answered — server version, model, context and slots — rather " +
					"than letting a typo look fine until your first message fails. " +
					"An address that does not answer is still saved, since the usual " +
					"reason is a server that has not been started yet.\n\n" +
					"While connected the header carries a REMOTE badge naming the " +
					"host, coloured by a background heartbeat: green when healthy, " +
					"amber after a missed beat, red when it has gone. `/config` " +
					"replaces its ENGINE and MEMORY sections with a REMOTE section " +
					"describing the server and the settings it decides.\n\n" +
					"While it is set, this install needs no engine and no model file — " +
					"that is the point, and it is what makes a laptop able to use a " +
					"desktop's GPU. /download becomes unnecessary.\n\n" +
					"What moves and what doesn't: only token generation is remote. " +
					"Tools run here, against the files in the directory you started " +
					"in, so an agentic session edits your machine while inferring on " +
					"the server's card.\n\n" +
					"ctx_size, gpu_layers and engine_variant are handed to llama-server " +
					"when it starts, so the server owns them. Setting them here while " +
					"remote saves the value but changes nothing until you go back to " +
					"local, and atlas.llm says so rather than accepting them silently. " +
					"/model is refused outright, since only the serving machine can " +
					"change which weights are loaded.\n\n" +
					"/reset clears your own history but deliberately leaves the " +
					"server's KV slots alone — they are shared, and erasing them " +
					"would force every other client to re-process its conversation."},
			{Name: "endpoint_key", Usage: "/set endpoint_key KEY",
				Detail: "Bearer token for a server started with --api-key. Servers on a " +
					"trusted LAN are usually started without one, in which case leave " +
					"this empty.\n\n" +
					"Stored in plain text in config.json. Treat it as a shared network " +
					"secret, not a real credential."},
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
					"  auto    default; picks cuda on a detected NVIDIA card when\n" +
					"          no engine is installed yet, cpu otherwise. On macOS\n" +
					"          cpu already includes Metal\n" +
					"  cpu     no GPU. Smallest download, works anywhere\n" +
					"  vulkan  NVIDIA, AMD, or Intel on Windows and Linux. ~32MB,\n" +
					"          the easiest GPU option and usually the right first try\n" +
					"  cuda    NVIDIA on Windows. Usually the fastest, but ~510MB\n" +
					"          because the CUDA runtime ships as a second archive\n" +
					"  hip     AMD Radeon on Windows and Linux. ~190MB\n\n" +
					"macOS needs none of this. The macOS archive has always carried " +
					"the Metal backend, so `auto` already runs on the GPU and the " +
					"only GPU-capable value here is cpu.\n\n" +
					"There is no single CUDA archive. CUDA 13 dropped Maxwell, " +
					"Pascal and Volta, and CUDA 12.4 predates Blackwell, so the " +
					"card's compute capability decides which one is fetched: 13.3 " +
					"for an RTX 50-series, 12.4 for a GTX 1080. A GPU no archive " +
					"covers is reported as such instead of downloading half a " +
					"gigabyte that cannot load a model.\n\n" +
					"Only variants llama.cpp actually publishes for your platform " +
					"are accepted — asking for cuda on macOS is rejected up front " +
					"with the list of what is available, rather than failing later " +
					"during the download.\n\n" +
					"A GPU build needs a working driver. NVIDIA is detected through " +
					"nvidia-smi, but only to choose for a fresh install — an engine " +
					"already installed is never replaced without you asking, and " +
					"vulkan is never chosen automatically because nothing reveals " +
					"whether a usable Vulkan driver is present. If a GPU build " +
					"misbehaves, `/set engine_variant cpu` then `/download engine` " +
					"puts you back."},
			{Name: "kv_offload", Usage: "/set kv_offload on|off",
				Detail: "Where the KV cache lives. `on` (the default) keeps it in VRAM " +
					"next to the offloaded layers. `off` allocates it in system RAM " +
					"(llama.cpp --no-kv-offload) while every weight layer stays on " +
					"the GPU — and the offload estimator knows, so `gpu_layers auto` " +
					"fits more layers.\n\n" +
					"The trade is speed: attention reads the cache on every generated " +
					"token, so with the cache behind the PCIe bus generation slows — " +
					"usually more than spilling one or two weight layers would. It " +
					"wins when the cache, not the weights, is what doesn't fit: huge " +
					"contexts on dense models. On hybrid-attention models (Qwen3.5, " +
					"qwen3.8-27b) the cache is small and this buys little either way."},
			{Name: "flash_attn", Usage: "/set flash_attn auto|on|off",
				Detail: "Flash attention (llama.cpp -fa). `auto` forces it on for the " +
					"optimized launch: it speeds up prompt processing and is what " +
					"allows the V cache to be quantized. The automatic fallback " +
					"launch omits it for engines that predate the syntax.\n\n" +
					"`off` is for backend/model combinations where forcing it breaks " +
					"the load — the symptom is a launch that dies immediately after " +
					"a model switch. It conflicts with a quantized cache_type_v, and " +
					"the conflict is refused at /set time."},
			{Name: "cache_type_k", Usage: "/set cache_type_k auto|q4_0|q4_1|q5_0|q5_1|q8_0|f16|bf16|f32",
				Detail: "Quantization of the K half of the KV cache. `auto` is q8_0: " +
					"half the memory of f16 at no measurable quality cost — it is " +
					"what pays for the second server slot.\n\n" +
					"q4_0 halves it again, but K tolerates quantization worse than " +
					"V; expect degradation on long contexts. f16 is the " +
					"full-precision fallback if a model misbehaves at q8_0.\n\n" +
					"The memory estimates in /list and the offload planner follow " +
					"this setting, so a smaller cache type genuinely raises what " +
					"`gpu_layers auto` will offload."},
			{Name: "cache_type_v", Usage: "/set cache_type_v auto|q4_0|q4_1|q5_0|q5_1|q8_0|f16|bf16|f32",
				Detail: "Quantization of the V half of the KV cache. `auto` is q8_0.\n\n" +
					"Quantizing V requires flash attention — llama-server refuses " +
					"the combination at load, so /set refuses it earlier: a " +
					"quantized value here conflicts with `flash_attn off`.\n\n" +
					"V tolerates quantization better than K, so if VRAM is desperate " +
					"q4_0 here is the cheaper of the two halvings."},
			{Name: "threads", Usage: "/set threads auto|N",
				Detail: "CPU threads for the engine (llama.cpp -t). `auto` is NumCPU-1 " +
					"capped at 6 — beyond that, extra threads fight the GPU feeding " +
					"loop for cache without prompt throughput to show for it.\n\n" +
					"Raise it on CPU-only setups or big partial offloads, where the " +
					"CPU genuinely does the generating and the cap costs real " +
					"speed. Lower it if the machine turns sluggish while a reply " +
					"streams."},
			{Name: "batch_size", Usage: "/set batch_size auto|N",
				Detail: "Logical batch size for prompt processing (llama.cpp -b). " +
					"`auto` keeps llama.cpp's default, 2048.\n\n" +
					"Mostly acts as the ceiling for ubatch_size. Lowering both " +
					"shrinks the compute buffers a launch allocates in VRAM, which " +
					"can be the difference for a model that almost fits — paid for " +
					"in slower prefill."},
			{Name: "ubatch_size", Usage: "/set ubatch_size auto|N",
				Detail: "Physical batch per engine step (llama.cpp -ub), and the knob " +
					"that actually sizes the compute buffers in VRAM. `auto` keeps " +
					"llama.cpp's default, 512.\n\n" +
					"128 or 256 can recover a few hundred MiB when a model almost " +
					"fits; prompt processing slows roughly in proportion. Values " +
					"above batch_size do nothing."},
			{Name: "parallel", Usage: "/set parallel auto|N",
				Detail: "Server slots — how many requests are handled concurrently " +
					"(llama.cpp --parallel). `auto` is 2: one for the conversation, " +
					"one so a /compact or /summarize doesn't evict the " +
					"conversation's KV cache.\n\n" +
					"The KV budget is fixed at what ctx_size implies: more slots " +
					"divide it, so each slot sees proportionally less context, and " +
					"VRAM use never grows with this setting. Mainly for --serve " +
					"machines feeding several clients; a --slots flag on the " +
					"command line still outranks it."},
			{Name: "cache_reuse", Usage: "/set cache_reuse auto|off|N",
				Detail: "Minimum length, in tokens, of a still-matching KV chunk worth " +
					"salvaging past the first divergence instead of reprocessing " +
					"everything after it (llama.cpp --cache-reuse). `auto` is 256.\n\n" +
					"Its main effect here is softening the full re-prefill after " +
					"/compact rewrites history. `off` disables salvage entirely — " +
					"try it if a model produces oddities right after a compact. " +
					"Runtime no-op for sliding-window models such as Gemma."},
			{Name: "mmap", Usage: "/set mmap on|off",
				Detail: "How the weights are loaded. `on` (the default) memory-maps " +
					"the GGUF: startup is instant and the OS pages weights in on " +
					"demand. `off` (llama.cpp --no-mmap) copies the whole file into " +
					"RAM up front — slower start, no first-token page-fault stalls, " +
					"and on some setups measurably faster CPU inference.\n\n" +
					"`off` needs RAM for the entire model file. Check the fit " +
					"column in /list first."},
			{Name: "mlock", Usage: "/set mlock on|off",
				Detail: "`on` pins the model's pages in memory (llama.cpp --mlock) so " +
					"the OS cannot swap them out under memory pressure — the cause " +
					"of a model that answers instantly for an hour and then stalls " +
					"mid-reply after something else used the RAM.\n\n" +
					"The price is that the RAM is genuinely gone for everything " +
					"else, and the model must actually fit. Some systems also need " +
					"a raised memlock limit; a launch failure right after enabling " +
					"this is that limit."},
			{Name: "seed", Usage: "/set seed auto|N",
				Detail: "Sampling seed (llama.cpp --seed). `auto` picks a fresh seed " +
					"per generation. A fixed number makes the same prompt at the " +
					"same settings reproduce the same output — useful when " +
					"comparing models or debugging a prompt, pointless for normal " +
					"chat.\n\n" +
					"Reproducibility also needs identical context: any difference " +
					"in history changes the output regardless of the seed."},
			{Name: "temperature", Usage: "/set temperature auto|X",
				Detail: "Sampling temperature, sent with every request rather than " +
					"bound at server launch — so it needs no restart, and it is the " +
					"one tuning setting that still applies when inference runs on a " +
					"remote endpoint.\n\n" +
					"`auto` is 0.2: low, because tool calls and code want " +
					"determinism. 0.7–0.9 is the usual range for creative prose; 0 " +
					"is greedy decoding — the same continuation every time. Range 0 " +
					"to 2."},
			{Name: "override_tensor", Usage: "/set override_tensor PATTERN=BACKEND | off",
				Detail: "Raw per-tensor placement (llama.cpp -ot), the power tool " +
					"behind most \"run a huge MoE on a small card\" recipes: " +
					"`exps=CPU` keeps every expert tensor in system RAM while all " +
					"attention stays on the GPU — finer-grained than the automatic " +
					"--n-cpu-moe, which moves experts whole layers at a time.\n\n" +
					"The pattern is a regex matched against tensor names, and the " +
					"engine — not atlas.llm — interprets it. The offload estimator " +
					"does not model it, so /config's VRAM arithmetic can be wrong " +
					"while one is set. A bad pattern fails at launch, loudly. `off` " +
					"clears it."},
			{Name: "context_shift", Usage: "/set context_shift on|off",
				Detail: "`on` lets the server slide the window when the context " +
					"fills — the oldest tokens are dropped and generation " +
					"continues (llama.cpp --context-shift). The model silently " +
					"forgets the start of the conversation.\n\n" +
					"`off` (the default) surfaces the limit instead, and /compact " +
					"does the same job while telling you what it kept. Needs an " +
					"engine recent enough to know the flag, and some models are " +
					"incompatible with shifting."},
		},
		Notes: []string{
			"`auto` never selects a GPU build on Windows or Linux. A GPU build " +
				"without a matching driver fails at load and there's no reliable way " +
				"to detect one, so that choice is left to you.",
		},
		SeeAlso: []string{"download"},
	},
	{
		Name:    "browser",
		Summary: "Inspect and clear the persistent browser profile.",
		Usage:   []string{"/browser", "/browser clear", "/browser clear chrome"},
		Detail: "The model can open a browser on three kinds of profile. Two of them " +
			"are throwaway temp directories deleted the moment the window " +
			"closes, and there is nothing to manage. The third — the one it uses " +
			"when you ask it to remember a session — is a directory atlas.llm " +
			"keeps under the data dir, one per browser family, and that one " +
			"survives on purpose.\n\n" +
			"That means cookies and logins accumulate there indefinitely. " +
			"/browser with no argument shows where each profile lives, how much " +
			"disk it has grown to, when it was last used, and whether a window " +
			"has it open right now.\n\n" +
			"Only one session may use a persistent profile at a time. A second " +
			"atlas.llm asking for it is refused rather than allowed to share it, " +
			"because two browsers writing one cookie store corrupt it. A profile " +
			"left locked by a run that crashed is taken over automatically.\n\n" +
			"Caches inside it are dropped on each relaunch, so it grows with " +
			"what you sign into rather than with what you browse.",
		Subcommands: []helpSub{
			{Name: "clear", Usage: "/browser clear [chrome|firefox]",
				Detail: "Delete the profile, signing out of every site in it. " +
					"With no family, clears both. Refused while a window has it open."},
		},
		Examples: [][2]string{
			{"/browser", "where the profiles are and how big they have grown"},
			{"/browser clear chrome", "sign out of everything in the Chrome profile"},
		},
		Notes: []string{
			"The persistent profile is separate from your real browser profile, " +
				"which atlas.llm never opens or writes to.",
			"It is listed in /config's FILES section once it exists.",
		},
		SeeAlso: []string{"tools"},
	},
	{
		Name:    "tools",
		Summary: "Let the model read, edit, and run things in your project.",
		Usage:   []string{"/tools", "/tools on", "/tools off", "/tools list"},
		Detail: "Off by default. When on, the model can call tools instead of " +
			"guessing — reading files, searching, editing, running commands — and " +
			"loops until it has what it needs before answering.\n\n" +
			"Sixteen built-in tools. read_file, list_dir, and grep are read-only " +
			"and run silently. write_file, edit_file, multi_edit, run_cmd, " +
			"web_fetch, browser_open, and browser_upload are destructive and open " +
			"a confirmation modal first: Enter approves, Esc denies. A denial is " +
			"fed back to the model as a tool error so it adapts instead of " +
			"retrying the same call.\n\n" +
			"Edits are partial: edit_file replaces one unique string and " +
			"multi_edit applies a batch of them atomically, both leaving the rest " +
			"of the file untouched. Only write_file rewrites a whole file.\n\n" +
			"web_fetch reads one web page as text — navigation and boilerplate " +
			"stripped, links kept absolute so the model can follow one. It needs " +
			"confirmation because it is outbound, not because it writes anything. " +
			"It reaches the public internet only: loopback, LAN and link-local " +
			"addresses are refused, at every redirect hop as well as the first. " +
			"It reads text formats and does not run JavaScript, so a page that " +
			"builds itself in the browser comes back empty and says so.\n\n" +
			"browser_open launches a visible Chrome or Firefox window on a " +
			"throwaway profile — none of your logins or history — and only the " +
			"launch needs confirmation. From there browser_navigate loads pages, " +
			"browser_read returns their text, links, HTML, or console output " +
			"(including JavaScript errors, and any alert/confirm/prompt dialogs, " +
			"which are auto-answered so they can never hang the session), " +
			"browser_act drives the page (click, type, hover, select, clear, get, " +
			"scroll, wait, back/forward/reload, or raw eval), browser_screenshot " +
			"saves a PNG of the page or of one element, browser_tabs lists, " +
			"switches, opens, and closes tabs, browser_upload attaches a local " +
			"file to a file-upload field (confirmed first, since it hands the " +
			"file to a website), and browser_close ends the session and deletes " +
			"the profile. browser_act targets elements by their " +
			"visible text — click \"Sign in\", type into \"Search\" — not just CSS " +
			"selectors, and a target it can't find comes back as an error naming " +
			"the miss instead of failing silently. Every element-targeted action " +
			"first glides a small fake cursor to its target and flashes a " +
			"highlight ring around it, so you can follow what the model is doing. " +
			"You watch the window the whole " +
			"time; pages that need JavaScript work here, unlike web_fetch.\n\n" +
			"Ask to browse as yourself and it opens on a copy of your real " +
			"browser profile instead, so your existing logins are available. It " +
			"is always a copy — your actual profile is never opened or changed, " +
			"and new logins in the window are not saved back to it.\n\n" +
			"For a site you return to, ask to keep the session and it launches on " +
			"a persistent profile atlas.llm owns (under the data dir, one per " +
			"browser family). Unlike the throwaway, this one is not deleted on " +
			"close, so cookies and signed-in sessions survive — sign in once and " +
			"the next launch is still logged in. Its first launch copies your " +
			"real profile, so you start out already signed in where you were. " +
			"See /help browser for where it lives and how to clear it.\n\n" +
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
			"Needs a model that can emit tool calls — Qwen3.5-9B, " +
				"Ministral-3-14B, or qwen3.8-27b (the strongest, needs a " +
				"16GB card). On Gemma 3 the feature will appear broken.",
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
		Summary: "Show the whole setup, or save/load named profiles.",
		Usage:   []string{"/config", "/config save <name>", "/config load <name>", "/config show <name>", "/config list", "/config delete <name>"},
		Detail: "With no argument, prints every persisted setting with its current " +
			"value, the active model and whether it's downloaded, the installed " +
			"engine build, the session-only state (tool-use, yesman, ama, MCP " +
			"connections), how much memory the model server is using, and where " +
			"everything lives on disk. If the current settings match a saved " +
			"profile, its name is shown at the top.\n\n" +
			"The subcommands manage named profiles — whole snapshots of your " +
			"settings you can switch between with one command:\n" +
			"  /config save <name>    snapshot the current settings under a name\n" +
			"  /config load <name>    make that profile the active config\n" +
			"  /config show <name>    print a profile's settings without loading it\n" +
			"  /config list           list saved profiles (● marks the active one)\n" +
			"  /config delete <name>  remove a profile\n\n" +
			"A profile captures everything in config.json — model, context size, " +
			"gpu_layers, reasoning, the engine-tuning settings, endpoint, tools " +
			"and ama. The classic use is a `fast` profile (small model, small " +
			"context, reasoning off) next to a `quality` one, flipped with a " +
			"single /config load.\n\n" +
			"Loading restarts the model server on your next message, since the " +
			"model or context may have changed. Saving never touches the active " +
			"config — it's a snapshot, not a switch.",
		Subcommands: []helpSub{
			{Name: "save", Usage: "/config save fast",
				Detail: "Snapshot the current settings as a named profile. Names use " +
					"letters, digits, dash, and underscore. Saving over an existing " +
					"name overwrites it."},
			{Name: "load", Usage: "/config load fast",
				Detail: "Make a saved profile the active config. The model server " +
					"restarts on your next message, and session state (tools, ama, " +
					"remote endpoint) is re-synced to the profile."},
			{Name: "show", Usage: "/config show fast",
				Detail: "Print a profile's saved settings without switching to it — " +
					"the model, context size, tuning, tools and ama it would apply. " +
					"Marks the profile ● active if it matches your current config."},
			{Name: "list", Usage: "/config list",
				Detail: "List saved profiles. A ● marks the one that matches your " +
					"current settings exactly; no mark means you've changed " +
					"something since loading."},
			{Name: "delete", Usage: "/config delete fast",
				Detail: "Remove a saved profile. The active config is untouched."},
		},
		Notes: []string{
			"The memory figure is measured from the running model server, not " +
				"predicted. It reads as \"not running\" until the first message, " +
				"since the server starts lazily.",
			"Profiles live in profiles/ under the data dir, one JSON file each — " +
				"the same shape as config.json, so they're easy to inspect or edit.",
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
		Name:    "ama",
		Summary: "Let the agent ask you questions with interactive lists.",
		Usage:   []string{"/ama", "/ama on", "/ama off"},
		Detail: "\"Ask me anything\" — with it on, the agent gains one extra tool, " +
			"ask_user, that turns a decision back to you as an interactive picker " +
			"instead of guessing. Use it when you'd rather be consulted on " +
			"ambiguous requests than have the model pick a direction on its own.\n\n" +
			"With no argument it shows the current state; `on` and `off` set it. " +
			"The preference is saved, so it carries across restarts.\n\n" +
			"It only does anything while tools are on (`/tools on`) — ask_user is " +
			"a tool, and it's offered to the model only during an agent turn. When " +
			"tools are off the setting is remembered but dormant.\n\n" +
			"Four kinds of question, chosen by the model to fit the moment:\n" +
			"  radio          pick exactly one option\n" +
			"  checkbox       pick any number (space toggles, enter submits)\n" +
			"  confirm        a yes/no step confirmation before it acts\n" +
			"  answer_confirm approve a drafted answer or plan, or ask for changes\n\n" +
			"In the picker: ↑/↓ move, space toggles a checkbox, enter submits, " +
			"esc dismisses. Dismissing isn't a dead end — the model is told to " +
			"proceed with its best judgment, so a turn never hangs on a question " +
			"you'd rather skip.",
		Notes: []string{
			"The agent decides when to ask. A capable model asks only when a " +
				"choice is genuinely yours; a weaker one may ask too often or not " +
				"at all — turning it off restores decide-on-its-own behavior.",
			"Your answer is fed back as the tool result, so it steers the rest " +
				"of the turn exactly like any other tool output.",
		},
		SeeAlso: []string{"tools", "yesman"},
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
	head := lipgloss.NewStyle().Foreground(ui.ColDim).Bold(true)
	cmd := lipgloss.NewStyle().Foreground(ui.ColAccent).Bold(true)
	dim := lipgloss.NewStyle().Foreground(ui.ColMuted)

	title := "/" + t.Name
	if len(t.Aliases) > 0 {
		for _, a := range t.Aliases {
			title += ", /" + a
		}
	}

	lines := []string{ui.BrandStyle.Render(title) + "  " + dim.Render(t.Summary), ""}

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
	cmd := lipgloss.NewStyle().Foreground(ui.ColAccent).Bold(true)
	dim := lipgloss.NewStyle().Foreground(ui.ColMuted)

	lines := []string{
		ui.BrandStyle.Render("/"+t.Name+" "+s.Name) + "  " + dim.Render("subcommand of /"+t.Name),
		"",
		"    " + cmd.Render(s.Usage),
		"",
	}
	for _, l := range wrapText(s.Detail, body) {
		lines = append(lines, "  "+l)
	}
	lines = append(lines, "", lipgloss.NewStyle().Foreground(ui.ColDim).Bold(true).
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
	dim := lipgloss.NewStyle().Foreground(ui.ColMuted)
	acc := lipgloss.NewStyle().Foreground(ui.ColAccent).Bold(true)
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
