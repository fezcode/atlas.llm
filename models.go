package main

// The model registry: every model atlas.llm knows how to download and run,
// lightest first. Pure data — the accessors that search it live in config.go.

type Model struct {
	Name     string `json:"name"`
	Filename string `json:"filename"`
	URL      string `json:"url"`
	Size     string `json:"size"`
}

var availableModels = []Model{
	{
		Name:     "gemma-3-1b-it",
		Filename: "gemma-3-1b-it-Q4_K_M.gguf",
		URL:      "https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf",
		Size:     "~700MB",
	},
	{
		Name:     "gemma-3-4b-it",
		Filename: "gemma-3-4b-it-Q4_K_M.gguf",
		URL:      "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf",
		Size:     "~2.5GB",
	},
	{
		// The lightest model in the registry that reliably emits tool calls,
		// so /tools and /mcp actually work without pulling the 5.7GB 9B.
		Name:     "qwen3.5-4b",
		Filename: "Qwen3.5-4B-Q4_K_M.gguf",
		URL:      "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
		Size:     "~2.7GB",
	},
	{
		Name:     "gemma-4-e2b-it",
		Filename: "gemma-4-E2B-it-Q4_K_M.gguf",
		URL:      "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf",
		Size:     "~2.9GB",
	},
	{
		// Gemma 4, unlike Gemma 3, ships a tool-calling chat template, so
		// this family can drive /tools and /mcp.
		Name:     "gemma-4-e4b-it",
		Filename: "gemma-4-E4B-it-Q4_K_M.gguf",
		URL:      "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf",
		Size:     "~5.0GB",
	},
	{
		Name:     "gemma-4-12b-it",
		Filename: "gemma-4-12b-it-Q4_K_M.gguf",
		URL:      "https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/main/gemma-4-12b-it-Q4_K_M.gguf",
		Size:     "~7.1GB",
	},
	{
		// Fills the gap between qwen3.5-4b and qwen3.5-9b. Same family as
		// the 14B below, which already tool-calls reliably.
		Name:     "ministral-3-8b-instruct",
		Filename: "Ministral-3-8B-Instruct-2512-Q4_K_M.gguf",
		URL:      "https://huggingface.co/unsloth/Ministral-3-8B-Instruct-2512-GGUF/resolve/main/Ministral-3-8B-Instruct-2512-Q4_K_M.gguf",
		Size:     "~5.2GB",
	},
	{
		// Ornith-1.5's dense 9B (Aug 2026, MIT) — the self-improvement-trained
		// family, pitched at reasoning and agentic work. Tool-calling chat
		// template (<tool_call> blocks), 256K native context. First-party
		// GGUF — no unsloth repo yet. Text-only: vision needs the separate
		// mmproj GGUF, which we don't download. New arch, so it needs a
		// recent llama.cpp build (`/download engine` refreshes an older
		// install).
		Name:     "ornith-1.5-9b",
		Filename: "Ornith-1.5-9B-Q4_K_M.gguf",
		URL:      "https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF/resolve/main/Ornith-1.5-9B-Q4_K_M.gguf",
		Size:     "~5.6GB",
	},
	{
		Name:     "qwen3.5-9b",
		Filename: "Qwen3.5-9B-Q4_K_M.gguf",
		URL:      "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
		Size:     "~5.7GB",
	},
	{
		Name:     "ministral-3-14b-instruct",
		Filename: "Ministral-3-14B-Instruct-2512-Q4_K_M.gguf",
		URL:      "https://huggingface.co/unsloth/Ministral-3-14B-Instruct-2512-GGUF/resolve/main/Ministral-3-14B-Instruct-2512-Q4_K_M.gguf",
		Size:     "~8.2GB",
	},
	{
		// First mixture-of-experts entry, and the first model aimed at code.
		// 30B of weights with 3B active per token, so it generates at roughly
		// 4B speed while knowing what a 30B knows — which is the trade a
		// 16GB card wants. IQ3 rather than the nicer Q4_K_M (18.6GB) because
		// it sits almost entirely in 16GB — measured against a 5070 Ti with
		// a desktop running, five of its 48 layers spill their experts to
		// system RAM. Q4_K_M works too and is the better model; it spills
		// about twenty.
		Name:     "qwen3-coder-30b-a3b",
		Filename: "Qwen3-Coder-30B-A3B-Instruct-UD-IQ3_XXS.gguf",
		URL:      "https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct-UD-IQ3_XXS.gguf",
		Size:     "~12.8GB",
	},
	{
		// The general-purpose counterpart to the coder above: 26B of weights,
		// 4B active. Same family as the gemma-4 entries, so it carries the
		// same tool-calling chat template.
		Name:     "gemma-4-26b-a4b-it",
		Filename: "gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf",
		URL:      "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf",
		Size:     "~12.9GB",
	},
	{
		// The MoE of the Ornith-1.5 family above: 35B of weights, 3B active
		// per token, so it generates at small-model speed like the coder
		// entry. First-party quants start at Q4_K_M (21.7GB) — too big for
		// the reference card — so this is bartowski's IQ3_XXS, the largest
		// cut that stays under 16GB. At 15.3GB more expert layers spill to
		// system RAM than the coder's, which MoE routing keeps affordable;
		// a dense model this size would not survive the same spill.
		Name:     "ornith-1.5-35b-a3b",
		Filename: "Ornith-1.5-35B-A3B-IQ3_XXS.gguf",
		URL:      "https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/resolve/main/Ornith-1.5-35B-A3B-IQ3_XXS.gguf",
		Size:     "~15.3GB",
	},
	{
		// Meta's local-agent model (Aug 2026), built around tool calling.
		// First entry aimed at 24/32GB machines — the fit column reads
		// "too big" on 16GB, and no quant of this dense 30B fits there.
		// unsloth ships only UD dynamic quants, hence no Q4_K_M. Text-only:
		// vision needs the separate mmproj GGUF, which we don't download.
		Name:     "muse-glimmer-30b",
		Filename: "Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
		URL:      "https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF/resolve/main/Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
		Size:     "~15.9GB",
	},
	{
		// Qwen's dense 27B (Aug 2026, Apache 2.0), aimed squarely at agentic
		// work — the strongest /tools driver in the registry. UD-Q3_K_XL
		// rather than Q4_K_M (17.1GB) because a dense model pays for every
		// spilled layer on every token; this quant sits entirely in 16GB
		// with room for the KV cache, which its hybrid attention keeps
		// small (most layers are linear-attention, like Qwen3.5). Text-only
		// here: vision needs the separate mmproj GGUF, which we don't
		// download. Needs a llama.cpp build recent enough to know the arch
		// (`/download engine` refreshes an older install).
		Name:     "qwen3.8-27b",
		Filename: "Qwen3.8-27B-UD-Q3_K_XL.gguf",
		URL:      "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q3_K_XL.gguf",
		Size:     "~13.4GB",
	},
	{
		// The 2-bit cut of the dense 27B above, kept for context rather
		// than quality: at ~9GB the weights leave a 12GB card room for a
		// ~150K-token q4_0 KV cache entirely on the GPU — affordable only
		// because the hybrid attention holds KV in a fraction of the
		// layers. 2-bit is audibly lossy; when 150K isn't the point, the
		// Q3_K_XL entry above is the better model.
		Name:     "qwen3.8-27b-iq2",
		Filename: "Qwen3.8-27B-UD-IQ2_XXS.gguf",
		URL:      "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-IQ2_XXS.gguf",
		Size:     "~9.0GB",
	},
	{
		// Abliterated cut of the dense 27B above — refusal directions ablated
		// so it answers where the stock model declines. Of the abliterations
		// published for qwen3.8, this Heretic ARA build is the one that
		// hardly costs quality: KL divergence 0.0085 against the base
		// (huihui's first-party attempt measures 6x worse) with refusals at
		// ~0/100. Q3_K_M so it sits entirely in 16GB like the stock entry;
		// the RVN- filename is the quantizer's own prefix, which the URL
		// must match. Text-only — this abliteration drops the vision stack.
		Name:     "qwen3.8-27b-heretic",
		Filename: "RVN-Q3_K_M.gguf",
		URL:      "https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF/resolve/main/RVN-Q3_K_M.gguf",
		Size:     "~13.3GB",
	},
	{
		// The Q4_K_M cut of the Heretic abliteration above, for 24GB cards —
		// on 16GB a dense model pays for every spilled layer on every token,
		// so the Q3_K_M entry is the right pick there. Same repo, same
		// weights, one quant step nicer.
		Name:     "qwen3.8-27b-heretic-q4",
		Filename: "Qwen3.8-27B-Heretic-Q4_K_M.gguf",
		URL:      "https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF/resolve/main/Qwen3.8-27B-Heretic-Q4_K_M.gguf",
		Size:     "~16.5GB",
	},
}

const defaultModel = "gemma-3-1b-it"
