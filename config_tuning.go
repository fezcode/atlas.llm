package main

import (
	"fmt"
	"runtime"
	"strconv"
	"strings"
)

// This file is the engine-tuning settings: fifteen knobs that map onto
// llama-server flags (temperature onto a per-request field). Each is one
// Config field, one registry entry with an Apply, and one place in
// buildServerArgs — nothing here launches anything.

// defaultTemperature is what every request sent before the setting existed.
const defaultTemperature = 0.2

// defaultCacheReuse is the --cache-reuse the optimized launch always passed.
const defaultCacheReuse = 256

// cacheTypeBytes maps a llama.cpp cache type to its per-element cost. The
// quantized types pack 32 elements into a fixed block: q8_0 is 34 bytes,
// q5_1 is 24, q5_0 is 22, q4_1 is 20, q4_0 is 18.
var cacheTypeBytes = map[string]float64{
	"f32":  4,
	"f16":  2,
	"bf16": 2,
	"q8_0": 34.0 / 32.0,
	"q5_1": 24.0 / 32.0,
	"q5_0": 22.0 / 32.0,
	"q4_1": 20.0 / 32.0,
	"q4_0": 18.0 / 32.0,
}

// cacheTypeElemBytes returns the per-element KV cost for a cache type.
// Unknown types report f16 — erring high is the estimator's stance
// everywhere: a fit figure that overstates never strands a user mid-load.
func cacheTypeElemBytes(t string) float64 {
	if b, ok := cacheTypeBytes[strings.ToLower(t)]; ok {
		return b
	}
	return 2
}

func validCacheTypeNames() []string {
	// Stable order for error messages, cheapest first.
	return []string{"q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "bf16", "f32"}
}

func isValidCacheType(t string) bool {
	_, ok := cacheTypeBytes[strings.ToLower(t)]
	return ok
}

// cacheTypeIsQuantized reports whether a type needs flash attention when
// used for the V cache — llama-server refuses a quantized V without it.
func cacheTypeIsQuantized(t string) bool {
	switch strings.ToLower(t) {
	case "", "f16", "f32", "bf16":
		return false
	}
	return true
}

func kvOffloadEnabled(cfg Config) bool { return cfg.KVOffload != "off" }

// vramKVCharge is what the VRAM fit math should charge for the KV cache.
// With kv_offload off the cache lives in system RAM and VRAM pays nothing,
// which is exactly why the setting exists: more weight layers fit.
func vramKVCharge(cfg Config, kv int64) int64 {
	if !kvOffloadEnabled(cfg) {
		return 0
	}
	return kv
}

// resolveThreads is the -t value: explicit, or NumCPU-1 capped at 6 — more
// threads than that fight the GPU feeding loop for cache without prompt
// throughput to show for it.
func resolveThreads(cfg Config) int {
	if cfg.Threads > 0 {
		return cfg.Threads
	}
	threads := runtime.NumCPU() - 1
	if threads < 1 {
		threads = 1
	}
	if threads > 6 {
		threads = 6
	}
	return threads
}

// resolveFlashAttn is the -fa value on the optimized flag set, where auto
// has always forced it on (it is a prefill speedup and the price of
// quantizing V).
func resolveFlashAttn(cfg Config) string {
	if cfg.FlashAttn != "" {
		return cfg.FlashAttn
	}
	return "on"
}

// resolveCacheType is the K or V cache type on the optimized flag set,
// where auto has always meant q8_0.
func resolveCacheType(explicit string) string {
	if explicit != "" {
		return explicit
	}
	return "q8_0"
}

func resolveCacheReuse(cfg Config) int {
	if cfg.CacheReuse != nil {
		return *cfg.CacheReuse
	}
	return defaultCacheReuse
}

func resolveTemperature(cfg Config) float64 {
	if cfg.Temperature != nil {
		return *cfg.Temperature
	}
	return defaultTemperature
}

// requestTemperature reads the setting per request rather than binding it at
// server launch — that is what lets /set temperature apply without a restart,
// locally and against a remote alike. One config read per message, not per
// frame.
func requestTemperature() float64 {
	cfg, _ := loadConfig()
	return resolveTemperature(cfg)
}

// kvElemBytesFor is the per-element KV cost the launch will actually pay,
// averaged over K and V. On a zero config this is exactly the q8_0 constant
// the estimator has always used.
func kvElemBytesFor(cfg Config) float64 {
	k := cacheTypeElemBytes(resolveCacheType(cfg.CacheTypeK))
	v := cacheTypeElemBytes(resolveCacheType(cfg.CacheTypeV))
	return (k + v) / 2
}

// tuningFingerprint is the restart identity of the tuning settings: the raw
// configured values, never resolved autos, so estimate drift can't evict a
// running server but a user change always does. temperature is deliberately
// absent — it rides on each request and needs no relaunch.
func tuningFingerprint(cfg Config) string {
	optInt := func(p *int) string {
		if p == nil {
			return "auto"
		}
		return strconv.Itoa(*p)
	}
	return fmt.Sprintf("kvo=%s fa=%s ctk=%s ctv=%s t=%d b=%d ub=%d par=%d cr=%s mmap=%s mlock=%s seed=%s ot=%s cs=%s",
		cfg.KVOffload, cfg.FlashAttn, cfg.CacheTypeK, cfg.CacheTypeV,
		cfg.Threads, cfg.BatchSize, cfg.UBatchSize, cfg.Parallel,
		optInt(cfg.CacheReuse), cfg.Mmap, cfg.Mlock, optInt(cfg.Seed),
		cfg.OverrideTensor, cfg.ContextShift)
}

// tuningArgs are the launch-set-independent flags: they mean the same thing
// on the optimized and fallback flag sets, so both get them.
func tuningArgs(cfg Config) []string {
	var args []string
	if !kvOffloadEnabled(cfg) {
		args = append(args, "--no-kv-offload")
	}
	if cfg.Mmap == "off" {
		args = append(args, "--no-mmap")
	}
	if cfg.Mlock == "on" {
		args = append(args, "--mlock")
	}
	if cfg.ContextShift == "on" {
		args = append(args, "--context-shift")
	}
	if cfg.Seed != nil {
		args = append(args, "--seed", strconv.Itoa(*cfg.Seed))
	}
	if cfg.BatchSize > 0 {
		args = append(args, "-b", strconv.Itoa(cfg.BatchSize))
	}
	if cfg.UBatchSize > 0 {
		args = append(args, "-ub", strconv.Itoa(cfg.UBatchSize))
	}
	if cfg.OverrideTensor != "" {
		args = append(args, "-ot", cfg.OverrideTensor)
	}
	return args
}

// --- Apply helpers -------------------------------------------------------

// parseOnOff normalizes a boolean-ish value. defaultState is what the empty
// config field means, so choosing it clears the field back to zero.
func parseOnOff(key, val string, set func(string)) error {
	switch strings.ToLower(strings.TrimSpace(val)) {
	case "on", "true", "yes":
		set("on")
	case "off", "false", "no":
		set("off")
	default:
		return fmt.Errorf("invalid %s=%q (expected on or off)", key, val)
	}
	return nil
}

// parseAutoInt parses "auto" (→0) or a positive integer with bounds.
func parseAutoInt(key, val string, min, max int, set func(int)) error {
	v := strings.ToLower(strings.TrimSpace(val))
	if v == "auto" || v == "default" {
		set(0)
		return nil
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < min || n > max {
		return fmt.Errorf("invalid %s=%q (expected `auto` or %d..%d)", key, val, min, max)
	}
	set(n)
	return nil
}

// onOffValue renders a default-on boolean field.
func onOffValue(field, def string) string {
	if field == "" {
		return fmt.Sprintf("auto (%s)", def)
	}
	return field
}

var tuningSettings = []setting{
	{
		Key:     "kv_offload",
		Usage:   "/set kv_offload on|off",
		Summary: "Where the KV cache lives: GPU (on) or system RAM (off).",
		Restart: true,
		Value:   func(c Config) string { return onOffValue(c.KVOffload, "on") },
		Detail: func(c Config) string {
			return "off keeps every weight layer on the GPU but allocates the KV cache " +
				"in system RAM (llama.cpp --no-kv-offload). The fit estimator stops " +
				"charging the cache against VRAM, so `gpu_layers auto` will offload " +
				"more layers.\n" +
				"The trade: attention reads the cache on every generated token, so " +
				"generation slows — usually more than spilling one or two weight " +
				"layers would. It wins when the cache is what doesn't fit (huge " +
				"contexts), not the weights."
		},
		Apply: func(c *Config, val string) error {
			return parseOnOff("kv_offload", val, func(v string) {
				if v == "on" {
					v = ""
				}
				c.KVOffload = v
			})
		},
	},
	{
		Key:     "flash_attn",
		Usage:   "/set flash_attn auto|on|off",
		Summary: "Flash attention (llama.cpp -fa).",
		Restart: true,
		Value: func(c Config) string {
			if c.FlashAttn == "" {
				return "auto (on)"
			}
			return c.FlashAttn
		},
		Detail: func(c Config) string {
			return "auto forces it on for the optimized launch — it speeds up prefill " +
				"and is required to quantize the V cache. off is for backends where " +
				"forcing it breaks the model; it also forbids a quantized " +
				"cache_type_v.\n" +
				"If a launch fails right after changing models, try off before " +
				"anything else."
		},
		Apply: func(c *Config, val string) error {
			v := strings.ToLower(strings.TrimSpace(val))
			switch v {
			case "auto", "default":
				c.FlashAttn = ""
				return nil
			case "on", "off":
				if v == "off" && cacheTypeIsQuantized(c.CacheTypeV) {
					return fmt.Errorf("flash_attn off conflicts with cache_type_v=%s — "+
						"llama-server refuses a quantized V cache without flash attention. "+
						"`/set cache_type_v f16` first", c.CacheTypeV)
				}
				c.FlashAttn = v
				return nil
			}
			return fmt.Errorf("invalid flash_attn=%q (expected auto, on, or off)", val)
		},
	},
	{
		Key:     "cache_type_k",
		Usage:   "/set cache_type_k auto|" + strings.Join(validCacheTypeNames(), "|"),
		Summary: "Quantization of the K half of the KV cache.",
		Restart: true,
		Value: func(c Config) string {
			if c.CacheTypeK == "" {
				return "auto (q8_0)"
			}
			return c.CacheTypeK
		},
		Detail: func(c Config) string {
			return "auto is q8_0: half the memory of f16 at no measurable quality " +
				"cost, which is what pays for the second server slot.\n" +
				"q4_0 halves it again — the K cache tolerates this worse than V; " +
				"expect degradation on long contexts. f16 is the full-precision " +
				"fallback if a model misbehaves.\n" +
				"The memory estimates in /list and the offload planner follow this " +
				"setting."
		},
		Apply: func(c *Config, val string) error {
			v := strings.ToLower(strings.TrimSpace(val))
			if v == "auto" || v == "default" {
				c.CacheTypeK = ""
				return nil
			}
			if !isValidCacheType(v) {
				return fmt.Errorf("invalid cache_type_k=%q (expected auto or %s)",
					val, strings.Join(validCacheTypeNames(), ", "))
			}
			c.CacheTypeK = v
			return nil
		},
	},
	{
		Key:     "cache_type_v",
		Usage:   "/set cache_type_v auto|" + strings.Join(validCacheTypeNames(), "|"),
		Summary: "Quantization of the V half of the KV cache.",
		Restart: true,
		Value: func(c Config) string {
			if c.CacheTypeV == "" {
				return "auto (q8_0)"
			}
			return c.CacheTypeV
		},
		Detail: func(c Config) string {
			return "auto is q8_0. Quantizing V requires flash attention, so any " +
				"quantized value conflicts with flash_attn off — the conflict is " +
				"refused here rather than failing at launch.\n" +
				"V tolerates quantization better than K; q4_0 here is the cheaper " +
				"of the two halvings if VRAM is desperate."
		},
		Apply: func(c *Config, val string) error {
			v := strings.ToLower(strings.TrimSpace(val))
			if v == "auto" || v == "default" {
				c.CacheTypeV = ""
				return nil
			}
			if !isValidCacheType(v) {
				return fmt.Errorf("invalid cache_type_v=%q (expected auto or %s)",
					val, strings.Join(validCacheTypeNames(), ", "))
			}
			if cacheTypeIsQuantized(v) && c.FlashAttn == "off" {
				return fmt.Errorf("cache_type_v=%s needs flash attention, and flash_attn "+
					"is off. `/set flash_attn auto` first", v)
			}
			c.CacheTypeV = v
			return nil
		},
	},
	{
		Key:     "threads",
		Usage:   "/set threads auto|N",
		Summary: "CPU threads for the engine (llama.cpp -t).",
		Restart: true,
		Value: func(c Config) string {
			if c.Threads == 0 {
				return fmt.Sprintf("auto (%d)", resolveThreads(c))
			}
			return fmt.Sprintf("%d", c.Threads)
		},
		Detail: func(c Config) string {
			return fmt.Sprintf("auto is NumCPU-1 capped at 6 (currently %d) — beyond "+
				"that, extra threads fight the GPU feeding loop for cache without "+
				"prompt throughput to show for it.\n"+
				"Raise it on CPU-only setups or big partial offloads, where the CPU "+
				"genuinely does the generating. Lower it if the machine gets sluggish "+
				"while a reply streams.", resolveThreads(Config{}))
		},
		Apply: func(c *Config, val string) error {
			return parseAutoInt("threads", val, 1, 1024, func(n int) { c.Threads = n })
		},
	},
	{
		Key:     "batch_size",
		Usage:   "/set batch_size auto|N",
		Summary: "Logical batch for prompt processing (llama.cpp -b).",
		Restart: true,
		Value: func(c Config) string {
			if c.BatchSize == 0 {
				return "auto (2048)"
			}
			return fmt.Sprintf("%d", c.BatchSize)
		},
		Detail: func(c Config) string {
			return "How many tokens are submitted per engine step during prefill. " +
				"auto keeps llama.cpp's default (2048).\n" +
				"Mostly a ceiling for ubatch_size; lowering both shrinks compute " +
				"buffers when VRAM is right at the edge, at the cost of slower " +
				"prompt processing."
		},
		Apply: func(c *Config, val string) error {
			return parseAutoInt("batch_size", val, 1, 1<<20, func(n int) { c.BatchSize = n })
		},
	},
	{
		Key:     "ubatch_size",
		Usage:   "/set ubatch_size auto|N",
		Summary: "Physical batch per engine step (llama.cpp -ub).",
		Restart: true,
		Value: func(c Config) string {
			if c.UBatchSize == 0 {
				return "auto (512)"
			}
			return fmt.Sprintf("%d", c.UBatchSize)
		},
		Detail: func(c Config) string {
			return "The real per-step batch, and the knob that sizes the compute " +
				"buffers in VRAM. auto keeps llama.cpp's default (512).\n" +
				"128 or 256 can recover a few hundred MiB when a model almost fits; " +
				"prefill slows roughly in proportion. Raising it above batch_size " +
				"does nothing."
		},
		Apply: func(c *Config, val string) error {
			return parseAutoInt("ubatch_size", val, 1, 1<<20, func(n int) { c.UBatchSize = n })
		},
	},
	{
		Key:     "parallel",
		Usage:   "/set parallel auto|N",
		Summary: "Server slots: concurrent requests (llama.cpp --parallel).",
		Restart: true,
		Value: func(c Config) string {
			if c.Parallel == 0 {
				return fmt.Sprintf("auto (%d)", kvCacheSlots)
			}
			return fmt.Sprintf("%d", c.Parallel)
		},
		Detail: func(c Config) string {
			return fmt.Sprintf("How many requests the server handles at once. auto is "+
				"%d: one for the conversation, one so a /compact or /summarize "+
				"doesn't evict the conversation's KV.\n"+
				"The KV budget is fixed — more slots divide it, so each slot sees "+
				"proportionally less context. Mainly for --serve machines feeding "+
				"several clients. A `--serve --slots N` flag on the command line "+
				"still wins over this.", kvCacheSlots)
		},
		Apply: func(c *Config, val string) error {
			return parseAutoInt("parallel", val, 1, 64, func(n int) { c.Parallel = n })
		},
	},
	{
		Key:     "cache_reuse",
		Usage:   "/set cache_reuse auto|off|N",
		Summary: "Min tokens for KV-chunk salvage (llama.cpp --cache-reuse).",
		Restart: true,
		Value: func(c Config) string {
			if c.CacheReuse == nil {
				return fmt.Sprintf("auto (%d)", defaultCacheReuse)
			}
			if *c.CacheReuse == 0 {
				return "off"
			}
			return fmt.Sprintf("%d", *c.CacheReuse)
		},
		Detail: func(c Config) string {
			return fmt.Sprintf("Salvages still-matching KV chunks past the first "+
				"divergence instead of reprocessing everything after it — mainly "+
				"softens the full re-prefill after /compact rewrites history. auto "+
				"is %d tokens; off disables salvage entirely.\n"+
				"Runtime no-op for sliding-window models (Gemma).", defaultCacheReuse)
		},
		Apply: func(c *Config, val string) error {
			v := strings.ToLower(strings.TrimSpace(val))
			switch v {
			case "auto", "default":
				c.CacheReuse = nil
				return nil
			case "off", "none":
				zero := 0
				c.CacheReuse = &zero
				return nil
			}
			n, err := strconv.Atoi(v)
			if err != nil || n < 0 {
				return fmt.Errorf("invalid cache_reuse=%q (expected auto, off, or a token count)", val)
			}
			c.CacheReuse = &n
			return nil
		},
	},
	{
		Key:     "mmap",
		Usage:   "/set mmap on|off",
		Summary: "Memory-map the weights (off copies them into RAM).",
		Restart: true,
		Value:   func(c Config) string { return onOffValue(c.Mmap, "on") },
		Detail: func(c Config) string {
			return "on (the default) maps the GGUF from disk, so startup is instant " +
				"and the OS pages weights in as needed. off reads the whole file " +
				"into RAM up front — slower start, but no first-token page-fault " +
				"stalls, and on some setups faster CPU inference.\n" +
				"off needs RAM for the entire model file; check /list's fit column " +
				"first."
		},
		Apply: func(c *Config, val string) error {
			return parseOnOff("mmap", val, func(v string) {
				if v == "on" {
					v = ""
				}
				c.Mmap = v
			})
		},
	},
	{
		Key:     "mlock",
		Usage:   "/set mlock on|off",
		Summary: "Pin the weights in RAM so the OS can't swap them out.",
		Restart: true,
		Value:   func(c Config) string { return onOffValue(c.Mlock, "off") },
		Detail: func(c Config) string {
			return "on locks the model's pages in memory (llama.cpp --mlock), which " +
				"prevents the mid-conversation stall of weights being swapped out " +
				"under memory pressure — at the price of that RAM being genuinely " +
				"gone for everything else.\n" +
				"Needs the model to actually fit in RAM, and on some systems a " +
				"raised memlock limit; a launch failure right after enabling this " +
				"is that limit."
		},
		Apply: func(c *Config, val string) error {
			return parseOnOff("mlock", val, func(v string) {
				if v == "off" {
					v = ""
				}
				c.Mlock = v
			})
		},
	},
	{
		Key:     "seed",
		Usage:   "/set seed auto|N",
		Summary: "Sampling seed, for reproducible generations.",
		Restart: true,
		Value: func(c Config) string {
			if c.Seed == nil {
				return "auto (random)"
			}
			return fmt.Sprintf("%d", *c.Seed)
		},
		Detail: func(c Config) string {
			return "auto lets the server pick a fresh seed per generation. A fixed " +
				"number makes the same prompt at the same settings reproduce the " +
				"same output — useful for comparing models or debugging a prompt, " +
				"pointless for normal chat.\n" +
				"Identical replies also need identical context, so any history " +
				"difference still changes the output."
		},
		Apply: func(c *Config, val string) error {
			v := strings.ToLower(strings.TrimSpace(val))
			if v == "auto" || v == "default" || v == "random" {
				c.Seed = nil
				return nil
			}
			n, err := strconv.Atoi(v)
			if err != nil || n < 0 {
				return fmt.Errorf("invalid seed=%q (expected auto or a non-negative integer)", val)
			}
			c.Seed = &n
			return nil
		},
	},
	{
		Key:     "temperature",
		Usage:   "/set temperature auto|X",
		Summary: "Sampling temperature sent with every request.",
		Value: func(c Config) string {
			if c.Temperature == nil {
				return fmt.Sprintf("auto (%.1f)", defaultTemperature)
			}
			return fmt.Sprintf("%.2f", *c.Temperature)
		},
		Detail: func(c Config) string {
			return fmt.Sprintf("Rides on each request rather than the server launch, so "+
				"it changes nothing about memory and needs no restart — and it still "+
				"applies when inference is remote.\n"+
				"auto is %.1f: low, because tool calls and code want determinism. "+
				"0.7–0.9 is the usual range for creative prose; 0 is greedy "+
				"decoding. Range 0 to 2.", defaultTemperature)
		},
		Apply: func(c *Config, val string) error {
			v := strings.ToLower(strings.TrimSpace(val))
			if v == "auto" || v == "default" {
				c.Temperature = nil
				return nil
			}
			f, err := strconv.ParseFloat(v, 64)
			if err != nil || f < 0 || f > 2 {
				return fmt.Errorf("invalid temperature=%q (expected auto or 0..2)", val)
			}
			c.Temperature = &f
			return nil
		},
	},
	{
		Key:     "override_tensor",
		Usage:   "/set override_tensor PATTERN=BACKEND | off",
		Summary: "Per-tensor placement regex (llama.cpp -ot). Power tool.",
		Restart: true,
		Value: func(c Config) string {
			if c.OverrideTensor == "" {
				return "off"
			}
			return c.OverrideTensor
		},
		Detail: func(c Config) string {
			return "Passes a raw tensor-placement rule to the engine, e.g. " +
				"`exps=CPU` to keep every expert tensor in system RAM while " +
				"attention stays on the GPU — finer-grained than the automatic " +
				"--n-cpu-moe. The pattern is a regex matched against tensor names; " +
				"the engine, not atlas.llm, interprets it.\n" +
				"The offload estimator does not model this, so /config's VRAM " +
				"arithmetic can be wrong while it is set. A bad pattern fails at " +
				"launch, loudly. `off` clears it."
		},
		Apply: func(c *Config, val string) error {
			v := strings.TrimSpace(val)
			if strings.EqualFold(v, "off") || strings.EqualFold(v, "none") {
				c.OverrideTensor = ""
				return nil
			}
			if !strings.Contains(v, "=") {
				return fmt.Errorf("invalid override_tensor=%q (expected PATTERN=BACKEND, e.g. exps=CPU)", val)
			}
			c.OverrideTensor = v
			return nil
		},
	},
	{
		Key:     "context_shift",
		Usage:   "/set context_shift on|off",
		Summary: "Slide the window when context fills instead of stopping.",
		Restart: true,
		Value:   func(c Config) string { return onOffValue(c.ContextShift, "off") },
		Detail: func(c Config) string {
			return "on lets the server drop the oldest tokens and keep generating " +
				"when the context fills (llama.cpp --context-shift). The model " +
				"silently forgets the start of the conversation — /compact does " +
				"the same job but says what it kept.\n" +
				"off (the default) surfaces the limit instead. Needs an engine " +
				"recent enough to know the flag; incompatible with some models."
		},
		Apply: func(c *Config, val string) error {
			return parseOnOff("context_shift", val, func(v string) {
				if v == "off" {
					v = ""
				}
				c.ContextShift = v
			})
		},
	},
}

func init() {
	settingsRegistry = append(settingsRegistry, tuningSettings...)
}
