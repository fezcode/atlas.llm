package engine

import (
	"fmt"
	"runtime"
	"strconv"
	"strings"

	"atlas.llm/internal/config"
)

// defaultTemperature is what every request sent before the setting existed.
const DefaultTemperature = 0.2

// defaultCacheReuse is the --cache-reuse the optimized launch always passed.
const DefaultCacheReuse = 256

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
func CacheTypeElemBytes(t string) float64 {
	if b, ok := cacheTypeBytes[strings.ToLower(t)]; ok {
		return b
	}
	return 2
}

func ValidCacheTypeNames() []string {
	// Stable order for error messages, cheapest first.
	return []string{"q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "bf16", "f32"}
}

func IsValidCacheType(t string) bool {
	_, ok := cacheTypeBytes[strings.ToLower(t)]
	return ok
}

// cacheTypeIsQuantized reports whether a type needs flash attention when
// used for the V cache — llama-server refuses a quantized V without it.
func CacheTypeIsQuantized(t string) bool {
	switch strings.ToLower(t) {
	case "", "f16", "f32", "bf16":
		return false
	}
	return true
}

func kvOffloadEnabled(cfg config.Config) bool { return cfg.KVOffload != "off" }

// vramKVCharge is what the VRAM fit math should charge for the KV cache.
// With kv_offload off the cache lives in system RAM and VRAM pays nothing,
// which is exactly why the setting exists: more weight layers fit.
func VramKVCharge(cfg config.Config, kv int64) int64 {
	if !kvOffloadEnabled(cfg) {
		return 0
	}
	return kv
}

// resolveThreads is the -t value: explicit, or NumCPU-1 capped at 6 — more
// threads than that fight the GPU feeding loop for cache without prompt
// throughput to show for it.
func ResolveThreads(cfg config.Config) int {
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
func resolveFlashAttn(cfg config.Config) string {
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

func resolveCacheReuse(cfg config.Config) int {
	if cfg.CacheReuse != nil {
		return *cfg.CacheReuse
	}
	return DefaultCacheReuse
}

func ResolveTemperature(cfg config.Config) float64 {
	if cfg.Temperature != nil {
		return *cfg.Temperature
	}
	return DefaultTemperature
}

// requestTemperature reads the setting per request rather than binding it at
// server launch — that is what lets /set temperature apply without a restart,
// locally and against a remote alike. One config read per message, not per
// frame.
func requestTemperature() float64 {
	cfg, _ := config.LoadConfig()
	return ResolveTemperature(cfg)
}

// kvElemBytesFor is the per-element KV cost the launch will actually pay,
// averaged over K and V. On a zero config this is exactly the q8_0 constant
// the estimator has always used.
func kvElemBytesFor(cfg config.Config) float64 {
	k := CacheTypeElemBytes(resolveCacheType(cfg.CacheTypeK))
	v := CacheTypeElemBytes(resolveCacheType(cfg.CacheTypeV))
	return (k + v) / 2
}

// tuningFingerprint is the restart identity of the tuning settings: the raw
// configured values, never resolved autos, so estimate drift can't evict a
// running server but a user change always does. temperature is deliberately
// absent — it rides on each request and needs no relaunch.
func TuningFingerprint(cfg config.Config) string {
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
func tuningArgs(cfg config.Config) []string {
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
