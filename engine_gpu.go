package main

import (
	"fmt"
	"log"
	"math"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// gpuVendorNVIDIA is the only vendor we can identify cheaply and reliably:
// nvidia-smi ships with every NVIDIA driver and reports the compute
// capability we need to pick a CUDA archive. AMD and Intel have no
// equivalent single-binary probe, so those stay opt-in via
// `/set engine_variant`.
const gpuVendorNVIDIA = "nvidia"

// gpuInfo describes the GPU an engine build would have to target.
type gpuInfo struct {
	Vendor string
	Name   string
	// ComputeCap is the CUDA compute capability scaled by ten, matching how
	// llama.cpp and CUDA name architectures: 8.9 (Ada) -> 89, 12.0
	// (Blackwell) -> 120. Scaling keeps the comparisons in cudaArchives
	// integer-only.
	ComputeCap int
	VRAMMiB    int
}

// nvidiaSmiOutput is the process-spawning seam. Tests replace it; nothing
// else should call runNvidiaSmi directly.
var nvidiaSmiOutput = runNvidiaSmi

func runNvidiaSmi() (string, error) {
	cmd := exec.Command("nvidia-smi",
		"--query-gpu=name,compute_cap,memory.total",
		"--format=csv,noheader,nounits")
	// Same creation flags as the engine subprocess: without them Windows
	// flashes a console window, and this runs during TUI startup.
	applyEngineSysProcAttr(cmd)
	out, err := cmd.Output()
	return string(out), err
}

var (
	gpuOnce   sync.Once
	gpuCached gpuInfo
	gpuFound  bool
)

// detectGPU probes for a usable GPU, caching the result for the process.
// The cache matters: resolveEngineVariant feeds the TUI header and settings
// rendering, which repaint on every keystroke — spawning nvidia-smi per
// frame would cost more than the feature saves.
func detectGPU() (gpuInfo, bool) {
	gpuOnce.Do(func() {
		out, err := nvidiaSmiOutput()
		if err != nil {
			// Overwhelmingly this just means "no NVIDIA driver installed",
			// which is not an error worth showing the user.
			log.Printf("gpu: nvidia-smi unavailable (%v) — assuming no CUDA GPU", err)
			return
		}
		gpuCached, gpuFound = parseNvidiaSmi(out)
		if gpuFound {
			log.Printf("gpu: detected %s (compute %d, %dMiB)",
				gpuCached.Name, gpuCached.ComputeCap, gpuCached.VRAMMiB)
		} else {
			log.Printf("gpu: nvidia-smi returned no parseable GPU: %q", strings.TrimSpace(out))
		}
	})
	return gpuCached, gpuFound
}

// resetGPUDetection clears the cache so a test can install a different
// nvidiaSmiOutput stub.
func resetGPUDetection() {
	gpuOnce = sync.Once{}
	gpuCached = gpuInfo{}
	gpuFound = false
}

// parseNvidiaSmi reads `--format=csv,noheader,nounits` rows of
// "name, compute_cap, memory.total". The first row that parses wins:
// multi-GPU machines run llama.cpp on device 0 unless told otherwise, so
// that's the card whose capability decides the archive.
func parseNvidiaSmi(out string) (gpuInfo, bool) {
	for _, line := range strings.Split(out, "\n") {
		fields := strings.Split(strings.TrimSpace(line), ",")
		if len(fields) < 2 {
			continue
		}
		cap, ok := parseComputeCap(fields[1])
		if !ok {
			continue
		}
		info := gpuInfo{
			Vendor:     gpuVendorNVIDIA,
			Name:       strings.TrimSpace(fields[0]),
			ComputeCap: cap,
		}
		if len(fields) >= 3 {
			if mib, err := strconv.Atoi(strings.TrimSpace(fields[2])); err == nil {
				info.VRAMMiB = mib
			}
		}
		return info, true
	}
	return gpuInfo{}, false
}

// parseComputeCap turns nvidia-smi's "12.0" into 120. A missing or
// malformed value must not resolve to some plausible-looking capability —
// picking the wrong CUDA archive fails at model load, well after the user
// has waited through a ~510MB download.
func parseComputeCap(s string) (int, bool) {
	major, minor, found := strings.Cut(strings.TrimSpace(s), ".")
	if !found {
		return 0, false
	}
	maj, err := strconv.Atoi(strings.TrimSpace(major))
	if err != nil || maj <= 0 {
		return 0, false
	}
	min, err := strconv.Atoi(strings.TrimSpace(minor))
	if err != nil || min < 0 || min > 9 {
		return 0, false
	}
	return maj*10 + min, true
}

// nvidiaSmiMemory is the memory-query seam, kept separate from the
// capability probe: capability never changes, free VRAM changes constantly.
var nvidiaSmiMemory = runNvidiaSmiMemory

func runNvidiaSmiMemory() (string, error) {
	cmd := exec.Command("nvidia-smi",
		"--query-gpu=memory.used,memory.total",
		"--format=csv,noheader,nounits")
	applyEngineSysProcAttr(cmd)
	out, err := cmd.Output()
	return string(out), err
}

// VRAM readings are cached and refreshed off the caller's path for the same
// reason process RSS is: nvidia-smi is a process spawn, and anything that
// renders must never pay for one.
const vramCacheTTL = 3 * time.Second

var (
	vramMu       sync.Mutex
	vramUsed     int
	vramTotal    int
	vramOK       bool
	vramAt       time.Time
	vramInflight bool
)

// gpuMemory reports VRAM use in MiB. Returns the last known reading
// immediately and refreshes in the background when stale.
func gpuMemory() (used, total int, ok bool) {
	vramMu.Lock()
	defer vramMu.Unlock()
	if time.Since(vramAt) >= vramCacheTTL && !vramInflight {
		vramInflight = true
		go func() {
			u, t, o := readGPUMemory()
			vramMu.Lock()
			vramUsed, vramTotal, vramOK = u, t, o
			vramAt, vramInflight = time.Now(), false
			vramMu.Unlock()
		}()
	}
	return vramUsed, vramTotal, vramOK
}

// gpuMemoryNow reads VRAM synchronously. Used before launching a server,
// where the answer decides how many layers to offload and a stale or absent
// reading would decide it wrongly.
func gpuMemoryNow() (used, total int, ok bool) {
	u, t, o := readGPUMemory()
	vramMu.Lock()
	vramUsed, vramTotal, vramOK, vramAt = u, t, o, time.Now()
	vramMu.Unlock()
	return u, t, o
}

func readGPUMemory() (used, total int, ok bool) {
	out, err := nvidiaSmiMemory()
	if err != nil {
		return 0, 0, false
	}
	return parseGPUMemory(out)
}

// parseGPUMemory reads "used, total" in MiB from the first GPU row.
func parseGPUMemory(out string) (used, total int, ok bool) {
	for _, line := range strings.Split(out, "\n") {
		fields := strings.Split(strings.TrimSpace(line), ",")
		if len(fields) < 2 {
			continue
		}
		u, err1 := strconv.Atoi(strings.TrimSpace(fields[0]))
		t, err2 := strconv.Atoi(strings.TrimSpace(fields[1]))
		if err1 != nil || err2 != nil || t <= 0 || u < 0 {
			continue
		}
		return u, t, true
	}
	return 0, 0, false
}

// vramHeadroomMiB is held back from the offload estimate for compute buffers,
// the CUDA context, fragmentation, and whatever else appears between the
// estimate and the load. Erring high costs a few layers; erring low costs a
// failed load after the user has waited for it.
const vramHeadroomMiB = 1024

// fitGPULayers estimates how many of a model's layers fit in free VRAM.
//
// ok=false means "don't clamp": something needed was unknown, and a guessed
// clamp is a permanent silent slowdown, which is worse than attempting a full
// offload and getting a loud failure.
func fitGPULayers(m Model, ctx int, cfg Config) (layers, totalLayers int, ok bool) {
	weights, kv, estOK := modelMemoryEstimate(m, ctx, cfg)
	if !estOK || weights <= 0 {
		return 0, 0, false
	}
	p, err := modelPath(m)
	if err != nil {
		return 0, 0, false
	}
	meta, err := readGGUFMetaCached(p)
	if err != nil || meta.BlockCount <= 0 {
		return 0, 0, false
	}
	used, total, memOK := gpuMemoryNow()
	if !memOK || total <= 0 {
		return 0, 0, false
	}

	// With kv_offload off the cache lives in system RAM and costs VRAM
	// nothing — which is why more layers fit.
	n, ok := layersThatFit(weights, vramKVCharge(cfg, kv), meta.BlockCount, used, total)
	return n, meta.BlockCount, ok
}

// layersThatFit is the offload arithmetic, split out so it can be checked
// without a GPU or a model file on disk.
//
// Every layer costs the same share of the weights, the KV cache is charged in
// full because it lives in VRAM alongside them, and vramHeadroomMiB is held
// back for compute buffers and the CUDA context.
func layersThatFit(weights, kv int64, blockCount, usedMiB, totalMiB int) (int, bool) {
	if blockCount <= 0 || weights <= 0 || totalMiB <= 0 {
		return 0, false
	}
	const mib = 1024 * 1024
	freeMiB := totalMiB - usedMiB - vramHeadroomMiB - int(kv/mib)
	if freeMiB <= 0 {
		return 0, true
	}
	perLayerMiB := float64(weights) / float64(blockCount) / mib
	if perLayerMiB <= 0 {
		return 0, false
	}
	n := int(float64(freeMiB) / perLayerMiB)
	if n < 0 {
		n = 0
	}
	if n > blockCount {
		n = blockCount
	}
	return n, true
}

// offloadPlan is how a launch divides a model between GPU and system RAM.
//
// Two ways to shed VRAM, and for a mixture-of-experts model they are not
// equally good. Dropping whole layers to the CPU (NGL) moves their
// attention as well, and attention is what every token pays for. Dropping
// only the expert tensors (CPUMoE) keeps all the attention on the GPU and
// moves the weights that a given token mostly does not route through —
// which is why a 30B-A3B can run largely on a 16GB card.
type offloadPlan struct {
	NGL    int // -ngl
	CPUMoE int // --n-cpu-moe; 0 means the flag is not passed
	// Setting is the gpu_layers setting the plan was derived from, carried
	// so a running server can be identified by what the user asked for
	// rather than by what the estimate happened to conclude. See
	// serverMatches.
	Setting int
}

// cpuMoELayers returns how many layers must keep their experts in system
// RAM for everything else to fit in VRAM.
//
// fits=false means even every layer's experts on the CPU is not enough, and
// the caller should fall back to plain layer offload.
func cpuMoELayers(weights, expertPerLayer, kv int64, blockCount, usedMiB, totalMiB int) (n int, fits bool) {
	if blockCount <= 0 || weights <= 0 || expertPerLayer <= 0 || totalMiB <= 0 {
		return 0, false
	}
	const mib = 1024 * 1024
	budgetMiB := totalMiB - usedMiB - vramHeadroomMiB - int(kv/mib)
	if budgetMiB <= 0 {
		return blockCount, false
	}
	deficitMiB := int(weights/mib) - budgetMiB
	if deficitMiB <= 0 {
		return 0, true // fits whole; no experts need to move
	}
	perLayerMiB := float64(expertPerLayer) / mib
	if perLayerMiB <= 0 {
		return 0, false
	}
	// Round up: a layer's experts move or they don't.
	n = int(math.Ceil(float64(deficitMiB) / perLayerMiB))
	if n > blockCount {
		return blockCount, false
	}
	return n, true
}

// planOffload decides -ngl and --n-cpu-moe for a model at a context size.
//
// Only consulted when gpu_layers is on auto: an explicit setting is the
// user's instruction and is passed through untouched.
func planOffload(m Model, ctx int, cfg Config) offloadPlan {
	if runtime.GOOS != "darwin" && !engineVariantIsGPU(effectiveEngineVariant(cfg.EngineVariant)) {
		return offloadPlan{NGL: 0}
	}
	weights, rawKV, estOK := modelMemoryEstimate(m, ctx, cfg)
	p, perr := modelPath(m)
	used, total, memOK := gpuMemoryNow()
	if !estOK || perr != nil || !memOK || total <= 0 {
		// An unknown answer means offload everything and let a real failure
		// be loud, rather than quietly halving performance forever.
		return offloadPlan{NGL: maxGPULayers}
	}
	meta, err := readGGUFMetaCached(p)
	if err != nil || meta.BlockCount <= 0 {
		return offloadPlan{NGL: maxGPULayers}
	}
	// Zero when kv_offload is off: the cache is in system RAM, so the VRAM
	// arithmetic below must not budget for it.
	kv := vramKVCharge(cfg, rawKV)

	if meta.isMoE() {
		if per := meta.expertBytesPerLayer(weights); per > 0 {
			if n, fits := cpuMoELayers(weights, per, kv, meta.BlockCount, used, total); fits {
				if n > 0 {
					log.Printf("gpu: %s is MoE and exceeds free VRAM — keeping experts of %d of %d layers in system RAM",
						m.Name, n, meta.BlockCount)
				}
				return offloadPlan{NGL: maxGPULayers, CPUMoE: n}
			}
			log.Printf("gpu: %s does not fit even with every expert in system RAM — falling back to layer offload", m.Name)
		}
	}

	n, ok := layersThatFit(weights, kv, meta.BlockCount, used, total)
	if !ok || n >= meta.BlockCount {
		return offloadPlan{NGL: maxGPULayers}
	}
	log.Printf("gpu: %s does not fit in free VRAM — offloading %d of %d layers",
		m.Name, n, meta.BlockCount)
	return offloadPlan{NGL: n}
}

// resolveOffload is the launch-time entry point: an explicit gpu_layers wins
// outright, and only auto gets a plan.
func resolveOffload(cfg Config) offloadPlan {
	setting := gpuLayersSetting(cfg)
	if cfg.GPULayers != nil {
		return offloadPlan{NGL: resolveGPULayers(cfg), Setting: setting}
	}
	m, err := currentModel()
	if err != nil {
		return offloadPlan{NGL: resolveGPULayers(cfg), Setting: setting}
	}
	plan := planOffload(m, resolveCtxSize(cfg), cfg)
	plan.Setting = setting
	return plan
}

// renderGPUSection is the GPU block in /config. Empty when there is no GPU to
// describe, so a CPU-only machine isn't told about hardware it doesn't have.
//
// This answers "is the model actually on the GPU", which previously could
// only be established by running nvidia-smi by hand — llama.cpp's own
// offload report isn't printed at default verbosity by current builds, so
// VRAM in use is the honest evidence.
func renderGPUSection(cfg Config) string {
	info, found := detectGPU()
	if !found {
		return ""
	}
	var b strings.Builder
	b.WriteString("\nGPU\n")
	fmt.Fprintf(&b, "  %-14s  %s (compute %.1f)\n", "device", info.Name, float64(info.ComputeCap)/10)

	// Synchronous here on purpose: /config is typed, not rendered per
	// frame, and a first view showing nothing would be worse than ~50ms.
	if used, total, ok := gpuMemoryNow(); ok && total > 0 {
		fmt.Fprintf(&b, "  %-14s  %d / %d MiB in use (%d%%)\n", "vram",
			used, total, used*100/total)
	} else {
		fmt.Fprintf(&b, "  %-14s  %d MiB\n", "vram", info.VRAMMiB)
	}

	if !engineVariantIsGPU(installedEngineVariant()) && !isDarwin() {
		fmt.Fprintf(&b, "  %-14s  %s\n", "offload",
			"none — installed engine is a CPU-only build")
		return b.String()
	}

	plan := resolveOffload(cfg)
	n := plan.NGL
	switch {
	case n == 0:
		fmt.Fprintf(&b, "  %-14s  %s\n", "offload", "disabled (gpu_layers = 0)")
	case plan.CPUMoE > 0:
		// Worth its own line: "all layers" would be true and misleading,
		// since a chunk of the weights is in system RAM regardless.
		fmt.Fprintf(&b, "  %-14s  %s\n", "offload", "all layers")
		fmt.Fprintf(&b, "  %-14s  experts of %d layers in system RAM (auto — model exceeds free VRAM)\n",
			"moe", plan.CPUMoE)
	case n >= maxGPULayers:
		fmt.Fprintf(&b, "  %-14s  %s\n", "offload", "all layers")
	default:
		// A partial offload is either the user's choice or auto's estimate,
		// and which one matters — an estimate that silently halves speed
		// should be visible, not buried.
		reason := "set explicitly"
		if cfg.GPULayers == nil {
			reason = "auto — model exceeds free VRAM"
		}
		if m, err := currentModel(); err == nil {
			if _, total, ok := fitGPULayers(m, resolveCtxSize(cfg), cfg); ok && total > 0 {
				fmt.Fprintf(&b, "  %-14s  %d of %d layers (%s)\n", "offload", n, total, reason)
				return b.String()
			}
		}
		fmt.Fprintf(&b, "  %-14s  %d layers (%s)\n", "offload", n, reason)
	}
	return b.String()
}
