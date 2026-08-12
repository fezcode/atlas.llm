package main

import (
	"log"
	"os/exec"
	"strconv"
	"strings"
	"sync"
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
