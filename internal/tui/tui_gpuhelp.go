package tui

import (
	"fmt"
	"runtime"
	"strings"

	"atlas.llm/internal/config"
	"atlas.llm/internal/engine"
)

// gpuHelpRows builds the "Performance" block of the in-app /help, tailored
// to what this platform can actually do. There's no point telling a Mac user
// to install a Vulkan build, or a Windows user that Metal is built in.
func gpuHelpRows() [][2]string {
	rows := [][2]string{
		{"/set gpu_layers", "auto (default) · 0 for CPU-only · N to offload N layers"},
	}
	cfg, _ := config.LoadConfig()
	installed := engine.InstalledEngineVariant()
	// installedEngineVariant answers "cpu" for a missing marker, which is
	// right for pre-variant installs but reads as a lie before the first
	// /download. Separate the two for display only.
	installedLabel := installed
	if !engine.IsEngineDownloaded() {
		installedLabel = "not installed"
	}

	switch {
	case runtime.GOOS == "darwin":
		rows = append(rows, [2]string{"", "Metal is built into the macOS engine — GPU is on by default"})
	case engine.EngineVariantIsGPU(installed):
		rows = append(rows, [2]string{"", fmt.Sprintf("engine: %s build installed — GPU offload active", installed)})
	default:
		opts := engine.EngineVariantNames()
		switch {
		case len(opts) <= 2:
			rows = append(rows, [2]string{"", "no GPU llama.cpp build published for this platform"})
		default:
			// Naming the detected card turns a generic menu into a specific
			// instruction, which is the difference between the setting being
			// discoverable and it sitting unused.
			if hint := engine.EngineUpgradeHint(); hint != "" {
				for i, line := range strings.Split(hint, "\n") {
					label := ""
					if i == 0 {
						label = "GPU detected"
					}
					rows = append(rows, [2]string{label, line})
				}
				break
			}
			rows = append(rows,
				[2]string{"/set engine_variant", "GPU builds here: " + strings.Join(opts[2:], ", ")},
				[2]string{"", "then /download engine to install it (CPU-only until you do)"},
			)
		}
	}
	rows = append(rows, [2]string{"/set", fmt.Sprintf("current: gpu_layers=%s, engine=%s",
		engine.GpuLayersDisplay(cfg), installedLabel)})
	return rows
}
