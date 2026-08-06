package main

import (
	"fmt"
	"runtime"
	"strings"
)

// isDarwin is spelled out here because the macOS engine archive always
// carries Metal, which changes the advice for two settings.
func isDarwin() bool { return runtime.GOOS == "darwin" }

// setting is one persisted option. Keeping the key, how to render its
// current value, and its guidance in one place means `/set`, `/config`, and
// `/help set` can't drift apart — they all read from here.
type setting struct {
	Key     string
	Usage   string
	Summary string
	// Value renders the current setting, including anything that makes it
	// concrete (the effective number behind "auto", the model's ceiling).
	Value func(Config) string
	// Detail is the guidance shown by `/set <key>` with no value: what it
	// does, what the limits are, and what it costs. The long-form prose
	// lives in help.go and is reachable via `/help set <key>`.
	Detail func(Config) string
}

var settingsRegistry = []setting{
	{
		Key:     "max_tokens",
		Usage:   "/set max_tokens N",
		Summary: "Cap on reply length, in tokens.",
		Value:   func(c Config) string { return fmt.Sprintf("%d", c.MaxTokens) },
		Detail: func(c Config) string {
			return fmt.Sprintf(
				"How long a single reply may be. Default %d, and the ceiling is %d — "+
					"three quarters of the %d-token context, since the rest is needed "+
					"for the prompt and history.\n"+
					"Raise it if replies are cut off mid-sentence, or if a reasoning "+
					"model returns nothing because thinking used the whole budget.",
				defaultMaxTokens, maxTokensCeiling(c), resolveCtxSize(c))
		},
	},
	{
		Key:     "ctx_size",
		Usage:   "/set ctx_size auto|N",
		Summary: "Context window: how much the model can see at once.",
		Value:   func(c Config) string { return ctxSizeDisplay(c) },
		Detail: func(c Config) string {
			var b strings.Builder
			fmt.Fprintf(&b, "How much conversation, tool output, and file content the model "+
				"sees at once. Default %d.\n", defaultCtxSize)
			if trained := currentModelTrainedContext(); trained > 0 {
				fmt.Fprintf(&b, "This model was trained for %d tokens; asking for more is "+
					"refused, because exceeding it degrades quality rather than "+
					"extending memory.\n", trained)
			} else {
				b.WriteString("The ceiling is whatever the active model was trained for, " +
					"read from its GGUF file.\n")
			}
			fmt.Fprintf(&b, "Range %d to %d. Costs memory: the KV cache grows roughly "+
				"linearly, so doubling the window roughly doubles what the server "+
				"holds beyond the weights.\n", minConfigurableCtx, maxConfigurableCtx)
			b.WriteString("If the window is filling up rather than being too small, " +
				"/compact is the cheaper fix.")
			return b.String()
		},
	},
	{
		Key:     "max_tool_rounds",
		Usage:   "/set max_tool_rounds N|off",
		Summary: "Tool-call rounds allowed for a single message.",
		Value:   func(c Config) string { return maxToolRoundsDisplay(c) },
		Detail: func(c Config) string {
			return fmt.Sprintf(
				"How many times the model may call tools while answering one message. "+
					"Default %d. `off` removes the cap.\n"+
					"Raise it for work that genuinely needs many steps — reading a dozen "+
					"files, or a long MCP chain. Hitting the cap often means the model is "+
					"stuck retrying something instead, so check the trace before raising it.\n"+
					"Turning it off is safe enough in practice: identical repeated calls "+
					"still stop the turn after %d attempts, and esc stops it at any time.",
				defaultMaxToolRounds, maxIdenticalCalls+1)
		},
	},
	{
		Key:     "gpu_layers",
		Usage:   "/set gpu_layers auto|0|N",
		Summary: "Model layers offloaded to the GPU (llama.cpp -ngl).",
		Value:   func(c Config) string { return gpuLayersDisplay(c) },
		Detail: func(c Config) string {
			var b strings.Builder
			b.WriteString("auto offloads everything when the installed engine has a GPU " +
				"backend, and stays on CPU otherwise. 0 forces CPU-only. A number " +
				"offloads that many layers, which is how you fit a model that would " +
				"otherwise exceed VRAM.\n")
			if isDarwin() {
				b.WriteString("On macOS the engine always includes Metal, so auto already " +
					"uses the GPU — nothing else to configure.\n")
			} else if !engineVariantIsGPU(installedEngineVariant()) {
				b.WriteString("The installed engine is a CPU-only build, so this currently " +
					"has no effect. Set engine_variant to a GPU build and run " +
					"/download engine first.\n")
			}
			b.WriteString("Changing this restarts the model server.")
			return b.String()
		},
	},
	{
		Key:     "engine_variant",
		Usage:   "/set engine_variant " + strings.Join(engineVariantNames(), "|"),
		Summary: "Which llama.cpp build to download (decides GPU support).",
		Value:   func(c Config) string { return engineVariantDisplay(c) },
		Detail: func(c Config) string {
			var b strings.Builder
			fmt.Fprintf(&b, "Decides whether the GPU can be used at all; gpu_layers does "+
				"nothing until a GPU-capable build is installed.\n"+
				"Available on this platform: %s. Currently installed: %s.\n",
				strings.Join(engineVariantNames(), ", "), installedEngineVariant())
			b.WriteString("Changing it takes two steps — the setting picks an archive, " +
				"/download engine then fetches it and replaces the installed engine.\n")
			if isDarwin() {
				b.WriteString("macOS needs none of this: its archive already carries Metal.")
			} else {
				b.WriteString("vulkan is the smallest and most portable GPU option; cuda is " +
					"usually fastest on NVIDIA but much larger. A GPU build needs a " +
					"working driver, so atlas.llm won't select one for you.")
			}
			return b.String()
		},
	},
}

func findSetting(key string) (setting, bool) {
	key = strings.ToLower(strings.TrimSpace(key))
	for _, s := range settingsRegistry {
		if s.Key == key {
			return s, true
		}
	}
	return setting{}, false
}

func settingKeys() []string {
	out := make([]string, 0, len(settingsRegistry))
	for _, s := range settingsRegistry {
		out = append(out, s.Key)
	}
	return out
}

// renderSettingsList is the body of `/set` with no arguments: every key,
// its current value, and a one-line summary.
func renderSettingsList(cfg Config) string {
	width := 0
	for _, s := range settingsRegistry {
		if len(s.Key) > width {
			width = len(s.Key)
		}
	}
	var b strings.Builder
	b.WriteString("Settings:\n")
	for _, s := range settingsRegistry {
		fmt.Fprintf(&b, "  %-*s = %s\n", width, s.Key, s.Value(cfg))
		fmt.Fprintf(&b, "  %-*s   %s\n", width, "", s.Summary)
	}
	b.WriteString("\n`/set <key>` explains one setting · `/config` shows everything")
	return b.String()
}

// renderSettingDetail is `/set <key>` with no value: the current value plus
// the guidance needed to choose a new one.
func renderSettingDetail(s setting, cfg Config) string {
	return fmt.Sprintf("%s = %s\n\n%s\n\n  usage: %s\n  full detail: /help set %s",
		s.Key, s.Value(cfg), s.Detail(cfg), s.Usage, s.Key)
}

// configState is the session-scoped state `/config` reports alongside the
// persisted settings, since "why is it behaving like this" often comes down
// to one of these rather than to config.json.
type configState struct {
	toolsEnabled bool
	yesman       bool
	mcpServers   int
	mcpConnected int
	mcpTools     int
}

// renderConfig is `/config`: everything about the current setup in one
// place — persisted settings, session-only state, and where things live on
// disk.
func renderConfig(cfg Config, st configState) string {
	var b strings.Builder

	b.WriteString("SETTINGS  (persisted, change with /set <key> <value>)\n")
	width := 0
	for _, s := range settingsRegistry {
		if len(s.Key) > width {
			width = len(s.Key)
		}
	}
	for _, s := range settingsRegistry {
		fmt.Fprintf(&b, "  %-*s  %s\n", width, s.Key, s.Value(cfg))
	}

	b.WriteString("\nMODEL\n")
	fmt.Fprintf(&b, "  %-14s  %s", "current", cfg.CurrentModel)
	if m, ok := findModel(cfg.CurrentModel); ok {
		if isModelDownloaded(m) {
			fmt.Fprintf(&b, "  (downloaded, %s)", m.Size)
		} else {
			fmt.Fprintf(&b, "  (NOT downloaded — run /download %s)", m.Name)
		}
	}
	b.WriteByte('\n')
	if trained := currentModelTrainedContext(); trained > 0 {
		fmt.Fprintf(&b, "  %-14s  %d tokens\n", "trained ctx", trained)
	}

	b.WriteString("\nENGINE\n")
	status := "NOT downloaded — run /download engine"
	if isEngineDownloaded() {
		status = "downloaded"
	}
	fmt.Fprintf(&b, "  %-14s  %s\n", "status", status)
	fmt.Fprintf(&b, "  %-14s  %s\n", "variant", installedEngineVariant())

	b.WriteString("\nSESSION  (resets when you quit)\n")
	fmt.Fprintf(&b, "  %-14s  %s\n", "tools", onOff(st.toolsEnabled))
	yes := onOff(st.yesman)
	if st.yesman {
		yes += "  ⚠ destructive tools run without confirmation"
	}
	fmt.Fprintf(&b, "  %-14s  %s\n", "yesman", yes)
	fmt.Fprintf(&b, "  %-14s  %d configured, %d connected, %d tools\n",
		"mcp", st.mcpServers, st.mcpConnected, st.mcpTools)

	fmt.Fprintf(&b, "\nMEMORY\n  %s\n", memoryDisplay(cfg))

	b.WriteString("\nFILES\n")
	if p, err := configPath(); err == nil {
		fmt.Fprintf(&b, "  %-14s  %s\n", "config", p)
	}
	if p, err := mcpConfigPath(); err == nil {
		fmt.Fprintf(&b, "  %-14s  %s\n", "mcp servers", p)
	}
	if p, err := mcpAuthPath(); err == nil {
		fmt.Fprintf(&b, "  %-14s  %s\n", "mcp tokens", p)
	}
	if p, err := modelsDir(); err == nil {
		fmt.Fprintf(&b, "  %-14s  %s\n", "models", p)
	}
	if p, err := engineDir(); err == nil {
		fmt.Fprintf(&b, "  %-14s  %s\n", "engine", p)
	}

	return strings.TrimRight(b.String(), "\n")
}

func onOff(b bool) string {
	if b {
		return "on"
	}
	return "off"
}
