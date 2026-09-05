package main

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
)

// handleConfig implements /config and its profile subcommands. Bare /config
// still shows the current setup; save/load/list/delete manage named configs.
func (m *chatModel) handleConfig(args []string) tea.Cmd {
	if len(args) == 0 {
		cfg, err := loadConfig()
		if err != nil {
			m.pushError("load config: " + err.Error())
			return nil
		}
		out := renderConfig(cfg, m.configState())
		if name := matchingProfile(cfg); name != "" {
			out = "profile: " + name + "\n" + out
		}
		m.pushSystem(out)
		return nil
	}
	sub := strings.ToLower(args[0])
	rest := args[1:]
	switch sub {
	case "list", "ls":
		m.handleConfigList()
	case "save":
		if len(rest) == 0 {
			m.pushError("usage: /config save <name>")
			return nil
		}
		m.handleConfigSave(rest[0])
	case "show", "view", "cat":
		if len(rest) == 0 {
			m.pushError("usage: /config show <name>")
			return nil
		}
		m.handleConfigShow(rest[0])
	case "load", "use":
		if len(rest) == 0 {
			m.pushError("usage: /config load <name>")
			return nil
		}
		return m.handleConfigLoad(rest[0])
	case "delete", "rm", "remove":
		if len(rest) == 0 {
			m.pushError("usage: /config delete <name>")
			return nil
		}
		m.handleConfigDelete(rest[0])
	default:
		m.pushError(fmt.Sprintf("unknown /config arg: %s (expected save, load, show, list, or delete)", sub))
	}
	return nil
}

func (m *chatModel) handleConfigSave(name string) {
	if err := validProfileName(name); err != nil {
		m.pushError(err.Error())
		return
	}
	cfg, err := loadConfig()
	if err != nil {
		m.pushError("load config: " + err.Error())
		return
	}
	existed := false
	if names, _ := listProfiles(); names != nil {
		for _, n := range names {
			if n == name {
				existed = true
			}
		}
	}
	if err := saveProfile(name, cfg); err != nil {
		m.pushError("save profile: " + err.Error())
		return
	}
	verb := "Saved"
	if existed {
		verb = "Overwrote"
	}
	m.pushSystem(fmt.Sprintf("%s the current settings as profile %q. Load it later with `/config load %s`.", verb, name, name))
}

func (m *chatModel) handleConfigList() {
	names, err := listProfiles()
	if err != nil {
		m.pushError("list profiles: " + err.Error())
		return
	}
	if len(names) == 0 {
		m.pushSystem("No saved profiles yet. Save the current settings with `/config save <name>` (e.g. `/config save fast`).")
		return
	}
	cfg, _ := loadConfig()
	active := matchingProfile(cfg)
	var b strings.Builder
	b.WriteString("Saved profiles:\n")
	for _, n := range names {
		marker := "  "
		if n == active {
			marker = "● "
		}
		b.WriteString("  " + marker + n + "\n")
	}
	b.WriteString("\n● = matches current settings · `/config load <name>` to switch")
	if active == "" {
		b.WriteString("\n(current settings don't match any saved profile — `/config save <name>` to keep them)")
	}
	m.pushSystem(b.String())
}

func (m *chatModel) handleConfigShow(name string) {
	cfg, err := loadProfile(name)
	if err != nil {
		m.pushError(err.Error())
		return
	}
	active, _ := loadConfig()
	m.pushSystem(renderProfile(name, cfg, configsEqual(cfg, active)))
}

func (m *chatModel) handleConfigDelete(name string) {
	if err := deleteProfile(name); err != nil {
		m.pushError(err.Error())
		return
	}
	m.pushSystem(fmt.Sprintf("Deleted profile %q.", name))
}

// handleConfigLoad makes a saved profile the active config: it overwrites
// config.json, restarts the model server so model/context/tuning changes take
// effect, and re-syncs the session state that mirrors config (tools, ama,
// remote endpoint).
func (m *chatModel) handleConfigLoad(name string) tea.Cmd {
	loaded, err := loadProfile(name)
	if err != nil {
		m.pushError(err.Error())
		return nil
	}
	prevEndpoint, _ := remoteEndpoint()
	if err := saveConfig(loaded); err != nil {
		m.pushError("save config: " + err.Error())
		return nil
	}

	// Session state that shadows the config file.
	m.agentEnabled = loaded.ToolsEnabled
	if !loaded.ToolsEnabled {
		m.agentMsgs = nil
		m.pendingCalls = nil
	}
	amaOn.Store(loaded.AMAEnabled)

	// The running server was built for the old settings; drop it so the next
	// message starts one for the new model/context/tuning.
	shutdownServer()

	// Track a remote-endpoint change the same way `/set endpoint` does, so the
	// header badge and heartbeat reflect reality rather than the old target.
	newEndpoint, key := remoteEndpoint()
	if newEndpoint != prevEndpoint {
		if newEndpoint == "" {
			clearRemoteStatus()
		} else {
			setRemoteStatus(probeRemote(newEndpoint, key))
			startHeartbeat()
		}
	}

	m.pushSystem(fmt.Sprintf("Loaded profile %q. Takes effect on your next message (the model server restarts).\n\n%s",
		name, renderConfig(loaded, m.configState())))
	return nil
}
