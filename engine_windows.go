//go:build windows

package main

import (
	"os/exec"
	"syscall"
)

// Windows-only process-creation flags for spawning llama-cli.exe:
//   - CREATE_NO_WINDOW (0x08000000): no console window, so llama-cli never
//     attaches to our TUI's console. Safe because stdout/stderr are piped.
//   - CREATE_NEW_PROCESS_GROUP (0x00000200): isolates the child from
//     Ctrl+C / Ctrl+Break events dispatched to our console group.
const (
	createNoWindow         = 0x08000000
	createNewProcessGroup  = 0x00000200
	engineChildCreateFlags = createNoWindow | createNewProcessGroup
)

func applyEngineSysProcAttr(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		HideWindow:    true,
		CreationFlags: engineChildCreateFlags,
	}
}
