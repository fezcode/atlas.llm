//go:build !windows

package engine

import "os/exec"

func ApplyEngineSysProcAttr(cmd *exec.Cmd) {}
