//go:build !windows

package main

import "os/exec"

func applyEngineSysProcAttr(cmd *exec.Cmd) {}
