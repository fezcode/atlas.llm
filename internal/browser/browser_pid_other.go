//go:build !windows

package browser

import (
	"errors"
	"os"
	"syscall"
)

// pidAlive reports whether a process with this pid is still running. Signal 0
// runs the kernel's existence and permission checks without delivering
// anything; EPERM means the process is there but owned by someone else, which
// still counts as alive.
func pidAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	p, err := os.FindProcess(pid) // never fails on unix
	if err != nil {
		return false
	}
	err = p.Signal(syscall.Signal(0))
	return err == nil || errors.Is(err, syscall.EPERM)
}
