//go:build windows

package browser

import "syscall"

// pidAlive reports whether a process with this pid is still running.
//
// os.Process.Signal cannot answer this on Windows — it has no signal 0 — so
// ask the kernel: open a handle and read the exit code. STILL_ACTIVE means
// the process has not exited. PROCESS_QUERY_LIMITED_INFORMATION is enough for
// GetExitCodeProcess and is granted more widely than the full query right.
func pidAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	const processQueryLimitedInformation = 0x1000
	h, err := syscall.OpenProcess(processQueryLimitedInformation, false, uint32(pid))
	if err != nil {
		return false
	}
	defer syscall.CloseHandle(h)
	var code uint32
	if err := syscall.GetExitCodeProcess(h, &code); err != nil {
		return false
	}
	const stillActive = 259
	return code == stillActive
}
