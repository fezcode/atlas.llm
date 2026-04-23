package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime/debug"
)

// logPath returns the path to atlas.llm.log inside the data dir.
func logPath() (string, error) {
	dir, err := atlasDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "atlas.llm.log"), nil
}

// setupLogging opens the TUI log file for append, routes the stdlib logger
// to it, and returns a close function plus the resolved path. Safe to call
// even if the file cannot be opened — logging is silently disabled in that
// case and the returned closer is a no-op.
func setupLogging() (string, func(), error) {
	p, err := logPath()
	if err != nil {
		return "", func() {}, err
	}
	f, err := os.OpenFile(p, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return p, func() {}, err
	}
	log.SetOutput(f)
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	log.Printf("=== atlas.llm session start ===")
	return p, func() {
		log.Printf("=== atlas.llm session end ===")
		_ = f.Close()
	}, nil
}

// logPanicln dumps panic info + stack trace to the log file. Intended for
// use inside a deferred recover() so the TUI alt-screen teardown doesn't
// eat the traceback.
func logPanicln(v any) {
	log.Printf("PANIC: %v\n%s", v, debug.Stack())
	fmt.Fprintf(os.Stderr, "atlas.llm crashed: %v\n", v)
	if p, err := logPath(); err == nil {
		fmt.Fprintf(os.Stderr, "full stack trace written to %s\n", p)
	}
}
