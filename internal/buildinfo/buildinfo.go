// Package buildinfo carries the version string the build stamps in.
//
// It exists so that engine, mcp and tools can report the version without
// importing package main, which Go forbids. cmd/atlas.llm keeps the
// -ldflags target (main.Version) and copies it here at startup.
package buildinfo

// Version is the running build's version, set from main at startup.
var Version = "dev"
