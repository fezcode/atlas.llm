package engine

import (
	"archive/tar"
	"compress/gzip"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// requireSymlinks skips on Windows, where creating a symlink needs elevated
// privileges or Developer Mode. Nothing is lost: Windows downloads the
// llama.cpp `.zip` asset, so extractTarGz is never reached there.
func requireSymlinks(t *testing.T) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("symlink extraction is not exercised on Windows (engine ships as .zip)")
	}
}

// tarEntry is one member of a synthetic archive built by makeTarGz.
type tarEntry struct {
	name     string
	body     string
	linkname string // non-empty => symlink entry
}

func makeTarGz(t *testing.T, entries []tarEntry) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "archive.tar.gz")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create archive: %v", err)
	}
	defer f.Close()
	gz := gzip.NewWriter(f)
	tw := tar.NewWriter(gz)
	for _, e := range entries {
		hdr := &tar.Header{Name: e.name, Mode: 0755}
		if e.linkname != "" {
			hdr.Typeflag = tar.TypeSymlink
			hdr.Linkname = e.linkname
		} else {
			hdr.Typeflag = tar.TypeReg
			hdr.Size = int64(len(e.body))
		}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatalf("write header %s: %v", e.name, err)
		}
		if e.linkname == "" {
			if _, err := tw.Write([]byte(e.body)); err != nil {
				t.Fatalf("write body %s: %v", e.name, err)
			}
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("close tar: %v", err)
	}
	if err := gz.Close(); err != nil {
		t.Fatalf("close gzip: %v", err)
	}
	return path
}

// The llama.cpp macOS/Linux tarballs ship versioned dylibs behind a chain of
// symlinks. Dropping them leaves llama-server with unresolvable @rpath
// references, so it dies at exec time — extraction must recreate them.
func TestExtractTarGzPreservesSymlinkChain(t *testing.T) {
	requireSymlinks(t)
	src := makeTarGz(t, []tarEntry{
		{name: "libggml.0.18.1.dylib", body: "real library bytes"},
		{name: "libggml.0.dylib", linkname: "libggml.0.18.1.dylib"},
		{name: "libggml.dylib", linkname: "libggml.0.dylib"},
	})
	dest := t.TempDir()
	if err := extractTarGz(src, dest); err != nil {
		t.Fatalf("extractTarGz: %v", err)
	}

	// The head of the chain must resolve through to the real file.
	head := filepath.Join(dest, "libggml.dylib")
	info, err := os.Lstat(head)
	if err != nil {
		t.Fatalf("lstat %s: %v", head, err)
	}
	if info.Mode()&os.ModeSymlink == 0 {
		t.Errorf("%s is not a symlink (mode %v)", head, info.Mode())
	}
	got, err := os.ReadFile(head)
	if err != nil {
		t.Fatalf("read through symlink chain: %v", err)
	}
	if string(got) != "real library bytes" {
		t.Errorf("symlink resolved to %q, want %q", got, "real library bytes")
	}
}

func TestExtractTarGzRejectsEscapingSymlink(t *testing.T) {
	requireSymlinks(t)
	src := makeTarGz(t, []tarEntry{
		{name: "evil.dylib", linkname: "../../../../etc/passwd"},
	})
	if err := extractTarGz(src, t.TempDir()); err == nil {
		t.Fatal("expected extractTarGz to reject a symlink escaping the archive root")
	}
}

// Re-running /download over an existing engine dir must not fail on EEXIST.
func TestExtractTarGzOverwritesExistingSymlink(t *testing.T) {
	requireSymlinks(t)
	src := makeTarGz(t, []tarEntry{
		{name: "lib.0.dylib", body: "v2"},
		{name: "lib.dylib", linkname: "lib.0.dylib"},
	})
	dest := t.TempDir()
	if err := extractTarGz(src, dest); err != nil {
		t.Fatalf("first extract: %v", err)
	}
	if err := extractTarGz(src, dest); err != nil {
		t.Fatalf("second extract over existing install: %v", err)
	}
	got, err := os.ReadFile(filepath.Join(dest, "lib.dylib"))
	if err != nil {
		t.Fatalf("read after re-extract: %v", err)
	}
	if string(got) != "v2" {
		t.Errorf("got %q, want %q", got, "v2")
	}
}
