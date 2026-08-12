package main

import (
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

// The sidecar port is a convention, not a setting — the client has to find it
// knowing only the endpoint the user typed.
func TestInfoURLFor(t *testing.T) {
	cases := map[string]string{
		"http://192.168.1.50:8080": "http://192.168.1.50:8081/atlas/info",
		"http://gpu-box:9001":      "http://gpu-box:9002/atlas/info",
		"https://host:443":         "https://host:444/atlas/info",
	}
	for in, want := range cases {
		got, ok := infoURLFor(in)
		if !ok {
			t.Errorf("infoURLFor(%q) failed", in)
			continue
		}
		if got != want {
			t.Errorf("infoURLFor(%q) = %q, want %q", in, got, want)
		}
	}
	// A port is required — normalizeEndpoint always supplies one, so an
	// endpoint without it means something upstream went wrong.
	for _, in := range []string{"http://host", "", "::::"} {
		if got, ok := infoURLFor(in); ok {
			t.Errorf("infoURLFor(%q) = %q, true; want failure", in, got)
		}
	}
}

// infoServer stands up a sidecar so the client path can be exercised without
// a GPU or a model.
func infoServer(t *testing.T, payload any) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc(atlasInfoPath, func(w http.ResponseWriter, r *http.Request) {
		switch v := payload.(type) {
		case string:
			fmt.Fprint(w, v)
		default:
			w.Header().Set("Content-Type", "application/json")
			_ = writeJSON(w, v)
		}
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func TestFetchAtlasInfo(t *testing.T) {
	want := atlasServerInfo{
		Service: "atlas.llm", Version: "0.31.0", Model: "ministral-3-14b-instruct",
		EngineVariant: "cuda", GPULayers: 999, CtxPerSlot: 8192, Slots: 4,
	}
	srv := infoServer(t, want)

	// infoURLFor derives port+1, so point it one below the test server's.
	base, ok := endpointOneBelow(srv.URL)
	if !ok {
		t.Fatalf("could not derive a base from %s", srv.URL)
	}
	got, found := fetchAtlasInfo(srv.Client(), base, "")
	if !found {
		t.Fatal("sidecar did not answer")
	}
	if got.Model != want.Model || got.EngineVariant != "cuda" || got.GPULayers != 999 {
		t.Errorf("info = %+v", got)
	}
}

// Anything can be listening on port+1. Only a payload that identifies itself
// as atlas.llm may be believed.
func TestFetchAtlasInfoRejectsStrangers(t *testing.T) {
	for _, payload := range []any{
		map[string]string{"service": "something-else"},
		map[string]string{"hello": "world"},
		"not json at all",
	} {
		srv := infoServer(t, payload)
		base, _ := endpointOneBelow(srv.URL)
		if _, ok := fetchAtlasInfo(srv.Client(), base, ""); ok {
			t.Errorf("believed a non-atlas payload: %v", payload)
		}
	}
}

// A server that answers inference but has no sidecar is normal — a plain
// llama-server, or an older atlas. It must not read as a failure.
func TestProbeHealthyWithoutSidecar(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"status":"ok"}`)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	st := probeRemote(srv.URL, "")
	if st.State != remoteHealthy {
		t.Errorf("state = %v, want healthy", st.State)
	}
	if st.HaveInfo {
		t.Error("claimed to have info from a server with no sidecar")
	}
	out := renderEndpointProbe(srv.URL, st)
	if !strings.Contains(out, "✓ connected") {
		t.Errorf("probe output does not report success:\n%s", out)
	}
	if !strings.Contains(out, "/atlas/info") {
		t.Errorf("probe output does not explain the missing detail:\n%s", out)
	}
}

// A key-protected server must be reported as such, not as "unreachable" with
// no explanation — the fix is a setting, and the message has to name it.
func TestProbeReportsMissingKey(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") == "" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		fmt.Fprint(w, `{"status":"ok"}`)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	st := probeRemote(srv.URL, "")
	if st.State == remoteHealthy {
		t.Fatal("a 401 must not read as connected")
	}
	if !strings.Contains(st.LastErr, "endpoint_key") {
		t.Errorf("error does not name the fix: %q", st.LastErr)
	}
	// With the key it succeeds.
	if st := probeRemote(srv.URL, "k"); st.State != remoteHealthy {
		t.Errorf("with a key, state = %v, want healthy", st.State)
	}
}

// Go's network errors are accurate and unreadable. The user needs the part
// they can act on.
func TestShortNetError(t *testing.T) {
	cases := map[string]string{
		`dial tcp 10.0.0.5:8080: connectex: No connection could be made because the target machine actively refused it.`: "connection refused",
		`dial tcp: lookup gpu-box: no such host`: "host not found",
		`dial tcp 10.0.0.5:8080: i/o timeout`:    "timed out",
		`context deadline exceeded`:              "timed out",
	}
	for in, wantSubstr := range cases {
		got := shortNetError(errors.New(in))
		if !strings.Contains(got, wantSubstr) {
			t.Errorf("shortNetError(%q) = %q, want it to mention %q", in, got, wantSubstr)
		}
	}
}

// The badge renders on every keystroke. It must read cached state and never
// touch the network — this is the exact shape of the bug that made typing
// cost 180ms a frame.
func TestRemoteBadgeIsCacheOnlyAndStateColoured(t *testing.T) {
	withTempHome(t)
	t.Cleanup(clearRemoteStatus)

	// Local: no badge at all. Absence is the clearest signal.
	clearRemoteStatus()
	if err := saveConfig(Config{CurrentModel: defaultModel}); err != nil {
		t.Fatal(err)
	}
	if got := renderRemoteBadge(); got != "" {
		t.Errorf("local mode rendered a badge: %q", got)
	}

	if err := saveConfig(Config{CurrentModel: defaultModel, Endpoint: "gpu-box:8080"}); err != nil {
		t.Fatal(err)
	}
	for _, state := range []remoteState{remoteUnknown, remoteHealthy, remoteDegraded, remoteUnreachable} {
		setRemoteStatus(remoteStatus{Endpoint: "http://gpu-box:8080", State: state})
		got := renderRemoteBadge()
		if !strings.Contains(got, "gpu-box:8080") {
			t.Errorf("state %v: badge does not name the host: %q", state, got)
		}
		if strings.Contains(got, "http://") {
			t.Errorf("state %v: scheme wastes header width: %q", state, got)
		}
		if !strings.Contains(got, "REMOTE") {
			t.Errorf("state %v: badge is not labelled: %q", state, got)
		}
	}

	// Cache-only, and that means no disk either: reading config.json per
	// frame would be a milder repeat of the tasklist-per-keystroke bug.
	// Delete the config outright — if rendering still works, it never
	// looked.
	setRemoteStatus(remoteStatus{Endpoint: "http://10.255.255.1:8080", State: remoteHealthy})
	if p, err := configPath(); err == nil {
		_ = os.Remove(p)
	}
	if got := renderRemoteBadge(); !strings.Contains(got, "10.255.255.1:8080") {
		t.Errorf("badge needs config.json to render: %q", got)
	}
	start := time.Now()
	for i := 0; i < 5000; i++ {
		_ = renderRemoteBadge()
	}
	if d := time.Since(start); d > 100*time.Millisecond {
		t.Errorf("5000 badge renders took %s — the badge is doing I/O", d)
	}
}

// /config must describe the server, not the local engine that isn't running.
func TestRemoteSectionDescribesTheServer(t *testing.T) {
	withTempHome(t)
	t.Cleanup(clearRemoteStatus)
	if err := saveConfig(Config{CurrentModel: defaultModel, Endpoint: "gpu-box:8080"}); err != nil {
		t.Fatal(err)
	}
	setRemoteStatus(remoteStatus{
		Endpoint: "http://gpu-box:8080",
		State:    remoteHealthy,
		HaveInfo: true,
		Info: atlasServerInfo{
			Service: "atlas.llm", Version: "0.31.0", Model: "ministral-3-14b-instruct",
			EngineVariant: "cuda", GPULayers: maxGPULayers, CtxPerSlot: 8192, Slots: 4,
			UptimeSeconds: 3600, LlamaBuild: "b10375",
		},
	})
	out := renderRemoteSection("http://gpu-box:8080")
	for _, want := range []string{
		"REMOTE", "gpu-box:8080", "connected", "0.31.0",
		"ministral-3-14b-instruct", "cuda", "8192", "b10375", "1h0m",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("REMOTE section missing %q:\n%s", want, out)
		}
	}
	// -ngl 999 is a sentinel, not a layer count anyone should read literally.
	if !strings.Contains(out, "all") {
		t.Errorf("gpu_layers sentinel rendered literally:\n%s", out)
	}
	// It must be clear these belong to the server, not the client.
	if !strings.Contains(out, "not this one") {
		t.Errorf("REMOTE section does not say whose config this is:\n%s", out)
	}
}

// An unreachable server still has to explain itself in /config.
func TestRemoteSectionWhenUnreachable(t *testing.T) {
	withTempHome(t)
	t.Cleanup(clearRemoteStatus)
	setRemoteStatus(remoteStatus{
		Endpoint: "http://gpu-box:8080",
		State:    remoteUnreachable,
		LastErr:  "connection refused — nothing is serving on that port",
	})
	out := renderRemoteSection("http://gpu-box:8080")
	if !strings.Contains(out, "UNREACHABLE") || !strings.Contains(out, "connection refused") {
		t.Errorf("unreachable state not surfaced:\n%s", out)
	}
}

func TestFormatUptime(t *testing.T) {
	cases := map[int64]string{
		5: "5s", 90: "1m", 3600: "1h0m", 5400: "1h30m", 90000: "1d1h",
	}
	for secs, want := range cases {
		if got := formatUptime(secs); got != want {
			t.Errorf("formatUptime(%d) = %q, want %q", secs, got, want)
		}
	}
}
