package tui

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"atlas.llm/internal/catalog"
	"atlas.llm/internal/config"
	"atlas.llm/internal/engine"
)

// People type a machine's address a dozen different ways for something on
// their own LAN. All of them have to reach the same server.
func TestNormalizeEndpoint(t *testing.T) {
	want := "http://192.168.1.50:8080"
	for _, in := range []string{
		"192.168.1.50",
		"192.168.1.50:8080",
		"http://192.168.1.50:8080",
		"http://192.168.1.50:8080/",
		"  192.168.1.50  ",
		"http://192.168.1.50",
	} {
		got, err := config.NormalizeEndpoint(in)
		if err != nil {
			t.Errorf("normalizeEndpoint(%q) errored: %v", in, err)
			continue
		}
		if got != want {
			t.Errorf("normalizeEndpoint(%q) = %q, want %q", in, got, want)
		}
	}

	// A non-default port must survive.
	if got, _ := config.NormalizeEndpoint("gpu-box:9001"); got != "http://gpu-box:9001" {
		t.Errorf("explicit port = %q", got)
	}
	// Paths are ours to append, so they're stripped rather than concatenated
	// into "/v1/chat/completions/v1/chat/completions".
	if got, _ := config.NormalizeEndpoint("http://host:8080/v1"); got != "http://host:8080" {
		t.Errorf("path not stripped: %q", got)
	}

	// Clearing the setting.
	for _, in := range []string{"", "   ", "local", "LOCAL", "off", "none"} {
		got, err := config.NormalizeEndpoint(in)
		if err != nil || got != "" {
			t.Errorf("normalizeEndpoint(%q) = %q, %v; want \"\", nil", in, got, err)
		}
	}

	// Garbage must be refused at set time, not at the first message.
	for _, in := range []string{"ftp://host:9", "http://", "://nope"} {
		if got, err := config.NormalizeEndpoint(in); err == nil {
			t.Errorf("normalizeEndpoint(%q) = %q, nil; want an error", in, got)
		}
	}
}

// fakeLlamaServer stands in for a served engine, so the client path can be
// tested without a GPU, a model file, or a network.
func fakeLlamaServer(t *testing.T, modelPath string) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"status":"ok"}`)
	})
	mux.HandleFunc("/props", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model_path":                  modelPath,
			"default_generation_settings": map[string]any{"n_ctx": 8192},
		})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

// A remote server is described by what it reports, since the client has no
// GGUF to read.
func TestNewRemoteServerReadsProps(t *testing.T) {
	srv := fakeLlamaServer(t, `C:\models\Qwen3.5-9B-Q4_K_M.gguf`)

	s, err := engine.NewRemoteServer(srv.URL, "")
	if err != nil {
		t.Fatalf("attach failed: %v", err)
	}
	if !s.IsRemote() {
		t.Error("a server we did not spawn must report isRemote")
	}
	if s.Model.Name != "Qwen3.5-9B-Q4_K_M" {
		t.Errorf("model name = %q, want the basename without .gguf", s.Model.Name)
	}
	if s.CtxN != 8192 {
		t.Errorf("ctx = %d, want the server's 8192", s.CtxN)
	}
	if got := s.Url("/v1/chat/completions"); got != srv.URL+"/v1/chat/completions" {
		t.Errorf("url() = %q", got)
	}
}

// Killing a process we never started is the one unrecoverable mistake here:
// one client quitting would take the model away from everyone else.
func TestRemoteServerIsNeverKilled(t *testing.T) {
	srv := fakeLlamaServer(t, "model.gguf")
	s, err := engine.NewRemoteServer(srv.URL, "")
	if err != nil {
		t.Fatal(err)
	}
	if s.Cmd != nil {
		t.Fatal("a remote server must own no subprocess")
	}
	s.StopLocked() // must not panic on the nil cmd
	// And it must not erase shared KV slots.
	if err := s.DropKVCache(); err != nil {
		t.Errorf("DropKVCache on a remote returned %v, want nil", err)
	}
}

// An unreachable endpoint is the common failure — a typo'd address, a
// firewall, a server that isn't up. The message has to point at that rather
// than at a local startup problem.
func TestRemoteServerUnreachableErrorNamesTheAddress(t *testing.T) {
	// Port 1 on loopback: nothing listens, and it fails fast.
	_, err := engine.NewRemoteServer("http://127.0.0.1:1", "")
	if err == nil {
		t.Fatal("expected an error attaching to a dead endpoint")
	}
	if !strings.Contains(err.Error(), "127.0.0.1:1") {
		t.Errorf("error does not name the address: %v", err)
	}
	if !strings.Contains(err.Error(), "--serve") {
		t.Errorf("error does not suggest the fix: %v", err)
	}
}

// The key must reach the server, or an --api-key deployment rejects every
// request with a 401 the user has no way to diagnose.
func TestRemoteServerSendsAPIKey(t *testing.T) {
	var gotAuth string
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		fmt.Fprint(w, `{"status":"ok"}`)
	})
	mux.HandleFunc("/props", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{}`)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	if _, err := engine.NewRemoteServer(srv.URL, "sekrit"); err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer sekrit" {
		t.Errorf("Authorization = %q, want \"Bearer sekrit\"", gotAuth)
	}
}

// A client has no engine and no weights — refusing to run without them would
// reject the exact setup this feature exists to support.
func TestRemoteModeSkipsLocalRequirements(t *testing.T) {
	withTempHome(t)
	noGPU(t)

	// No engine, no model: local mode must refuse.
	if err := config.SaveConfig(config.Config{CurrentModel: catalog.DefaultModel}); err != nil {
		t.Fatal(err)
	}
	if err := engine.RequireInference(); err == nil {
		t.Error("local mode with no engine installed should refuse to run")
	}

	// Same machine, endpoint set: allowed.
	if err := config.SaveConfig(config.Config{CurrentModel: catalog.DefaultModel, Endpoint: "192.168.1.50:8080"}); err != nil {
		t.Fatal(err)
	}
	if err := engine.RequireInference(); err != nil {
		t.Errorf("remote mode still demanded local files: %v", err)
	}
	if !config.IsRemoteMode() {
		t.Error("isRemoteMode should be true with an endpoint configured")
	}
}

// Settings passed to llama-server at spawn can't be changed by a client, and
// must be reported as such rather than silently saved.
func TestRemoteDecidesSetting(t *testing.T) {
	for _, k := range []string{"ctx_size", "gpu_layers", "engine_variant", "CTX_SIZE"} {
		if !config.RemoteDecidesSetting(k) {
			t.Errorf("%q is decided at spawn, so the server owns it", k)
		}
	}
	// These are per-request or purely local, so a client still controls them.
	for _, k := range []string{"max_tokens", "reasoning", "max_tool_rounds", "endpoint"} {
		if config.RemoteDecidesSetting(k) {
			t.Errorf("%q is not fixed at spawn — the client keeps it", k)
		}
	}
}

// A wildcard bind is not an address a client can connect to, so requests
// from the serving process itself must still go to loopback.
func TestServeOptionsHosts(t *testing.T) {
	cases := []struct{ bind, wantBind, wantClient string }{
		{"", "127.0.0.1", "127.0.0.1"},
		{"0.0.0.0", "0.0.0.0", "127.0.0.1"},
		{"::", "::", "127.0.0.1"},
		{"192.168.1.50", "192.168.1.50", "192.168.1.50"},
	}
	for _, tc := range cases {
		o := engine.ServeOptions{Bind: tc.bind}
		if got := o.BindHost(); got != tc.wantBind {
			t.Errorf("bind %q: bindHost = %q, want %q", tc.bind, got, tc.wantBind)
		}
		if got := o.ClientHost(); got != tc.wantClient {
			t.Errorf("bind %q: clientHost = %q, want %q", tc.bind, got, tc.wantClient)
		}
	}
}

// A wildcard bind has to expand into addresses someone can actually paste.
func TestClientURLs(t *testing.T) {
	got := engine.ClientURLs(engine.ServeOptions{Bind: "192.168.1.50", Port: 9001})
	if len(got) != 1 || got[0] != "http://192.168.1.50:9001" {
		t.Errorf("explicit bind = %v", got)
	}
	for _, u := range engine.ClientURLs(engine.ServeOptions{Bind: "0.0.0.0", Port: 8080}) {
		if strings.Contains(u, "0.0.0.0") {
			t.Errorf("wildcard leaked into a client URL: %s", u)
		}
	}
	// lanAddresses must never offer loopback as a LAN address.
	for _, ip := range engine.LanAddresses() {
		if strings.HasPrefix(ip, "127.") {
			t.Errorf("loopback %s offered as a LAN address", ip)
		}
	}
}

// Serving means running the model here, so being configured as someone
// else's client at the same time is a contradiction, not a default to guess.
func TestServeRefusesWhenConfiguredAsClient(t *testing.T) {
	withTempHome(t)
	if err := config.SaveConfig(config.Config{CurrentModel: catalog.DefaultModel, Endpoint: "192.168.1.50:8080"}); err != nil {
		t.Fatal(err)
	}
	err := engine.RunServe(engine.ServeOptions{Bind: "0.0.0.0", Port: 8080})
	if err == nil {
		t.Fatal("serving while configured as a client should be refused")
	}
	if !strings.Contains(err.Error(), "endpoint local") {
		t.Errorf("error does not say how to fix it: %v", err)
	}
}

// The KV budget must not scale with slot count: raising slots to serve more
// clients would otherwise multiply VRAM use and fail to load the model.
func TestServeSlotsDivideContextRatherThanMultiplyVRAM(t *testing.T) {
	const ctxN = 16384
	_, base := engine.ServeCapacity(ctxN, 0)
	if base != ctxN*engine.KvCacheSlots {
		t.Fatalf("default budget = %d, want %d", base, ctxN*engine.KvCacheSlots)
	}

	for _, want := range []int{2, 4, 8} {
		slots, total := engine.ServeCapacity(ctxN, want)
		if slots != want {
			t.Errorf("serveCapacity(_, %d) gave %d slots", want, slots)
		}
		if total != base {
			t.Errorf("slots=%d moved the KV budget to %d, want it fixed at %d — "+
				"a budget that scales with slots would OOM the card it was sized for",
				want, total, base)
		}
		if perSlot := total / slots; perSlot*slots != total {
			t.Errorf("slots=%d: %d per slot does not divide %d", want, perSlot, total)
		}
	}

	// The TUI's own server must be unaffected by any of this.
	slots, total := engine.ServeCapacity(ctxN, 0)
	if slots != engine.KvCacheSlots || total/slots != ctxN {
		t.Errorf("default launch changed: %d slots of %d, want %d of %d",
			slots, total/slots, engine.KvCacheSlots, ctxN)
	}
}
