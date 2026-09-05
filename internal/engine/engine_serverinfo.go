package engine

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// atlasInfoPath is the sidecar endpoint. It lives beside llama-server rather
// than in front of it: llama.cpp keeps the inference port untouched, so
// nothing this file does can affect streaming.
const AtlasInfoPath = "/atlas/info"

// atlasServerInfo is what a served instance says about itself. llama-server's
// /props covers the model and slots, but knows nothing about atlas — the
// engine variant, the -ngl it was launched with, and the atlas version are
// all invisible from there, and they are exactly what a client needs to
// explain what it is connected to.
type AtlasServerInfo struct {
	Service       string `json:"service"` // always "atlas.llm"
	Version       string `json:"version"`
	Model         string `json:"model"` // registry name, not the file path
	ModelFile     string `json:"model_file,omitempty"`
	EngineVariant string `json:"engine_variant"`
	GPULayers     int    `json:"gpu_layers"`
	CtxPerSlot    int    `json:"ctx_per_slot"`
	Slots         int    `json:"slots"`
	InferencePort int    `json:"inference_port"`
	AuthRequired  bool   `json:"auth_required"`
	UptimeSeconds int64  `json:"uptime_seconds"`
	LlamaBuild    string `json:"llama_build,omitempty"`
}

// infoPortFor derives the sidecar port from the inference port. A convention
// beats configuration here: the client has to find it with nothing but the
// endpoint the user typed, and asking people to remember two numbers for one
// server is how the feature goes unused.
func infoPortFor(inferencePort int) int { return inferencePort + 1 }

// infoURLFor turns an inference endpoint into its sidecar URL.
// "http://gpu-box:8080" becomes "http://gpu-box:8081/atlas/info".
func InfoURLFor(endpoint string) (string, bool) {
	u, err := url.Parse(strings.TrimRight(endpoint, "/"))
	if err != nil || u.Host == "" {
		return "", false
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil || port <= 0 || port >= 65535 {
		return "", false
	}
	u.Host = fmt.Sprintf("%s:%d", u.Hostname(), infoPortFor(port))
	u.Path = AtlasInfoPath
	return u.String(), true
}

// startInfoServer publishes info on the sidecar port. A failure here is not
// fatal: inference is the job, and a client that can't reach the sidecar
// degrades to what llama-server's own /props reports.
func startInfoServer(bind string, port int, info func() AtlasServerInfo) (*http.Server, error) {
	mux := http.NewServeMux()
	mux.HandleFunc(AtlasInfoPath, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		_ = enc.Encode(info())
	})
	srv := &http.Server{
		Addr:              fmt.Sprintf("%s:%d", bind, port),
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	ln, err := listenTCP(srv.Addr)
	if err != nil {
		return nil, err
	}
	go func() { _ = srv.Serve(ln) }()
	return srv, nil
}

// fetchAtlasInfo asks a remote what it is. Returns ok=false for a server
// that isn't atlas.llm, or one too old to have a sidecar — both are normal,
// and the caller falls back to /props.
func FetchAtlasInfo(client *http.Client, endpoint, apiKey string) (AtlasServerInfo, bool) {
	var info AtlasServerInfo
	u, ok := InfoURLFor(endpoint)
	if !ok {
		return info, false
	}
	req, err := http.NewRequest("GET", u, nil)
	if err != nil {
		return info, false
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := client.Do(req)
	if err != nil {
		return info, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return info, false
	}
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return info, false
	}
	// Anything can answer on a port; only trust a payload that identifies
	// itself as ours.
	if info.Service != "atlas.llm" {
		return info, false
	}
	return info, true
}

// formatUptime renders a duration the way a status line should read: coarse
// and short, never "2h13m47.221s".
func FormatUptime(seconds int64) string {
	d := time.Duration(seconds) * time.Second
	switch {
	case d < time.Minute:
		return fmt.Sprintf("%ds", int(d.Seconds()))
	case d < time.Hour:
		return fmt.Sprintf("%dm", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%dh%dm", int(d.Hours()), int(d.Minutes())%60)
	}
	return fmt.Sprintf("%dd%dh", int(d.Hours())/24, int(d.Hours())%24)
}

// listenTCP is split out so the caller sees a bind failure immediately
// rather than discovering it asynchronously inside Serve.
func listenTCP(addr string) (net.Listener, error) {
	return net.Listen("tcp", addr)
}

// writeJSON is a small helper shared with tests.
func WriteJSON(w io.Writer, v any) error {
	return json.NewEncoder(w).Encode(v)
}

// endpointOneBelow turns "http://host:P" into "http://host:P-1", so a test
// server standing in for the sidecar can be addressed through the same
// port+1 derivation the client uses.
func EndpointOneBelow(endpoint string) (string, bool) {
	u, err := url.Parse(endpoint)
	if err != nil || u.Host == "" {
		return "", false
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil || port <= 1 {
		return "", false
	}
	u.Host = fmt.Sprintf("%s:%d", u.Hostname(), port-1)
	u.Path = ""
	return u.String(), true
}
