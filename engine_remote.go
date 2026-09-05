package main

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Connection states for a remote endpoint, in order of increasing trouble.
type remoteState int

const (
	remoteUnknown     remoteState = iota // not probed yet this session
	remoteHealthy                        // last heartbeat answered
	remoteDegraded                       // one heartbeat missed; not yet given up
	remoteUnreachable                    // repeated failures
)

func (s remoteState) String() string {
	switch s {
	case remoteHealthy:
		return "connected"
	case remoteDegraded:
		return "unsteady"
	case remoteUnreachable:
		return "unreachable"
	}
	return "unknown"
}

// remoteStatus is the cached view of the connection. The header reads this
// and nothing else — probing from a render path is what made every keystroke
// cost 180ms once already, and a status indicator is exactly the kind of
// feature that invites the same mistake.
type remoteStatus struct {
	Endpoint string
	State    remoteState
	Info     atlasServerInfo
	HaveInfo bool
	LastOK   time.Time
	LastErr  string
	// Misses counts consecutive failed heartbeats, so a single dropped
	// packet shows as unsteady rather than jumping straight to unreachable.
	Misses int
}

var (
	remoteMu      sync.RWMutex
	remoteCurrent remoteStatus
	heartbeatOnce sync.Once
	heartbeatStop chan struct{}
)

// heartbeatInterval is slow on purpose. This exists to notice a server that
// went away, not to measure latency; polling harder would put steady traffic
// on someone else's machine for a status dot.
const heartbeatInterval = 15 * time.Second

// probeTimeout bounds a single health check. A LAN round trip is milliseconds,
// so anything past a few seconds is a server that is gone or wedged.
const probeTimeout = 3 * time.Second

// getRemoteStatus returns the cached connection state. Never blocks, never
// performs I/O.
func getRemoteStatus() remoteStatus {
	remoteMu.RLock()
	defer remoteMu.RUnlock()
	return remoteCurrent
}

func setRemoteStatus(s remoteStatus) {
	remoteMu.Lock()
	remoteCurrent = s
	remoteMu.Unlock()
}

// clearRemoteStatus drops the cached view, for going back to local.
func clearRemoteStatus() {
	remoteMu.Lock()
	remoteCurrent = remoteStatus{}
	remoteMu.Unlock()
}

// probeRemote checks an endpoint and describes what answered. This is the
// synchronous version used when the user sets an endpoint and is waiting to
// hear whether it worked; the heartbeat uses it too.
func probeRemote(endpoint, apiKey string) remoteStatus {
	st := remoteStatus{Endpoint: endpoint}
	client := &http.Client{Timeout: probeTimeout}

	ctx, cancel := context.WithTimeout(context.Background(), probeTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", endpoint+"/health", nil)
	if err != nil {
		st.State, st.LastErr = remoteUnreachable, err.Error()
		return st
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := client.Do(req)
	if err != nil {
		st.State = remoteUnreachable
		st.LastErr = shortNetError(err)
		return st
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized {
		st.State = remoteUnreachable
		st.LastErr = "server requires an API key — set one with `/set endpoint_key KEY`"
		return st
	}
	if resp.StatusCode != http.StatusOK {
		st.State = remoteUnreachable
		st.LastErr = fmt.Sprintf("health check returned HTTP %d", resp.StatusCode)
		return st
	}

	st.State = remoteHealthy
	st.LastOK = time.Now()
	// The sidecar is optional: an older atlas, or a plain llama-server,
	// answers inference fine and simply has less to say about itself.
	st.Info, st.HaveInfo = fetchAtlasInfo(client, endpoint, apiKey)
	return st
}

// shortNetError trims Go's layered network errors down to the part a user can
// act on. "dial tcp 192.168.1.9:8080: connectex: No connection could be made
// because the target machine actively refused it." is accurate and useless.
func shortNetError(err error) string {
	msg := err.Error()
	if i := strings.LastIndex(msg, ": "); i >= 0 && i+2 < len(msg) {
		msg = msg[i+2:]
	}
	switch {
	case strings.Contains(msg, "actively refused"), strings.Contains(msg, "connection refused"):
		return "connection refused — nothing is serving on that port"
	case strings.Contains(msg, "no such host"):
		return "host not found — check the address"
	case strings.Contains(msg, "did not properly respond"), strings.Contains(msg, "timeout"),
		strings.Contains(msg, "deadline exceeded"):
		return "timed out — host unreachable, or a firewall is dropping the connection"
	}
	return msg
}

// startHeartbeat begins polling the configured endpoint in the background.
// Safe to call repeatedly; only the first call starts the loop.
func startHeartbeat() {
	heartbeatOnce.Do(func() {
		heartbeatStop = make(chan struct{})
		go heartbeatLoop(heartbeatStop)
	})
}

func stopHeartbeat() {
	remoteMu.Lock()
	stop := heartbeatStop
	heartbeatStop = nil
	remoteMu.Unlock()
	if stop != nil {
		close(stop)
	}
	heartbeatOnce = sync.Once{}
}

func heartbeatLoop(stop <-chan struct{}) {
	ticker := time.NewTicker(heartbeatInterval)
	defer ticker.Stop()
	for {
		select {
		case <-stop:
			return
		case <-ticker.C:
			ep, key := remoteEndpoint()
			if ep == "" {
				clearRemoteStatus()
				continue
			}
			prev := getRemoteStatus()
			next := probeRemote(ep, key)
			// One miss is a blip; two is a problem. Reporting "unreachable"
			// on a single dropped packet would make the indicator noise
			// people learn to ignore.
			if next.State == remoteUnreachable {
				next.Misses = prev.Misses + 1
				if next.Misses < 2 {
					next.State = remoteDegraded
				}
				// Keep the last good description rather than blanking it.
				if prev.HaveInfo {
					next.Info, next.HaveInfo = prev.Info, true
				}
				next.LastOK = prev.LastOK
			}
			setRemoteStatus(next)
		}
	}
}

// renderEndpointProbe reports what answered at an endpoint the user just set.
// A reachable server is described by what it actually is, so the user can see
// they reached the machine they meant; an unreachable one names the failure
// and leaves the setting in place, since the usual cause is a server that
// simply is not started yet.
func renderEndpointProbe(ep string, st remoteStatus) string {
	var b strings.Builder
	fmt.Fprintf(&b, "endpoint = %s\n", ep)

	if st.State != remoteHealthy {
		fmt.Fprintf(&b, "\n  ✗ no answer — %s\n", st.LastErr)
		b.WriteString("\nThe setting is saved, so starting the server will make it work.\n")
		b.WriteString("On that machine: `atlas.llm --serve`\n")
		b.WriteString("`/set endpoint local` moves inference back here.")
		return b.String()
	}

	b.WriteString("\n  ✓ connected\n")
	if st.HaveInfo {
		i := st.Info
		fmt.Fprintf(&b, "    server    atlas.llm v%s, up %s\n", i.Version, formatUptime(i.UptimeSeconds))
		fmt.Fprintf(&b, "    model     %s\n", i.Model)
		fmt.Fprintf(&b, "    context   %d tokens per slot · %d slots\n", i.CtxPerSlot, i.Slots)
		engine := i.EngineVariant
		if engine == "" {
			engine = "unknown"
		}
		fmt.Fprintf(&b, "    engine    %s · %d layers on GPU\n", engine, i.GPULayers)
		if i.LlamaBuild != "" {
			fmt.Fprintf(&b, "    llama.cpp %s\n", i.LlamaBuild)
		}
	} else {
		// Reachable but not describing itself: a plain llama-server, or an
		// atlas too old to have the sidecar. Inference still works.
		b.WriteString("    The server answers inference but not /atlas/info, so there is\n")
		b.WriteString("    no detail to show. A plain llama-server, or an atlas.llm older\n")
		b.WriteString("    than 0.31.0, both look like this.\n")
	}
	b.WriteString("\nThis machine needs no engine or model now. Tools still run here,\n")
	b.WriteString("against your own files; only generation is remote.\n")
	if st.HaveInfo && st.Info.AuthRequired {
		if _, key := remoteEndpoint(); key == "" {
			b.WriteString("\nThe server requires an API key — set it with `/set endpoint_key KEY`.\n")
		}
	}
	b.WriteString("Takes effect on your next message.")
	return b.String()
}

// renderRemoteSection is the REMOTE block in /config. It replaces ENGINE and
// MEMORY, which describe a local llama-server that isn't running.
//
// Reads cached status rather than probing, so /config stays instant. The
// heartbeat keeps it current to within its interval, and "last ok" makes any
// staleness visible instead of pretending otherwise.
func renderRemoteSection(ep string) string {
	var b strings.Builder
	st := getRemoteStatus()

	b.WriteString("\nREMOTE  (inference runs on another machine)\n")
	fmt.Fprintf(&b, "  %-14s  %s\n", "endpoint", ep)

	state := st.State.String()
	switch st.State {
	case remoteHealthy:
		state = "connected"
	case remoteDegraded:
		state = fmt.Sprintf("unsteady — %d missed heartbeat(s), last reply %s ago",
			st.Misses, formatUptime(int64(time.Since(st.LastOK).Seconds())))
	case remoteUnreachable:
		state = "UNREACHABLE — " + st.LastErr
	case remoteUnknown:
		state = "not checked yet"
	}
	fmt.Fprintf(&b, "  %-14s  %s\n", "status", state)

	if _, key := remoteEndpoint(); key != "" {
		fmt.Fprintf(&b, "  %-14s  %s\n", "auth", "endpoint_key set")
	}

	if !st.HaveInfo {
		if st.State == remoteHealthy {
			fmt.Fprintf(&b, "  %-14s  %s\n", "detail",
				"server does not publish /atlas/info — plain llama-server, or atlas < 0.31.0")
		}
		return b.String()
	}

	i := st.Info
	b.WriteString("\n  server\n")
	fmt.Fprintf(&b, "    %-12s  atlas.llm v%s\n", "version", i.Version)
	fmt.Fprintf(&b, "    %-12s  %s\n", "uptime", formatUptime(i.UptimeSeconds))
	if i.LlamaBuild != "" {
		fmt.Fprintf(&b, "    %-12s  %s\n", "llama.cpp", i.LlamaBuild)
	}
	fmt.Fprintf(&b, "    %-12s  %s\n", "auth", boolLabel(i.AuthRequired, "API key required", "open"))

	b.WriteString("\n  its config  (set on that machine, not this one)\n")
	fmt.Fprintf(&b, "    %-12s  %s\n", "model", i.Model)
	fmt.Fprintf(&b, "    %-12s  %d tokens per slot · %d slots\n", "ctx_size", i.CtxPerSlot, i.Slots)
	variant := i.EngineVariant
	if variant == "" {
		variant = "unknown"
	}
	fmt.Fprintf(&b, "    %-12s  %s\n", "engine", variant)
	layers := fmt.Sprintf("%d", i.GPULayers)
	if i.GPULayers >= maxGPULayers {
		layers = "all"
	}
	fmt.Fprintf(&b, "    %-12s  %s\n", "gpu_layers", layers)
	return b.String()
}

func boolLabel(v bool, yes, no string) string {
	if v {
		return yes
	}
	return no
}
