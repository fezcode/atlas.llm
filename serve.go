package main

import (
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"sort"
	"syscall"
	"time"
)

// defaultServeSlots is wider than the TUI's default because a served engine
// answers several clients. llama-server routes each request to the idle slot
// with the longest matching prefix, so one slot per client keeps each
// conversation's KV cache intact; with fewer slots than clients they evict
// each other every turn and all of them pay a full re-prefill.
//
// The KV budget does not grow with the slot count — see launchLlamaServerOn.
// Four slots split the same memory four ways, so each client gets less
// context rather than the server needing more VRAM.
const defaultServeSlots = 4

// runServe hosts this machine's model on the network and blocks until
// interrupted. Inference is local by definition here, so a client endpoint
// configured on the same install is a contradiction worth refusing.
func runServe(opts serveOptions) error {
	if ep, _ := remoteEndpoint(); ep != "" {
		return fmt.Errorf("this install is configured as a client of %s — "+
			"run `atlas.llm` and `/set endpoint local` before serving", ep)
	}
	if _, err := requireEngine(); err != nil {
		return err
	}
	m, err := currentModel()
	if err != nil {
		return err
	}
	if _, err := requireModel(m); err != nil {
		return err
	}

	fmt.Printf("Starting %s on %s:%d — this can take a moment while the model loads.\n",
		m.Name, opts.bindHost(), opts.Port)

	started := time.Now()
	s, err := startLlamaServerWith(m, opts)
	if err != nil {
		return err
	}
	defer s.stopLocked()

	build := ""
	if p, ok := s.fetchProps(); ok {
		build = p.BuildInfo
	}
	infoPort := infoPortFor(opts.Port)
	info, infoErr := startInfoServer(opts.bindHost(), infoPort, func() atlasServerInfo {
		return atlasServerInfo{
			Service:       "atlas.llm",
			Version:       Version,
			Model:         m.Name,
			ModelFile:     m.Filename,
			EngineVariant: installedEngineVariant(),
			GPULayers:     s.gpuLayer,
			CtxPerSlot:    s.ctxN,
			Slots:         s.slots,
			InferencePort: opts.Port,
			AuthRequired:  opts.APIKey != "",
			UptimeSeconds: int64(time.Since(started).Seconds()),
			LlamaBuild:    build,
		}
	})
	if infoErr != nil {
		// Not fatal: inference is the job. Clients just fall back to what
		// llama-server's own /props reports.
		log.Printf("info endpoint unavailable on :%d (%v) — clients will see less detail", infoPort, infoErr)
	} else {
		defer func() { _ = info.Close() }()
	}
	printServeBanner(s, opts, infoErr == nil)

	// Ctrl+C is the documented way to stop this, so handle it rather than
	// letting the runtime kill us with the engine subprocess still alive.
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, os.Interrupt, syscall.SIGTERM)
	<-sigs
	fmt.Println("\nStopping.")
	return nil
}

func printServeBanner(s *llamaServer, opts serveOptions, infoUp bool) {
	fmt.Printf("\n  atlas.llm v%s serving %s\n", Version, s.model.Name)
	fmt.Printf("  %d slots · %d tokens of context each · %d layers on GPU\n",
		s.slots, s.ctxN, s.gpuLayer)
	if infoUp {
		fmt.Printf("  inference :%d · info :%d\n\n", opts.Port, infoPortFor(opts.Port))
	} else {
		fmt.Printf("  inference :%d · info endpoint unavailable\n\n", opts.Port)
	}

	addrs := clientURLs(opts)
	if len(addrs) == 0 {
		fmt.Printf("  No LAN address found. Clients on this machine can use:\n")
		fmt.Printf("    /set endpoint http://127.0.0.1:%d\n\n", opts.Port)
	} else {
		fmt.Printf("  On another machine, run atlas.llm and set:\n")
		for _, a := range addrs {
			fmt.Printf("    /set endpoint %s\n", a)
		}
		fmt.Println()
	}
	if opts.APIKey != "" {
		fmt.Printf("  Clients also need:  /set endpoint_key %s\n\n", opts.APIKey)
	} else if opts.bindHost() != "127.0.0.1" {
		fmt.Printf("  No API key set — anyone who can reach this port can use this model.\n\n")
	}
	fmt.Println("  Ctrl+C to stop.")
}

// clientURLs turns the bind address into addresses a client can actually
// paste. A wildcard bind is not itself routable, so it expands to this
// machine's real interface addresses.
func clientURLs(opts serveOptions) []string {
	host := opts.bindHost()
	if host != "0.0.0.0" && host != "::" && host != "[::]" {
		return []string{fmt.Sprintf("http://%s:%d", host, opts.Port)}
	}
	var out []string
	for _, ip := range lanAddresses() {
		out = append(out, fmt.Sprintf("http://%s:%d", ip, opts.Port))
	}
	return out
}

// lanAddresses lists this machine's non-loopback IPv4 addresses. IPv6 is
// omitted deliberately: link-local addresses need a zone index to be usable
// and would be more confusing than helpful in a "paste this" list.
func lanAddresses() []string {
	ifaces, err := net.Interfaces()
	if err != nil {
		return nil
	}
	var out []string
	for _, iface := range ifaces {
		if iface.Flags&net.FlagUp == 0 || iface.Flags&net.FlagLoopback != 0 {
			continue
		}
		addrs, err := iface.Addrs()
		if err != nil {
			continue
		}
		for _, addr := range addrs {
			var ip net.IP
			switch v := addr.(type) {
			case *net.IPNet:
				ip = v.IP
			case *net.IPAddr:
				ip = v.IP
			}
			if ip == nil || ip.IsLoopback() || ip.IsLinkLocalUnicast() {
				continue
			}
			if v4 := ip.To4(); v4 != nil {
				out = append(out, v4.String())
			}
		}
	}
	sort.Strings(out)
	return out
}
