package engine

import (
	"fmt"
	"strings"

	"atlas.llm/internal/config"
)

// resolveCtxSize returns the context size to start llama-server with,
// clamped to what the current model was trained for when that is known.
func ResolveCtxSize(cfg config.Config) int {
	n := cfg.CtxSize
	if n <= 0 {
		n = config.DefaultCtxSize
	}
	if trained := CurrentModelTrainedContext(); trained > 0 && n > trained {
		n = trained
	}
	if n > config.MaxConfigurableCtx {
		n = config.MaxConfigurableCtx
	}
	if n < config.MinConfigurableCtx {
		n = config.MinConfigurableCtx
	}
	return n
}

// maxTokensCeiling is the largest reply length that still leaves room for
// the prompt and history. Derived from the context window rather than fixed,
// so raising ctx_size raises this too.
func MaxTokensCeiling(cfg config.Config) int {
	n := effectiveCtxSize(cfg) * 3 / 4
	if n < 512 {
		n = 512
	}
	return n
}

// ctxSizeDisplay renders the setting for `/set`, including the model's
// trained ceiling so the headroom is visible.
func CtxSizeDisplay(cfg config.Config) string {
	eff := ResolveCtxSize(cfg)
	set := "auto"
	if cfg.CtxSize > 0 {
		set = fmt.Sprintf("%d", cfg.CtxSize)
	}
	trained := CurrentModelTrainedContext()
	if trained == 0 {
		return fmt.Sprintf("%s (using %d)", set, eff)
	}
	return fmt.Sprintf("%s (using %d; this model was trained for %d)", set, eff, trained)
}

// endpointDisplay renders the endpoint setting, naming the machine doing the
// work so "local" and "remote" are never ambiguous in /set output.
func EndpointDisplay(cfg config.Config) string {
	ep, err := config.NormalizeEndpoint(cfg.Endpoint)
	if err != nil {
		return "invalid (" + strings.TrimSpace(cfg.Endpoint) + ")"
	}
	if ep == "" {
		return "local (inference runs on this machine)"
	}
	return ep
}

// effectiveCtxSize is the context the next request will actually get.
//
// In remote mode the server fixed it at spawn, and this install's ctx_size
// says nothing about it — a client at the default 16384 talking to a server
// serving 8192 would compute a max_tokens ceiling larger than the server's
// entire context, accept it, and fail at request time.
func effectiveCtxSize(cfg config.Config) int {
	if ep, _ := config.RemoteEndpoint(); ep != "" {
		if st := GetRemoteStatus(); st.HaveInfo && st.Info.CtxPerSlot > 0 {
			return st.Info.CtxPerSlot
		}
		// Connected to something that doesn't report its context. Nothing
		// better than the local value is available, so the ceiling stays
		// advisory rather than accurate.
	}
	return ResolveCtxSize(cfg)
}
