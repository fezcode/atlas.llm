package mcpx

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/modelcontextprotocol/go-sdk/auth"
	"github.com/modelcontextprotocol/go-sdk/oauthex"
	"golang.org/x/oauth2"

	"atlas.llm/internal/engine"
)

// headerTransport injects static headers (API keys, tenant ids) into every
// request to a remote MCP server.
type headerTransport struct {
	base    http.RoundTripper
	headers map[string]string
}

func (t *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// RoundTrippers must not mutate the request they're given.
	clone := req.Clone(req.Context())
	for k, v := range t.headers {
		clone.Header.Set(k, v)
	}
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	return base.RoundTrip(clone)
}

// mcpHTTPClient builds the HTTP client used for a remote MCP server. The
// long timeout accommodates tool calls that do real work server-side.
func mcpHTTPClient(headers map[string]string) *http.Client {
	c := &http.Client{Timeout: mcpCallTimeout}
	if len(headers) > 0 {
		c.Transport = &headerTransport{headers: headers}
	}
	return c
}

// loopbackReceiver owns the 127.0.0.1 listener that catches the OAuth
// redirect. The listener is bound once — before the flow starts — because
// the redirect URI must be known when the client registers and when the
// authorization URL is built.
type loopbackReceiver struct {
	ln       net.Listener
	redirect string

	mu      sync.Mutex
	waiting chan *auth.AuthorizationResult
}

func newLoopbackReceiver() (*loopbackReceiver, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("bind callback listener: %w", err)
	}
	r := &loopbackReceiver{
		ln:       ln,
		redirect: fmt.Sprintf("http://127.0.0.1:%d/callback", ln.Addr().(*net.TCPAddr).Port),
	}
	srv := &http.Server{Handler: http.HandlerFunc(r.serveCallback)}
	go func() {
		if err := srv.Serve(ln); err != nil && err != http.ErrServerClosed {
			log.Printf("mcp oauth: callback listener stopped: %v", err)
		}
	}()
	return r, nil
}

func (r *loopbackReceiver) serveCallback(w http.ResponseWriter, req *http.Request) {
	q := req.URL.Query()
	w.Header().Set("Content-Type", "text/html; charset=utf-8")

	if errCode := q.Get("error"); errCode != "" {
		msg := errCode
		if d := q.Get("error_description"); d != "" {
			msg += ": " + d
		}
		fmt.Fprintf(w, oauthResultPage, "Authorization failed", msg)
		r.deliver(nil)
		return
	}
	code := q.Get("code")
	if code == "" {
		fmt.Fprintf(w, oauthResultPage, "Authorization failed", "no authorization code in the redirect")
		r.deliver(nil)
		return
	}
	fmt.Fprintf(w, oauthResultPage, "Authorized", "You can close this tab and return to atlas.llm.")
	r.deliver(&auth.AuthorizationResult{
		Code:  code,
		State: q.Get("state"),
		Iss:   q.Get("iss"),
	})
}

// deliver hands the result to whichever fetch call is currently waiting.
// A callback with no waiter (stale tab, refresh) is dropped.
func (r *loopbackReceiver) deliver(res *auth.AuthorizationResult) {
	r.mu.Lock()
	ch := r.waiting
	r.waiting = nil
	r.mu.Unlock()
	if ch != nil {
		ch <- res
	}
}

// fetch opens the browser at the authorization URL and blocks until the
// redirect lands on the loopback listener.
func (r *loopbackReceiver) fetch(ctx context.Context, args *auth.AuthorizationArgs) (*auth.AuthorizationResult, error) {
	ch := make(chan *auth.AuthorizationResult, 1)
	r.mu.Lock()
	r.waiting = ch
	r.mu.Unlock()

	if err := OpenBrowserHook(args.URL); err != nil {
		log.Printf("mcp oauth: could not open a browser (%v)", err)
	}
	// The URL is also logged so a headless/SSH user can paste it manually.
	log.Printf("mcp oauth: authorize at %s", args.URL)

	select {
	case res := <-ch:
		if res == nil {
			return nil, fmt.Errorf("authorization was denied or failed")
		}
		return res, nil
	case <-ctx.Done():
		r.mu.Lock()
		r.waiting = nil
		r.mu.Unlock()
		return nil, ctx.Err()
	}
}

const oauthResultPage = `<!doctype html><meta charset="utf-8">
<title>atlas.llm</title>
<style>body{font:16px -apple-system,system-ui,sans-serif;margin:15vh auto;max-width:32rem;text-align:center;color:#111}
h1{font-size:1.3rem}p{color:#555}</style>
<h1>%s</h1><p>%s</p>`

// newOAuthHandler wires the SDK's authorization-code handler to a loopback
// redirect, a browser launch, and on-disk token storage. Tokens (and the
// discovered endpoint config) are persisted so a refresh token can be
// replayed on the next run instead of re-prompting.
func newOAuthHandler(ctx context.Context, name string, cfg MCPServerConfig, httpClient *http.Client) (auth.OAuthHandler, error) {
	recv, err := newLoopbackReceiver()
	if err != nil {
		return nil, err
	}

	acc := &auth.AuthorizationCodeHandlerConfig{
		RedirectURL:              recv.redirect,
		AuthorizationCodeFetcher: recv.fetch,
		RequestRefreshToken:      true,
		Client:                   httpClient,
		NewTokenSource: func(ctx context.Context, oc *oauth2.Config, tok *oauth2.Token) (oauth2.TokenSource, error) {
			// Called right after a successful code exchange. This is the
			// only place we see the discovered endpoint config, so record it
			// alongside the token to enable refresh on the next launch.
			SaveMCPAuth(name, oc, tok)
			return &persistingTokenSource{name: name, src: oc.TokenSource(ctx, tok)}, nil
		},
	}

	if cfg.ClientID != "" {
		creds := &oauthex.ClientCredentials{ClientID: cfg.ClientID}
		if cfg.ClientSecret != "" {
			creds.ClientSecretAuth = &oauthex.ClientSecretAuth{ClientSecret: cfg.ClientSecret}
		}
		acc.PreregisteredClient = creds
	} else {
		// Hosted MCP servers (Atlassian, Linear, …) expect Dynamic Client
		// Registration rather than a client id baked into the app.
		acc.DynamicClientRegistrationConfig = &auth.DynamicClientRegistrationConfig{
			Metadata: &oauthex.ClientRegistrationMetadata{
				ClientName:              "atlas.llm",
				RedirectURIs:            []string{recv.redirect},
				GrantTypes:              []string{"authorization_code", "refresh_token"},
				ResponseTypes:           []string{"code"},
				TokenEndpointAuthMethod: "none",
				Scope:                   strings.Join(cfg.Scopes, " "),
			},
		}
	}

	// Reuse a previously stored token so a restart doesn't re-prompt.
	if src := restoreMCPTokenSource(ctx, name); src != nil {
		acc.InitialTokenSource = src
	}

	h, err := auth.NewAuthorizationCodeHandler(acc)
	if err != nil {
		return nil, err
	}
	return h, nil
}

// persistingTokenSource writes refreshed tokens back to storage so a later
// process can keep using the refresh token.
type persistingTokenSource struct {
	name string
	src  oauth2.TokenSource

	mu   sync.Mutex
	last string
}

func (p *persistingTokenSource) Token() (*oauth2.Token, error) {
	tok, err := p.src.Token()
	if err != nil {
		return nil, err
	}
	p.mu.Lock()
	changed := tok.AccessToken != p.last
	p.last = tok.AccessToken
	p.mu.Unlock()
	if changed {
		UpdateMCPToken(p.name, tok)
	}
	return tok, nil
}

// restoreMCPTokenSource rebuilds a refreshing token source from storage,
// or returns nil when there's nothing usable saved.
func restoreMCPTokenSource(ctx context.Context, name string) oauth2.TokenSource {
	sa, err := LoadMCPAuth(name)
	if err != nil || sa == nil || sa.Token == nil {
		return nil
	}
	// Without a refresh token an expired access token is dead weight —
	// fall through to a fresh authorization instead.
	if sa.Token.RefreshToken == "" && !sa.Token.Valid() {
		return nil
	}
	oc := sa.OauthConfig()
	if oc.Endpoint.TokenURL == "" {
		return nil
	}
	log.Printf("mcp oauth: reusing stored credentials for %q", name)
	return &persistingTokenSource{
		name: name,
		src:  oc.TokenSource(ctx, sa.Token),
		last: sa.Token.AccessToken,
	}
}

// openBrowserHook is indirection for tests, which need to observe the
// authorization URL without a browser actually opening.
var OpenBrowserHook = openBrowser

// openBrowser launches the platform's default browser at url.
func openBrowser(url string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", url)
	case "windows":
		// `start` is a cmd builtin; the empty string is the window title,
		// which `start` would otherwise take from a quoted URL.
		cmd = exec.Command("cmd", "/c", "start", "", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	engine.ApplyEngineSysProcAttr(cmd)
	return cmd.Start()
}

// mcpAuthTimeout bounds how long we wait for the user to finish the browser
// consent flow before giving up on a server.
const McpAuthTimeout = 5 * time.Minute

// prmTransport answers protected-resource-metadata probes locally for
// servers that don't publish any.
//
// The MCP auth flow discovers the authorization server from the resource's
// protected-resource metadata (RFC 9728), falling back to treating the MCP
// host itself as the authorization server. Atlassian publishes no such
// metadata, and the fallback then fails RFC 8414's issuer check: the
// document served at mcp.atlassian.com declares its issuer as
// cf.mcp.atlassian.com. Discovery gives up with "failed to get
// authorization server metadata".
//
// When a server config names its authorization server, we synthesize the
// metadata the server should have published. Everything after this point —
// fetching the authorization-server metadata, dynamic client registration,
// the code exchange — proceeds normally against the real endpoints.
type PrmTransport struct {
	base       http.RoundTripper
	resource   string // the MCP endpoint URL
	AuthServer string
	Host       string
}

func WithDeclaredAuthServer(c *http.Client, resourceURL, authServer string) *http.Client {
	u, err := url.Parse(resourceURL)
	if err != nil {
		return c
	}
	base := c.Transport
	if base == nil {
		base = http.DefaultTransport
	}
	clone := *c
	clone.Transport = &PrmTransport{
		base:       base,
		resource:   resourceURL,
		AuthServer: strings.TrimRight(authServer, "/"),
		Host:       u.Host,
	}
	return &clone
}

func (t *PrmTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.Host != t.Host && req.URL.Host != t.Host {
		return t.base.RoundTrip(req)
	}
	if !strings.HasPrefix(req.URL.Path, "/.well-known/oauth-protected-resource") {
		return t.base.RoundTrip(req)
	}
	// Let a real document win if the server ever starts publishing one.
	if resp, err := t.base.RoundTrip(req); err == nil && resp.StatusCode == http.StatusOK {
		return resp, nil
	} else if resp != nil {
		resp.Body.Close()
	}

	body, _ := json.Marshal(map[string]any{
		"resource":              t.resource,
		"authorization_servers": []string{t.AuthServer},
	})
	log.Printf("mcp oauth: synthesized protected-resource metadata for %s -> %s",
		t.resource, t.AuthServer)
	return &http.Response{
		Status:     "200 OK",
		StatusCode: http.StatusOK,
		Proto:      "HTTP/1.1",
		ProtoMajor: 1, ProtoMinor: 1,
		Header:  http.Header{"Content-Type": []string{"application/json"}},
		Body:    io.NopCloser(bytes.NewReader(body)),
		Request: req,
	}, nil
}
