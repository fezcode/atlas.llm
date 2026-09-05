package tools

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"mime"
	"net"
	"net/http"
	"net/url"
	"strings"
	"syscall"
	"time"

	"golang.org/x/net/html/charset"

	"atlas.llm/internal/buildinfo"
)

// web_fetch: read one web page.
//
// Deliberately not a search tool. Searching means either scraping an engine's
// HTML (unofficial, and it breaks silently) or an API key setting; /mcp add
// duckduckgo already covers it for anyone who wants it. This reads a URL the
// model was given or found in a page it already read, which is the half MCP
// fetch servers do worst and the half that needs no setup at all.

const (
	// webFetchTimeout bounds the whole request. Shorter than run_cmd's 30s:
	// a page that has not answered in twenty seconds is not going to.
	webFetchTimeout = 20 * time.Second

	// webFetchMaxDownload caps bytes read off the wire. Distinct from
	// toolResultSizeLimit, which caps what reaches the model — without this
	// a 500MB file would be pulled into memory on its way to being truncated
	// to 6KB anyway.
	webFetchMaxDownload = 2 << 20

	webFetchMaxRedirects = 5
)

// ipGuard decides whether a connection to a resolved address may proceed.
type ipGuard func(net.IP) error

var errNoAddress = errors.New("could not determine the address being connected to")

// requirePublicIP refuses anything that is not on the public internet.
//
// The argument to web_fetch comes from a model reading text it did not
// write, so "fetch this page" is one hop away from "connect to something on
// the user's network". Blocking the ranges here means a page cannot use the
// tool to reach a router admin panel, a cloud metadata endpoint, or the
// llama-server on the next desk.
func requirePublicIP(ip net.IP) error {
	if ip == nil {
		return errNoAddress
	}
	reason := ""
	switch {
	case ip.IsLoopback():
		reason = "loopback"
	case ip.IsPrivate():
		reason = "a private network"
	case ip.IsLinkLocalUnicast(), ip.IsLinkLocalMulticast():
		reason = "link-local"
	case ip.IsUnspecified():
		reason = "unspecified"
	case ip.IsMulticast():
		reason = "multicast"
	case isCGNAT(ip):
		reason = "carrier-grade NAT"
	default:
		return nil
	}
	return fmt.Errorf("refusing to connect to %s: web_fetch reaches the public "+
		"internet only, and that address is %s", ip, reason)
}

// isCGNAT reports whether ip is in 100.64.0.0/10, which net.IP.IsPrivate
// does not cover and which carries plenty of reachable infrastructure.
func isCGNAT(ip net.IP) bool {
	v4 := ip.To4()
	return v4 != nil && v4[0] == 100 && v4[1] >= 64 && v4[1] <= 127
}

// allowAnyIP is the guard the content tests use. httptest servers bind
// loopback, so exercising a successful fetch means standing the real guard
// down for that client — better than a flag in the shipping path.
func allowAnyIP(net.IP) error { return nil }

// newWebFetchClient builds the HTTP client, enforcing guard at dial time.
//
// The guard hangs off Dialer.Control rather than a pre-flight DNS lookup
// because Control runs after resolution with the concrete IP in hand, right
// before connect. That closes two holes at once: there is no second lookup
// that could answer differently from the one that was checked, and every
// redirect hop dials again, so every hop is guarded with no CheckRedirect
// logic of its own.
//
// No proxy, deliberately. Through a proxy the guard would be inspecting the
// proxy's address instead of the target's — and since proxies usually sit on
// the LAN, honouring HTTP_PROXY would mean the guard rejected every request.
func newWebFetchClient(guard ipGuard) *http.Client {
	dialer := &net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}
	dialer.Control = func(_, address string, _ syscall.RawConn) error {
		host, _, err := net.SplitHostPort(address)
		if err != nil {
			return errNoAddress
		}
		return guard(net.ParseIP(host))
	}
	return &http.Client{
		Timeout: webFetchTimeout,
		Transport: &http.Transport{
			Proxy:                 nil,
			DialContext:           dialer.DialContext,
			TLSHandshakeTimeout:   10 * time.Second,
			ResponseHeaderTimeout: 15 * time.Second,
			DisableCompression:    false,
			MaxIdleConns:          4,
		},
		CheckRedirect: func(_ *http.Request, via []*http.Request) error {
			if len(via) >= webFetchMaxRedirects {
				return fmt.Errorf("gave up after %d redirects", webFetchMaxRedirects)
			}
			return nil
		},
	}
}

func toolWebFetch(args map[string]any) (string, error) {
	raw, err := ArgString(args, "url", true)
	if err != nil {
		return "", err
	}
	return fetchURL(newWebFetchClient(requirePublicIP), raw)
}

// fetchURL is the tool's body with the client injected, so tests can supply
// one whose guard permits the loopback address httptest binds.
func fetchURL(client *http.Client, raw string) (string, error) {
	u, err := parseFetchURL(raw)
	if err != nil {
		return "", err
	}
	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "atlas.llm/"+buildinfo.Version+" (+https://github.com/fezcode/atlas.llm)")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,text/plain;q=0.9,application/json;q=0.9,*/*;q=0.1")

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return "", fmt.Errorf("HTTP %s from %s", resp.Status, resp.Request.URL)
	}

	data, err := io.ReadAll(io.LimitReader(resp.Body, webFetchMaxDownload))
	if err != nil {
		return "", fmt.Errorf("reading the response failed: %w", err)
	}

	ctype := resp.Header.Get("Content-Type")
	mediaType, _, _ := mime.ParseMediaType(ctype)
	mediaType = strings.ToLower(strings.TrimSpace(mediaType))

	final := resp.Request.URL
	var title, body string
	switch {
	case mediaType == "text/html" || mediaType == "application/xhtml+xml":
		// charset.NewReader honours the header, then the document's own
		// meta charset. Without it a windows-1252 page arrives as mojibake
		// and every quotation mark in it is noise.
		r, err := charset.NewReader(bytes.NewReader(data), ctype)
		if err != nil {
			r = bytes.NewReader(data)
		}
		title, body, err = htmlToText(r, final)
		if err != nil {
			return "", fmt.Errorf("could not parse the page: %w", err)
		}
	case isTextMediaType(mediaType):
		body = string(data)
	case mediaType == "" && isLikelyText(data):
		// No Content-Type at all. Plenty of small servers do this; if the
		// bytes look like text, treat them as text.
		body = string(data)
	default:
		return "", fmt.Errorf("web_fetch reads text pages; %s is not a text format "+
			"(from %s)", mediaType, final)
	}

	if strings.TrimSpace(body) == "" {
		return "", fmt.Errorf("no readable text at %s — the page may render its "+
			"content with JavaScript, which this tool does not run", final)
	}
	return TruncateForModel(assembleFetchResult(title, final.String(), body)), nil
}

// assembleFetchResult puts the title and the URL actually fetched above the
// content. The URL matters because it is the final one: a redirect that
// changed where the content came from should not be silent.
func assembleFetchResult(title, finalURL, body string) string {
	var b strings.Builder
	head := "# " + title
	if title != "" && !strings.HasPrefix(strings.TrimSpace(body), head) {
		b.WriteString(head + "\n")
	}
	b.WriteString("url: " + finalURL + "\n\n")
	b.WriteString(strings.TrimSpace(body))
	return b.String()
}

// isTextMediaType covers the formats worth handing over verbatim. Anything
// else is refused by name rather than dumped as bytes.
func isTextMediaType(mediaType string) bool {
	if strings.HasPrefix(mediaType, "text/") {
		return true
	}
	switch mediaType {
	case "application/json", "application/xml", "application/javascript",
		"application/x-yaml", "application/yaml", "application/toml":
		return true
	}
	// application/ld+json, application/vnd.api+json, image/svg+xml...
	return strings.HasSuffix(mediaType, "+json") || strings.HasSuffix(mediaType, "+xml")
}

// parseFetchURL validates the URL and supplies https:// when the model omits
// the scheme, which they routinely do when quoting a URL out of prose.
func parseFetchURL(raw string) (*url.URL, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, errors.New("url is empty")
	}
	u, err := url.Parse(raw)
	if err != nil {
		return nil, fmt.Errorf("%q is not a valid URL: %w", raw, err)
	}
	if u.Scheme == "" {
		if u, err = url.Parse("https://" + raw); err != nil {
			return nil, fmt.Errorf("%q is not a valid URL: %w", raw, err)
		}
	}
	switch strings.ToLower(u.Scheme) {
	case "http", "https":
	default:
		return nil, fmt.Errorf("web_fetch supports http and https only, not %q — "+
			"use read_file for local files", u.Scheme)
	}
	if u.Host == "" {
		return nil, fmt.Errorf("%q has no host", raw)
	}
	return u, nil
}
