package tools

import (
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// permissiveClient is how the content tests reach an httptest server:
// httptest binds loopback, which the real guard exists to refuse.
func permissiveClient() *http.Client { return newWebFetchClient(allowAnyIP) }

func htmlServer(t *testing.T, ctype, body string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if ctype != "" {
			w.Header().Set("Content-Type", ctype)
		}
		fmt.Fprint(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestRequirePublicIP(t *testing.T) {
	blocked := []string{
		"127.0.0.1", "127.0.0.53", "::1",
		"10.0.0.1", "192.168.1.50", "172.16.0.1", "fd00::1",
		"169.254.169.254", // cloud metadata, the classic target
		"0.0.0.0", "::", "224.0.0.1", "100.64.0.1", "100.127.255.254",
		"::ffff:127.0.0.1", // IPv4-mapped loopback
	}
	for _, s := range blocked {
		if err := requirePublicIP(net.ParseIP(s)); err == nil {
			t.Errorf("%s was allowed", s)
		}
	}
	allowed := []string{"93.184.216.34", "8.8.8.8", "2606:2800:220:1:248:1893:25c8:1946", "100.128.0.1", "99.255.255.255"}
	for _, s := range allowed {
		if err := requirePublicIP(net.ParseIP(s)); err != nil {
			t.Errorf("%s was blocked: %v", s, err)
		}
	}
	if err := requirePublicIP(nil); err == nil {
		t.Error("an unparseable address must not be allowed through")
	}
}

// Proves the guard is actually wired into the client, not merely written.
// httptest binds 127.0.0.1, so the only local server a test has is exactly
// the thing the guard must refuse.
func TestFetchRefusesLoopback(t *testing.T) {
	srv := htmlServer(t, "text/html", "<html><body><p>secret intranet</p></body></html>")

	out, err := fetchURL(newWebFetchClient(requirePublicIP), srv.URL)
	if err == nil {
		t.Fatalf("loopback fetch succeeded:\n%s", out)
	}
	if !strings.Contains(err.Error(), "127.0.0.1") || !strings.Contains(err.Error(), "loopback") {
		t.Errorf("error does not explain the refusal: %v", err)
	}
	if strings.Contains(fmt.Sprint(out), "secret") {
		t.Error("content leaked despite the refusal")
	}
}

// The reason the guard hangs off Dialer.Control rather than a pre-flight
// lookup: every redirect hop dials again, so every hop is checked. A public
// page must not be able to bounce the fetch onto something internal.
func TestGuardRunsOnEveryRedirectHop(t *testing.T) {
	target := htmlServer(t, "text/html", "<html><body><p>internal</p></body></html>")

	var mu sync.Mutex
	dials := 0
	countingGuard := func(net.IP) error {
		mu.Lock()
		defer mu.Unlock()
		dials++
		if dials > 1 {
			return fmt.Errorf("blocked hop %d", dials)
		}
		return nil
	}

	first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusFound)
	}))
	defer first.Close()

	out, err := fetchURL(newWebFetchClient(countingGuard), first.URL)
	if err == nil {
		t.Fatalf("redirect hop was not guarded:\n%s", out)
	}
	if !strings.Contains(err.Error(), "blocked hop 2") {
		t.Errorf("failed for the wrong reason: %v", err)
	}
	if dials < 2 {
		t.Errorf("the redirect never re-dialled (%d dials), so this proves nothing", dials)
	}
}

func TestFetchHTMLPage(t *testing.T) {
	srv := htmlServer(t, "text/html; charset=utf-8", `<html>
		<head><title>Widget API</title></head>
		<body><nav>menu</nav><main><h1>Widget API</h1><p>Call it twice.</p>
		<p><a href="/next">Next page</a></p></main></body></html>`)

	out, err := fetchURL(permissiveClient(), srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "url: "+srv.URL) {
		t.Errorf("result does not name the URL fetched:\n%s", out)
	}
	if !strings.Contains(out, "# Widget API") || !strings.Contains(out, "Call it twice.") {
		t.Errorf("content missing:\n%s", out)
	}
	if strings.Contains(out, "menu") {
		t.Errorf("nav survived:\n%s", out)
	}
	// The link has to come back absolute or the model cannot follow it.
	if !strings.Contains(out, "[Next page]("+srv.URL+"/next)") {
		t.Errorf("link not absolute:\n%s", out)
	}
	// The title is already the first heading; printing it twice wastes the
	// budget and reads like a duplicate.
	if strings.Count(out, "Widget API") > 2 {
		t.Errorf("title duplicated:\n%s", out)
	}
}

// A redirect that changed where the content came from must not be silent.
func TestFetchReportsFinalURL(t *testing.T) {
	target := htmlServer(t, "text/html", "<html><body><p>moved here</p></body></html>")
	first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL+"/final", http.StatusMovedPermanently)
	}))
	defer first.Close()

	out, err := fetchURL(permissiveClient(), first.URL)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "url: "+target.URL+"/final") {
		t.Errorf("final URL not reported:\n%s", out)
	}
}

func TestFetchNonOKStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "gone", http.StatusNotFound)
	}))
	defer srv.Close()

	_, err := fetchURL(permissiveClient(), srv.URL)
	if err == nil || !strings.Contains(err.Error(), "404") {
		t.Errorf("404 not reported usefully: %v", err)
	}
}

func TestFetchTextFormats(t *testing.T) {
	for _, tc := range []struct{ ctype, body, want string }{
		{"text/plain", "just words", "just words"},
		{"text/markdown", "# Heading\n\ntext", "# Heading"},
		{"application/json", `{"ok":true}`, `{"ok":true}`},
		{"application/vnd.api+json", `{"data":1}`, `{"data":1}`},
		{"", "no content type at all", "no content type"},
	} {
		srv := htmlServer(t, tc.ctype, tc.body)
		out, err := fetchURL(permissiveClient(), srv.URL)
		if err != nil {
			t.Errorf("%s: %v", tc.ctype, err)
			continue
		}
		if !strings.Contains(out, tc.want) {
			t.Errorf("%s: got %q, want it to contain %q", tc.ctype, out, tc.want)
		}
	}
}

// Refusing by name beats dumping bytes: the model can pick a different URL,
// but it cannot do anything with a PDF rendered as mojibake.
func TestFetchRefusesBinaryFormats(t *testing.T) {
	for _, ctype := range []string{"application/pdf", "image/png", "application/octet-stream", "font/woff2"} {
		srv := htmlServer(t, ctype, "\x00\x01binary")
		_, err := fetchURL(permissiveClient(), srv.URL)
		if err == nil {
			t.Errorf("%s was accepted", ctype)
			continue
		}
		if !strings.Contains(err.Error(), ctype) {
			t.Errorf("error does not name the type: %v", err)
		}
	}
}

// A page whose content is assembled by JavaScript extracts to nothing. Say
// why, or the model retries the same URL until it runs out of rounds.
func TestFetchExplainsEmptyPage(t *testing.T) {
	srv := htmlServer(t, "text/html", `<html><head><title>App</title></head>
		<body><div id="root"></div><script>render()</script></body></html>`)
	_, err := fetchURL(permissiveClient(), srv.URL)
	if err == nil {
		t.Fatal("expected an error for a page with no text")
	}
	if !strings.Contains(err.Error(), "JavaScript") {
		t.Errorf("error does not explain why it is empty: %v", err)
	}
}

// Without charset decoding, every quotation mark and accent on a legacy page
// arrives as mojibake and spends bytes saying nothing.
func TestFetchDecodesLegacyCharset(t *testing.T) {
	// 0xE9 is é in windows-1252 and invalid UTF-8.
	body := "<html><body><p>caf\xe9 cr\xe8me</p></body></html>"
	srv := htmlServer(t, "text/html; charset=windows-1252", body)

	out, err := fetchURL(permissiveClient(), srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "café crème") {
		t.Errorf("charset not decoded: %q", out)
	}
}

// webFetchMaxDownload guards memory, which the size of the returned string
// cannot show: truncateForModel would cut a 2MB page and a 200MB one to the
// same 6KB. Padding the document with a comment makes the cap observable —
// the marker after it only survives if the whole body was read.
func TestFetchCapsTheDownload(t *testing.T) {
	// Fixed, deliberately not derived from webFetchMaxDownload: sizing the
	// document from the constant under test makes the assertion tautological,
	// since raising the cap would grow the document with it.
	const docBytes = 3 << 20
	if webFetchMaxDownload >= docBytes {
		t.Fatalf("webFetchMaxDownload is %d, at or above this test's %d-byte document — "+
			"it can no longer tell a capped read from a complete one", webFetchMaxDownload, docBytes)
	}
	srv := htmlServer(t, "text/html",
		"<html><body><!-- "+strings.Repeat("x", docBytes)+" --><p>TAIL-MARKER</p></body></html>")

	out, err := fetchURL(permissiveClient(), srv.URL)
	if err == nil && strings.Contains(out, "TAIL-MARKER") {
		t.Error("read past the download cap")
	}
}

func TestParseFetchURL(t *testing.T) {
	// A model quoting a URL out of prose routinely drops the scheme.
	u, err := parseFetchURL("example.com/docs")
	if err != nil {
		t.Fatalf("bare host rejected: %v", err)
	}
	if u.String() != "https://example.com/docs" {
		t.Errorf("got %q, want https:// assumed", u)
	}

	for _, bad := range []string{"", "   ", "file:///etc/passwd", "ftp://host/x", "data:text/html,<b>x"} {
		if got, err := parseFetchURL(bad); err == nil {
			t.Errorf("parseFetchURL(%q) = %q, want a refusal", bad, got)
		}
	}
	// The refusal has to point somewhere useful.
	_, err = parseFetchURL("file:///etc/passwd")
	if err == nil || !strings.Contains(err.Error(), "read_file") {
		t.Errorf("file:// error should point at read_file: %v", err)
	}
}

// Identifying the client is basic manners, and it is what lets an operator
// tell atlas.llm apart from a scraper in their logs.
func TestFetchSendsUserAgent(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Get("User-Agent")
		w.Header().Set("Content-Type", "text/plain")
		fmt.Fprint(w, "ok")
	}))
	defer srv.Close()

	if _, err := fetchURL(permissiveClient(), srv.URL); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(got, "atlas.llm/") {
		t.Errorf("User-Agent = %q", got)
	}
}

func TestWebFetchIsRegisteredAndConfirmed(t *testing.T) {
	tool, ok := ToolRegistry["web_fetch"]
	if !ok {
		t.Fatal("web_fetch is not in the registry")
	}
	if !tool.Destructive {
		t.Error("web_fetch must route through the confirm modal — it reaches the network")
	}
	// The model needs to know it cannot reach localhost before it tries.
	if !strings.Contains(tool.Description, "Public internet only") {
		t.Errorf("description does not mention the network restriction: %q", tool.Description)
	}
}
