package main

import (
	"net/url"
	"strings"
	"testing"
)

func render(t *testing.T, doc string, base string) (string, string) {
	t.Helper()
	var u *url.URL
	if base != "" {
		var err error
		if u, err = url.Parse(base); err != nil {
			t.Fatal(err)
		}
	}
	title, text, err := htmlToText(strings.NewReader(doc), u)
	if err != nil {
		t.Fatal(err)
	}
	return title, text
}

// The reason extraction exists at all: a tool result is capped at 6KB, and
// on a real documentation page the chrome is most of the bytes.
func TestChromeIsDropped(t *testing.T) {
	_, got := render(t, `<html><body>
		<script>var tracking = 1;</script>
		<style>.x{color:red}</style>
		<nav><a href="/a">Home</a><a href="/b">Docs</a></nav>
		<header>SiteName</header>
		<div role="navigation">Sidebar links</div>
		<div aria-hidden="true">Decorative</div>
		<div hidden>Collapsed panel</div>
		<div style="display:none">Hidden by style</div>
		<p>The actual sentence.</p>
		<aside>Related reading</aside>
		<footer>Copyright 2026</footer>
	</body></html>`, "")

	if !strings.Contains(got, "The actual sentence.") {
		t.Fatalf("lost the content:\n%s", got)
	}
	for _, junk := range []string{
		"tracking", "color:red", "Home", "Docs", "SiteName", "Sidebar",
		"Decorative", "Collapsed", "Hidden by style", "Related", "Copyright",
	} {
		if strings.Contains(got, junk) {
			t.Errorf("chrome survived (%q):\n%s", junk, got)
		}
	}
}

// <main> is the author saying where the content is. Believing them is worth
// more than any heuristic we could apply to the whole body.
func TestMainAndArticleWin(t *testing.T) {
	_, got := render(t, `<html><body>
		<div><p>Body level noise</p></div>
		<main><p>Main content</p></main>
	</body></html>`, "")
	if !strings.Contains(got, "Main content") || strings.Contains(got, "noise") {
		t.Errorf("main not preferred:\n%s", got)
	}

	_, got = render(t, `<html><body>
		<div><p>Body level noise</p></div>
		<article><p>Article content</p></article>
	</body></html>`, "")
	if !strings.Contains(got, "Article content") || strings.Contains(got, "noise") {
		t.Errorf("article not preferred:\n%s", got)
	}

	// With neither, the whole body is fair game.
	_, got = render(t, `<html><body><p>Just a page</p></body></html>`, "")
	if !strings.Contains(got, "Just a page") {
		t.Errorf("body fallback failed:\n%s", got)
	}
}

func TestHeadingsAndParagraphs(t *testing.T) {
	title, got := render(t, `<html><head><title>  Page   Title </title></head><body>
		<h1>Top</h1><p>One.</p><h2>Sub</h2><p>Two.</p><h3>Deeper</h3>
	</body></html>`, "")
	if title != "Page Title" {
		t.Errorf("title = %q, want %q", title, "Page Title")
	}
	for _, want := range []string{"# Top", "## Sub", "### Deeper"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q:\n%s", want, got)
		}
	}
	// Blocks are separated, so the model can tell where one ends.
	if !strings.Contains(got, "One.\n\n## Sub") {
		t.Errorf("blocks not separated:\n%s", got)
	}
}

// A list rendered as one block per item reads as a run of paragraphs. Items
// have to sit on adjacent lines to look like a list.
func TestListsAreTightAndNest(t *testing.T) {
	_, got := render(t, `<ul><li>alpha</li><li>beta<ul><li>nested</li></ul></li></ul>`, "")
	if !strings.Contains(got, "- alpha\n- beta") {
		t.Errorf("items are not adjacent:\n%s", got)
	}
	if !strings.Contains(got, "  - nested") {
		t.Errorf("nested list not indented:\n%s", got)
	}

	_, got = render(t, `<ol start="3"><li>three</li><li>four</li></ol>`, "")
	if !strings.Contains(got, "3. three\n4. four") {
		t.Errorf("ordered list numbering wrong:\n%s", got)
	}
}

func TestCodeBlocksSurvive(t *testing.T) {
	_, got := render(t, "<pre><code>func main() {\n\tprintln(\"hi\")\n}</code></pre>", "")
	if !strings.Contains(got, "```\nfunc main() {\n\tprintln(\"hi\")\n}\n```") {
		t.Errorf("code block not fenced with its line breaks intact:\n%s", got)
	}
	// Syntax highlighting wraps every token in a span; the code has to come
	// through as code regardless.
	_, got = render(t, `<pre><span class="k">go</span> <span class="n">build</span></pre>`, "")
	if !strings.Contains(got, "go build") {
		t.Errorf("highlighted code lost:\n%s", got)
	}

	_, got = render(t, `<p>Run <code>go test</code> first.</p>`, "")
	if !strings.Contains(got, "Run `go test` first.") {
		t.Errorf("inline code wrong:\n%s", got)
	}
}

// Links are the only way the model gets from one page to the next, so a
// relative href has to come out usable.
func TestLinksResolveAbsolute(t *testing.T) {
	_, got := render(t, `<p><a href="../guide/intro.html">Intro</a></p>`, "https://example.com/docs/api/index.html")
	if !strings.Contains(got, "[Intro](https://example.com/docs/guide/intro.html)") {
		t.Errorf("relative link not resolved:\n%s", got)
	}

	// <base href> overrides the fetch URL, which is the whole point of it.
	_, got = render(t, `<html><head><base href="https://cdn.example.net/v2/"></head>
		<body><p><a href="page.html">P</a></p></body></html>`, "https://example.com/docs/")
	if !strings.Contains(got, "[P](https://cdn.example.net/v2/page.html)") {
		t.Errorf("base href ignored:\n%s", got)
	}

	// In-page anchors and javascript: are not somewhere to fetch; keep the
	// words, drop the target.
	_, got = render(t, `<p><a href="#section">Jump</a> <a href="javascript:void(0)">Click</a></p>`, "https://example.com/")
	if strings.Contains(got, "](") {
		t.Errorf("non-navigable link kept its target:\n%s", got)
	}
	if !strings.Contains(got, "Jump") || !strings.Contains(got, "Click") {
		t.Errorf("link text lost:\n%s", got)
	}
}

// pkg.go.dev and every other doc generator hangs a ¶ permalink off each
// heading. Kept, it is one useless token per heading on a page that is
// mostly headings.
func TestPermalinkGlyphsAreDropped(t *testing.T) {
	_, got := render(t, `<h2>Overview <a href="#pkg-overview">¶</a></h2>
		<p>Text <a href="#x">§</a> here.</p>
		<p><a href="/real">Real link</a></p>`, "https://pkg.go.dev/net/http")
	if strings.ContainsAny(got, "¶§") {
		t.Errorf("permalink glyph survived:\n%s", got)
	}
	if !strings.Contains(got, "## Overview") {
		t.Errorf("heading damaged:\n%s", got)
	}
	if !strings.Contains(got, "[Real link](https://pkg.go.dev/real)") {
		t.Errorf("a real link was dropped with the glyphs:\n%s", got)
	}
}

func TestTablesBecomeRows(t *testing.T) {
	_, got := render(t, `<table>
		<thead><tr><th>Flag</th><th>Meaning</th></tr></thead>
		<tbody><tr><td>-v</td><td>version</td></tr><tr><td>-h</td><td>help</td></tr></tbody>
	</table>`, "")
	if !strings.Contains(got, "Flag | Meaning\n-v | version\n-h | help") {
		t.Errorf("table rows wrong:\n%s", got)
	}
}

// HTML source is indented for humans. Left alone, a page's prose arrives
// carrying the file's indentation and pays for it against the size cap.
func TestWhitespaceAndEntitiesCollapse(t *testing.T) {
	_, got := render(t, "<p>one\n\t\t   two&nbsp;three &amp; four &lt;five&gt;</p>", "")
	if got != "one two three & four <five>" {
		t.Errorf("got %q", got)
	}
}

func TestEmptyDocumentIsEmpty(t *testing.T) {
	for _, doc := range []string{"", "<html></html>", "<html><body><div><span></span></div></body></html>"} {
		if _, got := render(t, doc, ""); strings.TrimSpace(got) != "" {
			t.Errorf("doc %q produced %q", doc, got)
		}
	}
}
