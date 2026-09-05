package tools

import (
	"io"
	"net/url"
	"strconv"
	"strings"
	"unicode"

	"golang.org/x/net/html"
	"golang.org/x/net/html/atom"
)

// HTML to readable text, for web_fetch.
//
// There is no network in this file. Extraction is where the volume and the
// bugs are, so it is a pure function over a reader and testable against
// string literals — the same reason layersThatFit and serveCapacity were
// pulled out of their callers.
//
// The output is markdown-ish rather than plain text because the model reads
// it: headings tell it what part of the page it is looking at, and links let
// it fetch the next page. It is not round-trippable markdown and does not
// try to be.
//
// Quality matters more here than it looks. A tool result is capped at
// toolResultSizeLimit, so on a modern documentation page a naive tag strip
// spends the entire budget on a cookie banner and a nav menu before reaching
// a word of content.

// skipTags are elements whose subtrees never carry page content.
var skipTags = map[atom.Atom]bool{
	atom.Script: true, atom.Style: true, atom.Noscript: true,
	atom.Nav: true, atom.Header: true, atom.Footer: true, atom.Aside: true,
	atom.Form: true, atom.Svg: true, atom.Iframe: true, atom.Template: true,
	atom.Button: true, atom.Select: true, atom.Textarea: true, atom.Input: true,
	atom.Object: true, atom.Embed: true, atom.Canvas: true, atom.Audio: true,
	atom.Video: true, atom.Map: true, atom.Menu: true,
}

// skipRoles mark chrome on pages that use semantic divs instead of the
// corresponding elements, which is most sites built on a component framework.
var skipRoles = map[string]bool{
	"navigation": true, "banner": true, "contentinfo": true,
	"complementary": true, "search": true, "dialog": true, "alertdialog": true,
	"menu": true, "menubar": true, "toolbar": true, "tablist": true,
}

// htmlToText renders a page as text. base is the URL the document was
// fetched from, used to make links absolute; a <base href> in the document
// overrides it. The returned title comes from <title> and is empty when the
// document has none.
func htmlToText(r io.Reader, base *url.URL) (title, text string, err error) {
	doc, err := html.Parse(r)
	if err != nil {
		return "", "", err
	}
	title = strings.TrimSpace(collapseSpace(textContent(findTag(doc, atom.Title))))
	if href := attr(findTag(doc, atom.Base), "href"); href != "" && base != nil {
		if u, err := base.Parse(href); err == nil {
			base = u
		}
	}
	rend := &textRenderer{base: base}
	rend.walk(contentRoot(doc))
	rend.flush()
	return title, strings.Join(rend.blocks, "\n\n"), nil
}

// contentRoot picks the subtree worth reading. <main> and <article> are the
// author telling us where the content is; falling back to <body> means
// reading the whole page and relying on pruning.
func contentRoot(doc *html.Node) *html.Node {
	for _, a := range []atom.Atom{atom.Main, atom.Article, atom.Body} {
		if n := findTag(doc, a); n != nil {
			return n
		}
	}
	return doc
}

type textRenderer struct {
	base   *url.URL
	blocks []string
	line   strings.Builder // inline content of the block being built
	prefix string          // heading hashes or list marker, applied at flush
}

// flush ends the current block. Empty blocks are dropped, which is what
// keeps the wrapper divs of a component-framework page from producing
// hundreds of blank lines.
func (r *textRenderer) flush() {
	s := strings.TrimSpace(r.line.String())
	r.line.Reset()
	prefix := r.prefix
	r.prefix = ""
	if s == "" {
		return
	}
	r.blocks = append(r.blocks, prefix+s)
}

func (r *textRenderer) block(s string) {
	r.flush()
	if strings.TrimSpace(s) != "" {
		r.blocks = append(r.blocks, s)
	}
}

// writeInline appends inline content, collapsing whitespace across node
// boundaries. Trimming each text node on its own would be wrong: the space
// in "<code>go test</code> first" belongs to the text node after the tag,
// and dropping it as leading whitespace glues the words together.
func (r *textRenderer) writeInline(s string) {
	if s == "" {
		return
	}
	if s[0] == ' ' && r.endsWithSpace() {
		if s = strings.TrimLeft(s, " "); s == "" {
			return
		}
	}
	r.line.WriteString(s)
}

// endsWithSpace also reports true for an empty line, so a block never opens
// with a space that flush would only have to trim.
func (r *textRenderer) endsWithSpace() bool {
	cur := r.line.String()
	return cur == "" || cur[len(cur)-1] == ' '
}

func (r *textRenderer) children(n *html.Node) {
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		r.walk(c)
	}
}

func (r *textRenderer) walk(n *html.Node) {
	if n == nil {
		return
	}
	switch n.Type {
	case html.TextNode:
		r.writeInline(collapseSpace(n.Data))
		return
	case html.ElementNode:
	default:
		return // comments, doctype
	}
	if skipNode(n) {
		return
	}

	switch n.DataAtom {
	case atom.Br:
		r.writeInline(" ")
	case atom.Hr:
		r.block("---")
	case atom.H1, atom.H2, atom.H3, atom.H4, atom.H5, atom.H6:
		r.flush()
		level := int(n.Data[1] - '0')
		r.prefix = strings.Repeat("#", level) + " "
		r.children(n)
		r.flush()
	case atom.Pre:
		r.block(fence(textContent(n)))
	case atom.Code, atom.Kbd, atom.Samp:
		// Inline code. <pre><code> is handled by the Pre case above and
		// never reaches here.
		if s := strings.TrimSpace(collapseSpace(textContent(n))); s != "" {
			r.writeInline("`" + s + "`")
		}
	case atom.A:
		r.link(n)
	case atom.Img:
		// Alt text only, and only when it says something. Decorative images
		// carry alt="" and layout icons carry alt="icon", neither of which
		// is worth bytes the page's prose could have used.
		if alt := strings.TrimSpace(attr(n, "alt")); len(alt) > 2 {
			r.writeInline("[image: " + strings.TrimSpace(collapseSpace(alt)) + "]")
		}
	case atom.Ul, atom.Ol:
		r.block(renderList(n, r.base, 0))
	case atom.Table:
		r.block(renderTable(n, r.base))
	case atom.Li, atom.Tr, atom.Td, atom.Th:
		// Only reachable when the parent list or table element is missing
		// from the tree; treat as an ordinary block rather than dropping it.
		r.flush()
		r.children(n)
		r.flush()
	default:
		if blockTags[n.DataAtom] {
			r.flush()
			r.children(n)
			r.flush()
			return
		}
		r.children(n)
	}
}

// link emits [text](url) with the href resolved absolute.
//
// Links cost real bytes on a link-heavy page and are kept anyway: web_fetch
// does not search, so following a link is the only way the model gets from
// one page to the next.
func (r *textRenderer) link(n *html.Node) {
	text := strings.TrimSpace(collapseSpace(textContent(n)))
	if text == "" || anchorGlyph(text) {
		return
	}
	href := strings.TrimSpace(attr(n, "href"))
	abs := ""
	if href != "" && !strings.HasPrefix(href, "#") && !strings.HasPrefix(href, "javascript:") {
		if r.base != nil {
			if u, err := r.base.Parse(href); err == nil {
				abs = u.String()
			}
		} else if u, err := url.Parse(href); err == nil && u.IsAbs() {
			abs = u.String()
		}
	}
	if abs == "" {
		r.writeInline(text)
		return
	}
	r.writeInline("[" + text + "](" + abs + ")")
}

// anchorGlyph reports whether a link's text is a permalink marker rather
// than words — the ¶ and § that documentation generators hang off every
// heading. On a large reference page that is one piece of noise per
// heading, paid for out of a 6KB budget, saying nothing.
func anchorGlyph(s string) bool {
	n := 0
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return false
		}
		n++
	}
	return n > 0 && n <= 2
}

// blockTags separate paragraphs. Anything not listed here and not handled
// explicitly is treated as inline, which is the right default: an unknown
// custom element wrapping a sentence should not split it.
var blockTags = map[atom.Atom]bool{
	atom.P: true, atom.Div: true, atom.Section: true, atom.Article: true,
	atom.Main: true, atom.Blockquote: true, atom.Figure: true,
	atom.Figcaption: true, atom.Dl: true, atom.Dt: true, atom.Dd: true,
	atom.Details: true, atom.Summary: true, atom.Fieldset: true,
	atom.Address: true, atom.Caption: true, atom.Body: true,
}

// renderList produces one block for a whole list so items sit on adjacent
// lines. Emitting each item as its own block would blank-line-separate them,
// which reads as a series of paragraphs rather than a list.
func renderList(list *html.Node, base *url.URL, depth int) string {
	if depth > 4 {
		return "" // pathological nesting; the content is not worth the recursion
	}
	indent := strings.Repeat("  ", depth)
	ordered := list.DataAtom == atom.Ol
	num := 1
	if start := attr(list, "start"); ordered && start != "" {
		if v, err := strconv.Atoi(start); err == nil {
			num = v
		}
	}
	var lines []string
	for li := list.FirstChild; li != nil; li = li.NextSibling {
		if li.Type != html.ElementNode || li.DataAtom != atom.Li || skipNode(li) {
			continue
		}
		marker := "- "
		if ordered {
			marker = strconv.Itoa(num) + ". "
			num++
		}
		sub := &textRenderer{base: base}
		for c := li.FirstChild; c != nil; c = c.NextSibling {
			if c.Type == html.ElementNode && (c.DataAtom == atom.Ul || c.DataAtom == atom.Ol) {
				sub.flush()
				if nested := renderList(c, base, depth+1); nested != "" {
					sub.blocks = append(sub.blocks, nested)
				}
				continue
			}
			sub.walk(c)
		}
		sub.flush()
		item := strings.Join(sub.blocks, "\n")
		if strings.TrimSpace(item) == "" {
			continue
		}
		for i, l := range strings.Split(item, "\n") {
			if i == 0 {
				lines = append(lines, indent+marker+l)
			} else if strings.HasPrefix(l, "  ") {
				lines = append(lines, indent+l) // already-indented nested list
			} else {
				lines = append(lines, indent+strings.Repeat(" ", len(marker))+l)
			}
		}
	}
	return strings.Join(lines, "\n")
}

// renderTable joins cells with " | ", one line per row. Not a markdown table
// — no alignment row — because the model needs the values, not a renderable
// grid, and the separator row costs bytes on every table.
func renderTable(table *html.Node, base *url.URL) string {
	var lines []string
	var walkRows func(n *html.Node)
	walkRows = func(n *html.Node) {
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			if c.Type != html.ElementNode || skipNode(c) {
				continue
			}
			if c.DataAtom == atom.Tr {
				var cells []string
				for cell := c.FirstChild; cell != nil; cell = cell.NextSibling {
					if cell.Type != html.ElementNode {
						continue
					}
					if cell.DataAtom != atom.Td && cell.DataAtom != atom.Th {
						continue
					}
					sub := &textRenderer{base: base}
					sub.children(cell)
					sub.flush()
					cells = append(cells, strings.TrimSpace(strings.Join(sub.blocks, " ")))
				}
				if len(cells) > 0 && strings.TrimSpace(strings.Join(cells, "")) != "" {
					lines = append(lines, strings.Join(cells, " | "))
				}
				continue
			}
			walkRows(c) // thead / tbody / tfoot
		}
	}
	walkRows(table)
	return strings.Join(lines, "\n")
}

// fence wraps preformatted text in a code fence, keeping its line breaks.
func fence(code string) string {
	code = strings.Trim(strings.ReplaceAll(code, "\r\n", "\n"), "\n")
	if code == "" {
		return ""
	}
	// A fence inside the code would end the block early.
	delim := "```"
	for strings.Contains(code, delim) {
		delim += "`"
	}
	return delim + "\n" + code + "\n" + delim
}

// skipNode reports whether an element's subtree should be dropped: chrome by
// tag, chrome by ARIA role, and anything the page has hidden.
func skipNode(n *html.Node) bool {
	if skipTags[n.DataAtom] {
		return true
	}
	for _, a := range n.Attr {
		switch strings.ToLower(a.Key) {
		case "hidden":
			return true
		case "aria-hidden":
			if strings.EqualFold(a.Val, "true") {
				return true
			}
		case "role":
			if skipRoles[strings.ToLower(strings.TrimSpace(a.Val))] {
				return true
			}
		case "style":
			v := strings.ToLower(a.Val)
			if strings.Contains(v, "display:none") || strings.Contains(v, "display: none") {
				return true
			}
		}
	}
	return false
}

// findTag returns the first element with the given tag, depth-first.
func findTag(n *html.Node, a atom.Atom) *html.Node {
	if n == nil {
		return nil
	}
	if n.Type == html.ElementNode && n.DataAtom == a {
		return n
	}
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		if found := findTag(c, a); found != nil {
			return found
		}
	}
	return nil
}

func attr(n *html.Node, key string) string {
	if n == nil {
		return ""
	}
	for _, a := range n.Attr {
		if strings.EqualFold(a.Key, key) {
			return a.Val
		}
	}
	return ""
}

// textContent concatenates a subtree's text, skipping chrome so a <pre> full
// of syntax-highlighting spans still comes out as code.
func textContent(n *html.Node) string {
	if n == nil {
		return ""
	}
	var b strings.Builder
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.TextNode {
			b.WriteString(n.Data)
			return
		}
		if n.Type == html.ElementNode && skipNode(n) {
			return
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(n)
	return b.String()
}

// collapseSpace turns every run of whitespace into one space. HTML source is
// indented for humans; without this, a page's prose arrives with the file's
// indentation embedded in it and pays for it against the size cap.
//
// Leading and trailing spaces are kept — they carry meaning between inline
// elements. writeInline drops the redundant ones and flush trims the block.
func collapseSpace(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	space := false
	for _, r := range s {
		// The last case below is U+00A0. Pages emit &nbsp; for layout, and
		// left alone it survives every trim and reads as a word character.
		switch r {
		case ' ', '\t', '\n', '\r', '\f', '\v', ' ':
			space = true
		default:
			if space {
				b.WriteByte(' ')
			}
			space = false
			b.WriteRune(r)
		}
	}
	if space {
		b.WriteByte(' ')
	}
	return b.String()
}
