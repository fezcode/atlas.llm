package browser

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"atlas.llm/internal/tools"
)

// The second wave of browser tools: screenshots, tab management, and file
// upload. Like the first wave they are built on the browserSession interface,
// with the per-protocol work (capture, target switching, DOM file attach)
// implemented in browser_cdp.go and browser_bidi.go.

func init() {
	tools.ToolRegistry["browser_screenshot"] = tools.Tool{
		Name: "browser_screenshot",
		Description: "Save a PNG screenshot of the page in the open browser to a file. " +
			"By default it captures the visible page; give text or a selector to capture just that element. " +
			"Returns the saved file's path — the user can open it, you cannot see it.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"text": map[string]any{
					"type":        "string",
					"description": "Visible text locating one element to capture instead of the whole page. Optional.",
				},
				"selector": map[string]any{
					"type":        "string",
					"description": "CSS selector locating one element to capture instead of the whole page. Optional.",
				},
				"file": map[string]any{
					"type":        "string",
					"description": "Where to save the PNG. Optional; defaults to atlas-shot-<timestamp>.png in the project directory. An existing file is never overwritten.",
				},
			},
		},
		Run: toolBrowserScreenshot,
	}
	tools.ToolRegistry["browser_tabs"] = tools.Tool{
		Name: "browser_tabs",
		Description: "Manage the open browser's tabs. Set action to one of:\n" +
			"- list: show every open tab with its number, title, and URL.\n" +
			"- switch: drive another tab (tab = its number from list) — needed after a click opened a page in a new tab.\n" +
			"- open: open a new tab (optionally at url) and switch to it.\n" +
			"- close: close a tab (tab = its number; defaults to the current one).",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"action": map[string]any{
					"type": "string",
					"enum": []string{"list", "switch", "open", "close"},
				},
				"tab": map[string]any{
					"type":        "integer",
					"description": "Tab number from action=list. Required for switch; optional for close (defaults to the current tab).",
				},
				"url": map[string]any{
					"type":        "string",
					"description": "For open: the page to load in the new tab. Optional; defaults to a blank tab.",
				},
			},
			"required": []string{"action"},
		},
		Run: toolBrowserTabs,
	}
	tools.ToolRegistry["browser_upload"] = tools.Tool{
		Name: "browser_upload",
		Description: "Attach a local file to a file-upload field (<input type=file>) on the page in the open browser. " +
			"Target the field by its visible label text or a selector. This selects the file the way a user picking it " +
			"from the file dialog would — the page's own submit/upload step still has to happen afterwards. " +
			"Sends local file content to a website, so confirmation is required.",
		Destructive: true,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"text": map[string]any{
					"type":        "string",
					"description": "The upload field's visible label, name, or nearby text.",
				},
				"selector": map[string]any{
					"type":        "string",
					"description": "CSS selector for the <input type=file>, e.g. 'input[type=file]'. Use when text doesn't find it.",
				},
				"file": map[string]any{
					"type":        "string",
					"description": "Path of the file to attach, relative to the project root or absolute. Separate multiple files with commas.",
				},
			},
			"required": []string{"file"},
		},
		Run: toolBrowserUpload,
	}
}

func toolBrowserScreenshot(args map[string]any) (string, error) {
	text, _ := tools.ArgString(args, "text", false)
	selector, _ := tools.ArgString(args, "selector", false)
	file, _ := tools.ArgString(args, "file", false)
	path, err := screenshotPath(file)
	if err != nil {
		return "", err
	}
	return withBrowser(func(s browserSession) (string, error) {
		var clip *screenshotClip
		what := "the visible page"
		if text != "" || selector != "" {
			out, err := s.Eval(elementRectJS(selector, text))
			if err != nil {
				return "", err
			}
			msg, err := decodeActResult(out)
			if err != nil {
				return "", err
			}
			var r struct {
				X float64 `json:"x"`
				Y float64 `json:"y"`
				W float64 `json:"w"`
				H float64 `json:"h"`
			}
			if err := json.Unmarshal([]byte(msg), &r); err != nil || r.W <= 0 || r.H <= 0 {
				return "", fmt.Errorf("could not measure the element to capture (%s)", msg)
			}
			clip = &screenshotClip{X: r.X, Y: r.Y, Width: r.W, Height: r.H}
			what = "the element"
		}
		data, err := s.Screenshot(clip)
		if err != nil {
			return "", err
		}
		if err := os.WriteFile(path, data, 0644); err != nil {
			return "", err
		}
		abs, aerr := filepath.Abs(path)
		if aerr != nil {
			abs = path
		}
		if w, h, err := pngDimensions(data); err == nil {
			return fmt.Sprintf("Saved a %dx%d px screenshot of %s to %s", w, h, what, abs), nil
		}
		return fmt.Sprintf("Saved a screenshot of %s to %s (%d bytes)", what, abs, len(data)), nil
	})
}

// screenshotPath resolves where a screenshot lands. No name: a timestamped
// default, bumped with a counter if taken twice in one second. A given name:
// .png appended if missing, and an existing file refused rather than
// overwritten — a screenshot must never destroy user data.
func screenshotPath(file string) (string, error) {
	if file == "" {
		base := "atlas-shot-" + time.Now().Format("20060102-150405")
		p := base + ".png"
		for i := 2; ; i++ {
			if _, err := os.Stat(p); os.IsNotExist(err) {
				return p, nil
			}
			if i > 99 {
				return "", fmt.Errorf("could not find a free screenshot file name")
			}
			p = fmt.Sprintf("%s-%d.png", base, i)
		}
	}
	if !strings.EqualFold(filepath.Ext(file), ".png") {
		file += ".png"
	}
	if _, err := os.Stat(file); err == nil {
		return "", fmt.Errorf("%s already exists — pass a different file name", file)
	}
	return file, nil
}

// pngDimensions reads width and height from a PNG's IHDR header.
func pngDimensions(b []byte) (int, int, error) {
	if len(b) < 24 || string(b[1:4]) != "PNG" {
		return 0, 0, fmt.Errorf("not a PNG image")
	}
	return int(binary.BigEndian.Uint32(b[16:20])), int(binary.BigEndian.Uint32(b[20:24])), nil
}

// elementRectJS measures one element in document coordinates, for an
// element-clipped screenshot. Returns the {ok,msg} envelope with the rect as
// JSON in msg.
func elementRectJS(selector, text string) string {
	return fmt.Sprintf(`(async () => {%s
		const el = __findAny(%s, %s);
		if (!el) return __err(%s);
		el.scrollIntoView({block: "center"});
		await __animateTo(el);
		const r = el.getBoundingClientRect();
		return __ok(JSON.stringify({x: r.left + window.scrollX, y: r.top + window.scrollY, w: r.width, h: r.height}));
	})()`, actJSHelpers, jsStr(selector), jsStr(text), jsStr(notFoundHint("element", selector, text)))
}

func toolBrowserTabs(args map[string]any) (string, error) {
	action, err := tools.ArgString(args, "action", true)
	if err != nil {
		return "", err
	}
	switch action {
	case "list", "switch", "open", "close":
	default:
		return "", fmt.Errorf("unknown action %q (expected list, switch, open, or close)", action)
	}
	url, _ := tools.ArgString(args, "url", false)
	n, hasN := argTabNumber(args)
	if action == "switch" && !hasN {
		return "", fmt.Errorf("switch needs tab — the tab number shown by action=list")
	}
	return withBrowser(func(s browserSession) (string, error) {
		if action == "open" {
			if err := s.NewTab(normalizeURL(url)); err != nil {
				return "", err
			}
			return "Opened a new tab and switched to it. " + pageSummary(s), nil
		}
		tabs, err := s.Tabs()
		if err != nil {
			return "", err
		}
		if len(tabs) == 0 {
			return "", fmt.Errorf("no open tabs found")
		}
		if hasN && (n < 1 || n > len(tabs)) {
			return "", fmt.Errorf("tab %d does not exist — there are %d tabs (action=list shows them)", n, len(tabs))
		}
		switch action {
		case "list":
			return renderTabs(tabs), nil
		case "switch":
			t := tabs[n-1]
			if err := s.SwitchTab(t.ID); err != nil {
				return "", err
			}
			return fmt.Sprintf("Switched to tab %d (%s). Call browser_read to see it.", n, tabDisplay(t)), nil
		default: // close
			idx := -1
			if hasN {
				idx = n - 1
			} else {
				for i, t := range tabs {
					if t.Active {
						idx = i
					}
				}
				if idx < 0 {
					idx = 0
				}
			}
			if len(tabs) == 1 {
				return "", fmt.Errorf("that is the only tab — closing it would close the whole browser; use browser_close for that")
			}
			target := tabs[idx]
			if target.Active {
				// Never leave the session driving a closed tab.
				other := 0
				if idx == 0 {
					other = 1
				}
				if err := s.SwitchTab(tabs[other].ID); err != nil {
					return "", err
				}
			}
			if err := s.CloseTab(target.ID); err != nil {
				return "", err
			}
			return fmt.Sprintf("Closed tab %d (%s). %s", idx+1, tabDisplay(target), pageSummary(s)), nil
		}
	})
}

// argTabNumber reads the tab argument, tolerating the string numbers small
// models like to emit.
func argTabNumber(args map[string]any) (int, bool) {
	switch v := args["tab"].(type) {
	case float64:
		return int(v), true
	case string:
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil {
			return n, true
		}
	}
	return 0, false
}

func renderTabs(tabs []browserTab) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%d open tab(s):\n", len(tabs))
	for i, t := range tabs {
		marker := ""
		if t.Active {
			marker = "  (active)"
		}
		fmt.Fprintf(&b, "%2d. %s%s\n", i+1, tabDisplay(t), marker)
	}
	b.WriteString("Use action=switch tab=N to drive another tab.")
	return b.String()
}

func tabDisplay(t browserTab) string {
	title := strings.TrimSpace(t.Title)
	if title == "" {
		return t.URL
	}
	return title + " — " + t.URL
}

func toolBrowserUpload(args map[string]any) (string, error) {
	selector, _ := tools.ArgString(args, "selector", false)
	text, _ := tools.ArgString(args, "text", false)
	if selector == "" && text == "" {
		return "", fmt.Errorf("upload needs either text (the upload field's label) or a selector for the file input")
	}
	fileArg, err := tools.ArgString(args, "file", true)
	if err != nil {
		return "", err
	}
	var paths []string
	for _, p := range strings.Split(fileArg, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		// Browsers require absolute paths for file inputs.
		abs, err := filepath.Abs(p)
		if err != nil {
			return "", err
		}
		if info, err := os.Stat(abs); err != nil || info.IsDir() {
			return "", fmt.Errorf("file %s does not exist (or is a directory)", p)
		}
		paths = append(paths, abs)
	}
	if len(paths) == 0 {
		return "", fmt.Errorf("upload needs file — the path of the file to attach")
	}
	return withBrowser(func(s browserSession) (string, error) {
		out, err := s.Eval(markUploadTargetJS(selector, text))
		if err != nil {
			return "", err
		}
		if _, err := decodeActResult(out); err != nil {
			return "", err
		}
		if err := s.SetFiles(paths); err != nil {
			return "", err
		}
		_, _ = s.Eval(unmarkUploadTargetJS)
		return fmt.Sprintf("Attached %d file(s) to the file input. The page now has the file selected — "+
			"click the page's own submit/upload control to actually send it.", len(paths)), nil
	})
}

// markUploadTargetJS finds the file input and tags it with a marker attribute
// the protocol-level SetFiles can select. File inputs are matched without the
// visibility filter the other finders use: real pages routinely hide the
// <input type=file> behind a styled button.
func markUploadTargetJS(selector, text string) string {
	return fmt.Sprintf(`(() => {%s
		const sel = %s, want = __norm(%s);
		const inputs = Array.from(document.querySelectorAll("input[type=file]"));
		let el = null;
		if (sel) {
			el = document.querySelector(sel);
			if (el && !(el.tagName === "INPUT" && el.type === "file"))
				return __err("the selector matched <" + el.tagName.toLowerCase() + ">, not an <input type=file>");
		} else {
			const attr = (e, a) => __norm(e.getAttribute(a));
			el = inputs.find(e => [attr(e, "name"), attr(e, "id"), attr(e, "aria-label")].includes(want));
			if (!el) {
				const lbl = Array.from(document.querySelectorAll("label")).find(l => __norm(l.innerText).includes(want));
				if (lbl) el = (lbl.getAttribute("for") && document.getElementById(lbl.getAttribute("for"))) || lbl.querySelector("input[type=file]");
			}
			if (!el && inputs.length === 1) el = inputs[0];
		}
		if (!el || el.type !== "file")
			return __err("no file input (<input type=file>) found" + (inputs.length ? " matching that — the page has " + inputs.length + " file input(s); try selector='input[type=file]'" : " — the page has none"));
		document.querySelectorAll("[data-atlas-upload]").forEach(e => e.removeAttribute("data-atlas-upload"));
		el.setAttribute("data-atlas-upload", "1");
		return __ok("found the file input");
	})()`, actJSHelpers, jsStr(selector), jsStr(text))
}

// unmarkUploadTargetJS removes the marker again once the files are attached.
const unmarkUploadTargetJS = `(() => {
	document.querySelectorAll("[data-atlas-upload]").forEach(e => e.removeAttribute("data-atlas-upload"));
	return "";
})()`
