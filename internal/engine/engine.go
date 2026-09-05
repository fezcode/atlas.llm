package engine

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"atlas.llm/internal/catalog"
	"atlas.llm/internal/config"
	"atlas.llm/internal/tools"
)

// ProgressFn is called as bytes stream in. total may be -1 if unknown.
type ProgressFn func(written, total int64)

type countingWriter struct {
	written int64
	total   int64
	onWrite ProgressFn
}

func (cw *countingWriter) Write(p []byte) (int, error) {
	n := len(p)
	cw.written += int64(n)
	if cw.onWrite != nil {
		cw.onWrite(cw.written, cw.total)
	}
	return n, nil
}

func downloadFile(dest, url string, onProgress ProgressFn) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer out.Close()

	cw := &countingWriter{total: resp.ContentLength, onWrite: onProgress}
	if _, err := io.Copy(io.MultiWriter(out, cw), resp.Body); err != nil {
		return err
	}
	return nil
}

// githubAsset is the subset of GitHub's release-asset JSON we care about.
type GithubAsset struct {
	Name               string `json:"name"`
	BrowserDownloadURL string `json:"browser_download_url"`
}

type GithubRelease struct {
	TagName string        `json:"tag_name"`
	Assets  []GithubAsset `json:"assets"`
}

// latestLlamacppAsset resolves the correct llama.cpp release asset URL for
// this OS/arch by querying GitHub for the latest release.
func latestLlamacppAsset(variant string) (string, string, string, error) {
	asset, err := EngineAssetSuffix(variant)
	if err != nil {
		return "", "", "", err
	}

	resp, err := http.Get(llamacppLatestURL)
	if err != nil {
		return "", "", "", fmt.Errorf("fetch latest release: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", "", "", fmt.Errorf("GitHub API returned %s", resp.Status)
	}

	var rel GithubRelease
	if err := json.NewDecoder(resp.Body).Decode(&rel); err != nil {
		return "", "", "", fmt.Errorf("decode release JSON: %w", err)
	}

	engineURL := FindReleaseAsset(rel, asset.Suffix, EnginePrefix)
	if engineURL == "" {
		return "", "", "", fmt.Errorf("no asset ending in %q in release %s", asset.Suffix, rel.TagName)
	}
	var companionURL string
	if asset.Companion != "" {
		// The companion filename has no build tag, so match it exactly.
		companionURL = FindReleaseAsset(rel, asset.Companion, "")
		if companionURL == "" {
			return "", "", "", fmt.Errorf("release %s has the %s engine but not its runtime %q",
				rel.TagName, variant, asset.Companion)
		}
	}
	return engineURL, companionURL, rel.TagName, nil
}

// enginePrefix distinguishes the engine archive from its CUDA runtime
// companion — `llama-b10280-bin-win-cuda-12.4-x64.zip` and
// `cudart-llama-bin-win-cuda-12.4-x64.zip` share a suffix, so matching on
// the tail alone would be ambiguous.
const EnginePrefix = "llama-"

func FindReleaseAsset(rel GithubRelease, suffix, prefix string) string {
	for _, a := range rel.Assets {
		if !strings.HasSuffix(a.Name, suffix) {
			continue
		}
		if prefix != "" && !strings.HasPrefix(a.Name, prefix) {
			continue
		}
		return a.BrowserDownloadURL
	}
	return ""
}

// downloadEngine fetches the latest llama.cpp prebuilt archive for the
// current platform, extracts it into the engine dir, and removes any legacy
// llamafile binary left over from older atlas.llm versions.
func DownloadEngine(onProgress ProgressFn) error {
	cfg, _ := config.LoadConfig()
	variant := PlannedEngineVariant(cfg)

	// An engine from a different variant would leave the wrong binaries
	// behind, so a variant switch forces a clean re-download.
	if IsEngineDownloaded() && InstalledEngineVariant() == variant {
		return nil
	}
	url, companionURL, _, err := latestLlamacppAsset(variant)
	if err != nil {
		return err
	}
	dir, err := EngineDir()
	if err != nil {
		return err
	}
	if IsEngineDownloaded() {
		if err := clearEngineDir(); err != nil {
			return fmt.Errorf("clear previous engine: %w", err)
		}
	}

	if err := fetchAndExtract(url, dir, "llamacpp", onProgress); err != nil {
		return err
	}
	// CUDA builds link against runtime DLLs shipped in a second archive;
	// without it llama-server won't start.
	if companionURL != "" {
		if err := fetchAndExtract(companionURL, dir, "cudart", onProgress); err != nil {
			return fmt.Errorf("CUDA runtime: %w", err)
		}
	}

	if runtime.GOOS != "windows" {
		if bin, err := findEngineBinary(); err == nil {
			_ = os.Chmod(bin, 0755)
		}
	}

	if err := WriteEngineVariant(variant); err != nil {
		return err
	}

	// Best-effort cleanup of the old llamafile binary from pre-0.4 installs.
	if base, err := config.AtlasDir(); err == nil {
		for _, name := range []string{"llamafile", "llamafile.exe"} {
			_ = os.Remove(filepath.Join(base, name))
		}
	}

	return nil
}

// engineVariantFile records which build variant is currently extracted, so
// a later /set engine_variant can tell whether a re-download is needed.
func engineVariantFile() (string, error) {
	dir, err := EngineDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, ".variant"), nil
}

// plannedEngineVariant is the build `/download engine` would install.
//
// An explicit setting always wins. Under auto we deliberately keep whatever
// is already installed rather than acting on detection: nvidia-smi answering
// doesn't prove the CUDA build will load (a driver can be too old for the
// runtime), and silently replacing a working engine with a ~510MB download
// is a bad trade to make on the user's behalf. Detection therefore only
// decides fresh installs; existing ones get a suggestion instead, via
// engineUpgradeHint.
func PlannedEngineVariant(cfg config.Config) string {
	if explicit := strings.TrimSpace(cfg.EngineVariant); explicit != "" &&
		!strings.EqualFold(explicit, EngineVariantAuto) {
		return ResolveEngineVariant(explicit)
	}
	if IsEngineDownloaded() {
		return InstalledEngineVariant()
	}
	return ResolveEngineVariant(EngineVariantAuto)
}

// engineNeedsDownload reports whether `/download engine` has work to do:
// nothing installed, or an installed build that differs from the planned
// one. Without the second case a variant switch silently no-ops, which is
// what made `/set engine_variant cuda` impossible to act on.
func EngineNeedsDownload() bool {
	if !IsEngineDownloaded() {
		return true
	}
	cfg, _ := config.LoadConfig()
	return InstalledEngineVariant() != PlannedEngineVariant(cfg)
}

// engineUpgradeHint returns a one-line suggestion when a detected GPU is
// faster than the installed engine, or "" when there's nothing to say. The
// user asked for this rather than an automatic switch, so it stays advice.
func EngineUpgradeHint() string {
	if IsDarwin() || !IsEngineDownloaded() || EngineVariantIsGPU(InstalledEngineVariant()) {
		return ""
	}
	info, ok := DetectGPU()
	if !ok {
		return ""
	}
	asset, err := EngineAssetSuffix(EngineVariantCUDA)
	if err != nil {
		return ""
	}
	return fmt.Sprintf(
		"%s detected, but the installed engine is a CPU-only build.\n"+
			"Run `/set engine_variant cuda` then `/download engine` (%s) to offload the model to the GPU.",
		info.Name, asset.Size)
}

// installedEngineVariant reports the extracted engine's variant. Installs
// predating variant support have no marker; they're CPU builds.
func InstalledEngineVariant() string {
	p, err := engineVariantFile()
	if err != nil {
		return EngineVariantCPU
	}
	data, err := os.ReadFile(p)
	if err != nil {
		return EngineVariantCPU
	}
	v := strings.TrimSpace(string(data))
	if v == "" {
		return EngineVariantCPU
	}
	return v
}

func WriteEngineVariant(variant string) error {
	p, err := engineVariantFile()
	if err != nil {
		return err
	}
	return os.WriteFile(p, []byte(variant+"\n"), 0644)
}

// clearEngineDir empties the engine directory ahead of installing a
// different variant.
func clearEngineDir() error {
	dir, err := EngineDir()
	if err != nil {
		return err
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if err := os.RemoveAll(filepath.Join(dir, e.Name())); err != nil {
			return err
		}
	}
	return nil
}

// fetchAndExtract downloads one archive into destDir and unpacks it,
// removing the archive afterwards. basename only names the temp file.
func fetchAndExtract(url, destDir, basename string, onProgress ProgressFn) error {
	archiveName := basename + filepath.Ext(url)
	if strings.HasSuffix(url, ".tar.gz") {
		archiveName = basename + ".tar.gz"
	}
	archivePath := filepath.Join(destDir, archiveName)

	if err := downloadFile(archivePath, url, onProgress); err != nil {
		return err
	}
	defer os.Remove(archivePath)

	if strings.HasSuffix(archiveName, ".zip") {
		if err := extractZip(archivePath, destDir); err != nil {
			return fmt.Errorf("extract zip: %w", err)
		}
		return nil
	}
	if err := extractTarGz(archivePath, destDir); err != nil {
		return fmt.Errorf("extract tar.gz: %w", err)
	}
	return nil
}

func extractZip(src, destDir string) error {
	r, err := zip.OpenReader(src)
	if err != nil {
		return err
	}
	defer r.Close()
	cleanDest := filepath.Clean(destDir) + string(os.PathSeparator)
	for _, f := range r.File {
		target := filepath.Join(destDir, f.Name)
		if !strings.HasPrefix(target, cleanDest) {
			return fmt.Errorf("zip slip: %s", f.Name)
		}
		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(target, 0755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
			return err
		}
		if err := writeZipEntry(f, target); err != nil {
			return err
		}
	}
	return nil
}

func writeZipEntry(f *zip.File, target string) error {
	rc, err := f.Open()
	if err != nil {
		return err
	}
	defer rc.Close()
	out, err := os.OpenFile(target, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, rc)
	return err
}

func extractTarGz(src, destDir string) error {
	f, err := os.Open(src)
	if err != nil {
		return err
	}
	defer f.Close()
	gzr, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gzr.Close()
	tr := tar.NewReader(gzr)
	cleanDest := filepath.Clean(destDir) + string(os.PathSeparator)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		target := filepath.Join(destDir, hdr.Name)
		if !strings.HasPrefix(target, cleanDest) {
			return fmt.Errorf("tar slip: %s", hdr.Name)
		}
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
				return err
			}
			if err := writeTarEntry(tr, target, os.FileMode(hdr.Mode)); err != nil {
				return err
			}
		case tar.TypeSymlink:
			// The macOS/Linux llama.cpp tarballs ship versioned dylibs plus a
			// chain of symlinks (libggml.dylib -> libggml.0.dylib ->
			// libggml.0.18.1.dylib). Dropping these leaves llama-server with
			// unresolvable @rpath references, so it dies at exec time.
			if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
				return err
			}
			if err := writeTarSymlink(hdr.Linkname, target, cleanDest); err != nil {
				return err
			}
		}
	}
}

// writeTarSymlink recreates a symlink entry, rejecting any link that would
// resolve outside the extraction root.
func writeTarSymlink(linkname, target, cleanDest string) error {
	resolved := linkname
	if !filepath.IsAbs(resolved) {
		resolved = filepath.Join(filepath.Dir(target), resolved)
	}
	if !strings.HasPrefix(filepath.Clean(resolved), strings.TrimSuffix(cleanDest, string(os.PathSeparator))) {
		return fmt.Errorf("symlink escapes archive root: %s -> %s", target, linkname)
	}
	// Re-extracting over a previous install must not fail on EEXIST.
	if err := os.Remove(target); err != nil && !os.IsNotExist(err) {
		return err
	}
	return os.Symlink(linkname, target)
}

func writeTarEntry(tr *tar.Reader, target string, mode os.FileMode) error {
	out, err := os.OpenFile(target, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, mode)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, tr)
	return err
}

// downloadModel fetches a model into models/. No-op if already present.
func DownloadModel(m catalog.Model, onProgress ProgressFn) error {
	p, err := config.ModelPath(m)
	if err != nil {
		return err
	}
	if config.IsModelDownloaded(m) {
		return nil
	}
	return downloadFile(p, m.URL, onProgress)
}

// requireEngine returns the path to llama-cli[.exe] or an error asking the
// user to /download. Does NOT download automatically.
func requireEngine() (string, error) {
	if !IsEngineDownloaded() {
		return "", fmt.Errorf("inference engine is not downloaded — run /download engine (or /download) in chat")
	}
	return findEngineBinary()
}

// requireModel returns the model path or an error asking the user to /download.
// Does NOT download automatically.
func requireModel(m catalog.Model) (string, error) {
	p, err := config.ModelPath(m)
	if err != nil {
		return "", err
	}
	if !config.IsModelDownloaded(m) {
		return "", fmt.Errorf("model %q is not downloaded — run /download %s in chat", m.Name, m.Name)
	}
	return p, nil
}

// runChat drives a /v1/chat/completions call against the persistent
// llama-server. The server is lazy-started on the first call per process
// (or whenever the active model changes) so the GGUF mmap + warmup cost is
// paid once per session, not once per turn.
//
// Using chat completions (instead of raw /completion) means llama-server
// applies the model's native chat template — Gemma 3's
// <start_of_turn>/<end_of_turn> sentinels, ChatML, etc. — and stops at the
// turn boundary. Raw completion with "User:/Assistant:" markers was causing
// the model to hallucinate additional fake turns after its real answer.
func runChat(ctx context.Context, msgs []ChatMsg, maxTokens int) (string, error) {
	if err := RequireInference(); err != nil {
		return "", err
	}

	s, err := EnsureServer()
	if err != nil {
		return "", fmt.Errorf("server: %w", err)
	}
	cfg, _ := config.LoadConfig()
	out, err := s.chatCompleteRespectingReasoning(ctx, msgs, maxTokens, cfg)
	if err != nil {
		return "", fmt.Errorf("inference failed: %w", err)
	}
	return strings.TrimSpace(out), nil
}

// runSingleUser is a convenience wrapper for one-shot tasks (summarize,
// grep) that have no conversational history — just a single user prompt.
// Thinking is disabled for these: a reasoning model's <think> block adds
// nothing to a summarization or extraction task, and on Qwen3.5 it consumed
// the whole budget and returned an empty answer.
func RunSingleUser(ctx context.Context, system, user string, maxTokens int) (string, error) {
	msgs := []ChatMsg{}
	if system != "" {
		msgs = append(msgs, ChatMsg{Role: "system", Content: system})
	}
	msgs = append(msgs, ChatMsg{Role: "user", Content: user})
	return runChatNoThinking(ctx, msgs, maxTokens)
}

// runChatNoThinking mirrors runChat but asks the chat template to skip the
// reasoning block.
func runChatNoThinking(ctx context.Context, msgs []ChatMsg, maxTokens int) (string, error) {
	if err := RequireInference(); err != nil {
		return "", err
	}
	s, err := EnsureServer()
	if err != nil {
		return "", fmt.Errorf("server: %w", err)
	}
	out, err := s.ChatCompleteNoThinking(ctx, msgs, maxTokens)
	if err != nil {
		return "", fmt.Errorf("inference failed: %w", err)
	}
	return strings.TrimSpace(out), nil
}

// summarizeMaxChars caps the amount of file content we send to the model
// per /summarize request. Roughly 10K chars ≈ 2.5K tokens, well inside the
// server's 16K ctx even after the system prompt, chat-template overhead,
// and the 512-token reply budget.
const summarizeMaxChars = 10000

func SummarizeContent(content string) (string, error) {
	truncated := content
	if len(truncated) > summarizeMaxChars {
		truncated = truncated[:summarizeMaxChars] + "\n\n... (file truncated for summary)"
	}
	return RunSingleUser(
		context.Background(),
		"You are a concise code summarizer. Respond with only 1-3 plain sentences describing the file's purpose. Do not use markdown, code blocks, or lists.",
		"Summarize this file:\n\n"+truncated,
		512,
	)
}

type ChatMessage struct {
	Role    string // "user" or "assistant"
	Content string
}

// runAgentStep advances a tool-enabled conversation by one round-trip: it
// POSTs the current message list (plus tool definitions) and returns the
// assistant content and any requested tool calls. Callers loop until the
// returned toolCalls list is empty.
func RunAgentStep(ctx context.Context, msgs []ChatMsg, maxTokens int) (string, []ToolCall, error) {
	if err := RequireInference(); err != nil {
		return "", nil, err
	}
	s, err := EnsureServer()
	if err != nil {
		return "", nil, fmt.Errorf("server: %w", err)
	}
	cfgTools, _ := config.LoadConfig()
	// An agentic turn: thinking earns its cost here by improving which tool
	// gets chosen.
	content, calls, err := s.ChatCompleteWithToolsOpt(ctx, msgs, tools.ToolDefsJSON(), maxTokens,
		!config.ReasoningEnabledFor(cfgTools, true))
	if err != nil {
		return "", nil, fmt.Errorf("inference failed: %w", err)
	}
	return strings.TrimSpace(content), calls, nil
}

// agentSystemPrompt is prepended to the conversation when tools are
// enabled. Tells the model it has filesystem + shell capabilities, and
// that destructive actions require user approval (so it doesn't loop on
// unexpected denials).
const agentSystemPrompt = `You are atlas, a concise coding assistant with access to the user's local project via tools. All paths are relative to the project root — the directory atlas.llm was started in — and tools cannot read or write outside it. Start with list_dir on "." if you are unsure of the layout, rather than guessing directory names. run_cmd starts a fresh shell each call, so pass cwd to work in a subdirectory instead of using "cd". Use tools when you need to inspect or change files or run commands — don't guess about file contents. Some of the tools you are given come from connected MCP servers and reach external systems such as Slack or Confluence; prefer calling them over guessing when the user asks about something outside the local project. Some tools require the user to approve each call; if a call is denied, acknowledge and continue without retrying. When the user asks you to browse or use a website, open a visible browser with browser_open, then drive it with browser_navigate, browser_read, and browser_act — the user is watching the window, so narrate briefly what you are doing. browser_open uses a fresh profile with no logins by default; only pass profile="default" when the user explicitly asks to browse signed in as themselves. After gathering what you need, answer plainly in markdown.`

// agentSystemPromptNow is the system prompt with the current project root
// and its top-level layout filled in. Built per turn rather than baked in as
// a constant, because a model that cannot see the layout guesses directory
// names and then burns its tool-call budget retrying the guesses.
func AgentSystemPromptNow() string {
	var b strings.Builder
	b.WriteString(agentSystemPrompt)
	b.WriteString("\n\nProject root: ")
	b.WriteString(tools.SessionRoot())
	if overview := tools.ProjectOverview(40); overview != "" {
		b.WriteString("\nTop level: ")
		b.WriteString(overview)
	}
	b.WriteString("\nYou are already in the project root. There is no need to cd into it.")
	return b.String()
}

func Chat(ctx context.Context, history []ChatMessage, userInput string) (string, error) {
	msgs := []ChatMsg{
		{Role: "system", Content: "You are a concise, helpful coding assistant. Keep replies under three short paragraphs unless more detail is explicitly requested."},
	}
	for _, m := range history {
		msgs = append(msgs, ChatMsg{Role: m.Role, Content: m.Content})
	}
	msgs = append(msgs, ChatMsg{Role: "user", Content: userInput})
	cfg, _ := config.LoadConfig()
	return runChat(ctx, msgs, cfg.MaxTokens)
}

func FormatBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for n2 := n / unit; n2 >= unit; n2 /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "KMGTPE"[exp])
}

// downloadMeter turns raw progress callbacks into a display-worthy transfer
// speed and elapsed time. The speed is exponentially smoothed with a ~1.5s
// time constant: raw 100ms deltas flicker, while an average over the whole
// download would keep showing the old rate long after the link slows down.
// Timestamps come in as arguments so tests never have to sleep.
type DownloadMeter struct {
	Start time.Time
	prevT time.Time
	prevB int64
	Rate  float64 // bytes/sec, smoothed
}

func (d *DownloadMeter) Observe(now time.Time, written int64) {
	if d.Start.IsZero() {
		d.Start, d.prevT, d.prevB = now, now, written
		return
	}
	dt := now.Sub(d.prevT).Seconds()
	if dt <= 0 {
		return
	}
	inst := float64(written-d.prevB) / dt
	if inst < 0 {
		inst = 0 // the stream restarted from an earlier offset
	}
	const tau = 1.5
	if d.Rate == 0 {
		d.Rate = inst // adopt the first measurement instead of warming up from 0
	} else {
		alpha := dt / (dt + tau)
		d.Rate += alpha * (inst - d.Rate)
	}
	d.prevT, d.prevB = now, written
}

// speed returns the smoothed transfer rate in bytes per second, 0 until two
// samples have arrived.
func (d *DownloadMeter) Speed() float64 { return d.Rate }

func (d *DownloadMeter) Elapsed(now time.Time) time.Duration {
	if d.Start.IsZero() {
		return 0
	}
	return now.Sub(d.Start)
}

func FormatSpeed(bytesPerSec float64) string {
	return FormatBytes(int64(bytesPerSec)) + "/s"
}

// formatElapsed renders a wall-clock style duration: m:ss, or h:mm:ss once
// a download has run for over an hour.
func FormatElapsed(d time.Duration) string {
	s := int(d.Seconds())
	if s < 3600 {
		return fmt.Sprintf("%d:%02d", s/60, s%60)
	}
	return fmt.Sprintf("%d:%02d:%02d", s/3600, s%3600/60, s%60)
}

// runChatStream is the streaming counterpart to runChat: it forwards each
// delta to onDelta as it arrives and returns the assembled reply.
func runChatStream(ctx context.Context, msgs []ChatMsg, maxTokens int, onDelta func(StreamDelta)) (string, error) {
	if err := RequireInference(); err != nil {
		return "", err
	}
	s, err := EnsureServer()
	if err != nil {
		return "", fmt.Errorf("server: %w", err)
	}
	cfg, _ := config.LoadConfig()
	out, err := s.ChatCompleteStreamOpt(ctx, msgs, maxTokens, !config.ReasoningEnabled(cfg), onDelta)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(out), nil
}

// chatStream mirrors chat() but streams the reply.
func ChatStream(ctx context.Context, history []ChatMessage, userInput string, onDelta func(StreamDelta)) (string, error) {
	msgs := []ChatMsg{
		{Role: "system", Content: "You are a concise, helpful coding assistant. Keep replies under three short paragraphs unless more detail is explicitly requested."},
	}
	for _, m := range history {
		msgs = append(msgs, ChatMsg{Role: m.Role, Content: m.Content})
	}
	msgs = append(msgs, ChatMsg{Role: "user", Content: userInput})
	cfg, _ := config.LoadConfig()
	return runChatStream(ctx, msgs, cfg.MaxTokens, onDelta)
}

// requireInference checks that this machine can actually run a turn. In
// remote mode it always can: the engine and the model file live on the
// server, so demanding them locally would reject exactly the setup the
// endpoint setting exists to enable.
func RequireInference() error {
	if config.IsRemoteMode() {
		return nil
	}
	if _, err := requireEngine(); err != nil {
		return err
	}
	if m, err := config.CurrentModel(); err == nil {
		if _, err := requireModel(m); err != nil {
			return err
		}
	}
	return nil
}
