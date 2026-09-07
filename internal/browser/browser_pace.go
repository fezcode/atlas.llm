package browser

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net/url"
	"strings"
	"sync"
	"time"
)

// Politeness and challenge handling — the two things that only bite once the
// browser is driven hard. The persistent profile is where they bite first:
// every visit arrives as the same signed-in identity, so a site's per-account
// counters accumulate instead of resetting the way a throwaway profile's do.
//
// The pacer answers rate limiting: a jittered minimum gap between loads of one
// host, widened for a host that has already pushed back and relaxed again once
// it stops. The detector answers the rest — an interstitial is recognised as
// one instead of being handed to the model as if it were the page it asked
// for, and because the window is headed and the user is watching it, the
// recovery for a human check is a human.

const (
	paceBase       = 900 * time.Millisecond // floor between two loads of one host
	paceJitterMax  = 700 * time.Millisecond // added at random, so the rhythm is not machine-regular
	paceGapMax     = 30 * time.Second       // ceiling for a host that keeps pushing back
	paceBackoffAdd = 2 * time.Second        // flat addition per push-back: doubling alone starts too small to feel
)

// pacer spaces out loads per host. All of its policy lives in delay() so it
// can be tested without a clock.
type pacer struct {
	mu   sync.Mutex
	last map[string]time.Time     // when we last loaded each host
	gap  map[string]time.Duration // that host's earned penalty, if any
	host string                   // most recent host, for actions that reload the page they are on

	now    func() time.Time
	sleep  func(time.Duration)
	jitter func() time.Duration
}

func newPacer() *pacer {
	return &pacer{
		last:   map[string]time.Time{},
		gap:    map[string]time.Duration{},
		now:    time.Now,
		sleep:  time.Sleep,
		jitter: func() time.Duration { return time.Duration(rand.Int63n(int64(paceJitterMax))) },
	}
}

// browserPacer is process-wide: one browser window is driven at a time, and a
// host's penalty should outlive the session that earned it.
var browserPacer = newPacer()

// hostOf reduces a URL to the key the pacer counts against. Anything that is
// not ordinary http(s) — about:blank, a file:// path — paces against nothing,
// since there is no server there to annoy.
func hostOf(raw string) string {
	u, err := url.Parse(raw)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") {
		return ""
	}
	return strings.ToLower(u.Hostname())
}

// delay reports how long to hold off before loading host again. A host we
// have not seen is never delayed: the cost of pacing should fall on repeated
// hammering, not on the first visit.
func (p *pacer) delay(host string, now time.Time) time.Duration {
	if host == "" {
		return 0
	}
	last, seen := p.last[host]
	if !seen {
		return 0
	}
	gap := max(p.gap[host], paceBase)
	gap += p.jitter()
	if elapsed := now.Sub(last); elapsed < gap {
		return gap - elapsed
	}
	return 0
}

// pace blocks until it is polite to load host, then records the hit.
func (p *pacer) pace(host string) {
	p.mu.Lock()
	wait := p.delay(host, p.now())
	p.mu.Unlock()

	if wait > 0 {
		p.sleep(wait)
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	if host != "" {
		p.last[host] = p.now()
		p.host = host
	}
}

// pushBack widens a host's gap after it has rate-limited or challenged us.
func (p *pacer) pushBack(host string) {
	if host == "" {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	gap := max(p.gap[host], paceBase)
	p.gap[host] = min(gap*2+paceBackoffAdd, paceGapMax)
}

// relax halves a host's penalty after a clean load, so one bad stretch does
// not slow the rest of the session to a crawl.
func (p *pacer) relax(host string) {
	if host == "" {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	gap, ok := p.gap[host]
	if !ok {
		return
	}
	if gap <= paceBase {
		delete(p.gap, host)
		return
	}
	p.gap[host] = gap / 2
}

// lastHost is the host of the page we most recently loaded. Actions that
// reload the page they are already on (click, press, reload) have no URL to
// pace against, so they pace against this.
func (p *pacer) lastHost() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.host
}

// --- challenge detection ----------------------------------------------------

// challenge is a page that is not the content the model asked for: a human
// check, or a notice that the site is counting our requests.
type challenge struct {
	Kind string // "captcha" or "ratelimit"
	Why  string // the signature that matched, quoted back so the claim is checkable
}

// ratePhrases mark a site that is counting requests rather than doubting our
// humanity. Only unambiguous wordings belong here: a false positive stalls a
// page that was working.
var ratePhrases = []string{
	"too many requests",
	"rate limited",
	"rate limit exceeded",
	"error 1015",
	"you are being rate limited",
	"slow down and try again",
}

// captchaPhrases are wordings that appear on a human-verification
// interstitial and essentially nowhere else.
var captchaPhrases = []string{
	"verify you are human",
	"verify you're human",
	"confirm you are human",
	"are you a robot",
	"i'm not a robot",
	"checking your browser",
	"enable javascript and cookies to continue",
	"complete the security check",
	"unusual traffic from your computer network",
	"our systems have detected unusual traffic",
	"press and hold to confirm",
}

// captchaTitles are matched against the title alone. "Just a moment" is
// Cloudflare's interstitial, but it is also an ordinary English phrase that
// could appear anywhere in an article's body.
var captchaTitles = []string{
	"just a moment",
	"attention required",
}

// captchaAssets are the widget providers. A page that pulls one in is asking
// for a human whatever its wording says.
var captchaAssets = []string{
	"challenges.cloudflare.com",
	"hcaptcha.com",
	"recaptcha.net",
	"google.com/recaptcha",
	"gstatic.com/recaptcha",
	"geo.captcha-delivery.com", // DataDome
	"captcha.awswaf.com",
	"funcaptcha.com",
	"arkoselabs.com",
	"perimeterx.net",
}

// detectChallenge decides whether what loaded is an interstitial. Rate limits
// are tested first: a page that says it is throttling us wants us to wait, and
// no amount of clicking a checkbox helps.
func detectChallenge(title, text string, assets []string) *challenge {
	hay := strings.ToLower(title + "\n" + text)
	for _, phrase := range ratePhrases {
		if strings.Contains(hay, phrase) {
			return &challenge{Kind: "ratelimit", Why: phrase}
		}
	}
	for _, phrase := range captchaPhrases {
		if strings.Contains(hay, phrase) {
			return &challenge{Kind: "captcha", Why: phrase}
		}
	}
	lowTitle := strings.ToLower(title)
	for _, phrase := range captchaTitles {
		if strings.Contains(lowTitle, phrase) {
			return &challenge{Kind: "captcha", Why: phrase}
		}
	}
	for _, src := range assets {
		low := strings.ToLower(src)
		for _, host := range captchaAssets {
			if strings.Contains(low, host) {
				return &challenge{Kind: "captcha", Why: host}
			}
		}
	}
	return nil
}

// challengeProbeJS collects everything detectChallenge needs in one round
// trip: the title, the first screenful of text, and the sources the page
// pulls in (captcha widgets arrive as an iframe or a script).
func challengeProbeJS() string {
	return `(() => JSON.stringify({
		title: document.title || "",
		text: ((document.body && document.body.innerText) || "").slice(0, 4000),
		assets: Array.from(document.querySelectorAll("iframe[src], script[src]")).slice(0, 40).map(e => e.src || "")
	}))()`
}

// pageChallenge probes the live page. Any failure reads as "no challenge": a
// probe that cannot run must never turn a working page into an error.
func pageChallenge(s browserSession) *challenge {
	out, err := s.Eval(challengeProbeJS())
	if err != nil {
		return nil
	}
	var probe struct {
		Title  string   `json:"title"`
		Text   string   `json:"text"`
		Assets []string `json:"assets"`
	}
	if err := json.Unmarshal([]byte(out), &probe); err != nil {
		return nil
	}
	return detectChallenge(probe.Title, probe.Text, probe.Assets)
}

// message says what the page actually is and what to do about it. The window
// is headed and the user is already watching it, so for a human check the
// answer is to ask them.
func (c *challenge) message(host string) string {
	where := host
	if where == "" {
		where = "This site"
	}
	if c.Kind == "ratelimit" {
		return fmt.Sprintf("%s is rate-limiting this session (the page says %q) — what follows is the block notice, "+
			"not the content you asked for. atlas.llm has widened the gap between requests to this host on its own. "+
			"Do not reload in a loop: that is what got us here. Tell the user what happened and ask whether to wait "+
			"and retry or stop.", where, c.Why)
	}
	return fmt.Sprintf("%s is showing a human-verification challenge (matched %q) — what follows is the challenge page, "+
		"not the content you asked for. The browser window is visible to the user: ask them to solve the check in it, "+
		`then call browser_act action="wait_human" to wait for the real page to come through.`, where, c.Why)
}

// navigateWithCare is the whole policy in one place: pace the host, load, then
// report a challenge rather than letting an interstitial pass for content.
func navigateWithCare(s browserSession, rawURL string) (*challenge, error) {
	host := hostOf(rawURL)
	browserPacer.pace(host)
	if err := s.Navigate(rawURL); err != nil {
		return nil, err
	}
	if c := pageChallenge(s); c != nil {
		browserPacer.pushBack(host)
		return c, nil
	}
	browserPacer.relax(host)
	return nil, nil
}

// humanWait is how long browser_act action="wait_human" holds the tool open.
// Long enough for someone to notice the window, read the check and solve it;
// short enough that a forgotten session does not hang the model forever.
const humanWait = 3 * time.Minute

// waitForHuman polls until the page stops being a challenge. It is the other
// half of the handoff: the user solves the check in the window they were
// already watching, and this is what notices they finished.
func waitForHuman(s browserSession) (string, error) {
	deadline := time.Now().Add(humanWait)
	for {
		c := pageChallenge(s)
		if c == nil {
			browserPacer.relax(browserPacer.lastHost())
			page, err := readPage(s, "text")
			if err != nil {
				return "", err
			}
			return "The challenge cleared.\n\n" + page, nil
		}
		if time.Now().After(deadline) {
			return "", fmt.Errorf("waited %s and the page is still a verification challenge (matched %q). "+
				"Ask the user whether they solved it, whether to keep waiting, or whether to try another way in",
				humanWait, c.Why)
		}
		time.Sleep(2 * time.Second)
	}
}
