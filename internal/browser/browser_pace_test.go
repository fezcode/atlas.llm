package browser

import (
	"strings"
	"testing"
	"time"
)

// testPacer is a pacer with the clock and the jitter pinned, so the policy can
// be asserted exactly.
func testPacer(now time.Time) *pacer {
	p := newPacer()
	p.now = func() time.Time { return now }
	p.sleep = func(time.Duration) {}
	p.jitter = func() time.Duration { return 0 }
	return p
}

func TestHostOf(t *testing.T) {
	cases := map[string]string{
		"https://example.com/a?b=c": "example.com",
		"http://EXAMPLE.com:8080/":  "example.com",
		"https://sub.example.co.uk": "sub.example.co.uk",
		"about:blank":               "",
		"file:///tmp/x.html":        "",
		"":                          "",
		"not a url":                 "",
	}
	for in, want := range cases {
		if got := hostOf(in); got != want {
			t.Errorf("hostOf(%q) = %q, want %q", in, got, want)
		}
	}
}

// The first visit to a host is never delayed — pacing is a cost we impose on
// repetition, not on arriving.
func TestPacerFirstVisitIsFree(t *testing.T) {
	now := time.Now()
	p := testPacer(now)
	if d := p.delay("example.com", now); d != 0 {
		t.Errorf("first visit delay = %v, want 0", d)
	}
	if d := p.delay("", now); d != 0 {
		t.Errorf("empty host delay = %v, want 0", d)
	}
}

func TestPacerSpacesRepeatVisits(t *testing.T) {
	now := time.Now()
	p := testPacer(now)
	p.pace("example.com")

	// Straight back to the same host: hold for the full base gap.
	if d := p.delay("example.com", now); d != paceBase {
		t.Errorf("immediate revisit delay = %v, want %v", d, paceBase)
	}
	// Half the gap has already elapsed, so only the remainder is owed.
	half := now.Add(paceBase / 2)
	if d := p.delay("example.com", half); d != paceBase/2 {
		t.Errorf("half-elapsed delay = %v, want %v", d, paceBase/2)
	}
	// Long enough later, nothing is owed.
	if d := p.delay("example.com", now.Add(2*paceBase)); d != 0 {
		t.Errorf("settled delay = %v, want 0", d)
	}
	// A different host is unaffected: the budget is per host.
	if d := p.delay("other.example", now); d != 0 {
		t.Errorf("other host delay = %v, want 0", d)
	}
}

func TestPacerPushBackWidensAndCaps(t *testing.T) {
	now := time.Now()
	p := testPacer(now)
	p.pace("example.com")

	p.pushBack("example.com")
	want := paceBase*2 + paceBackoffAdd
	if d := p.delay("example.com", now); d != want {
		t.Errorf("delay after one push-back = %v, want %v", d, want)
	}

	// Repeated push-back keeps widening, but never past the ceiling.
	for range 10 {
		p.pushBack("example.com")
	}
	if d := p.delay("example.com", now); d != paceGapMax {
		t.Errorf("delay after many push-backs = %v, want the %v cap", d, paceGapMax)
	}
}

func TestPacerRelaxDecaysThePenalty(t *testing.T) {
	now := time.Now()
	p := testPacer(now)
	p.pace("example.com")
	p.pushBack("example.com")

	penalised := p.delay("example.com", now)
	p.relax("example.com")
	relaxed := p.delay("example.com", now)
	if relaxed >= penalised {
		t.Errorf("relax did not shrink the gap: %v then %v", penalised, relaxed)
	}

	// Relaxing repeatedly returns the host to the plain base gap and stops.
	for range 10 {
		p.relax("example.com")
	}
	if d := p.delay("example.com", now); d != paceBase {
		t.Errorf("fully relaxed delay = %v, want the %v base", d, paceBase)
	}
}

func TestPacerTracksLastHost(t *testing.T) {
	p := testPacer(time.Now())
	if p.lastHost() != "" {
		t.Errorf("lastHost on a new pacer = %q, want empty", p.lastHost())
	}
	p.pace("example.com")
	p.pace("other.example")
	if got := p.lastHost(); got != "other.example" {
		t.Errorf("lastHost = %q, want other.example", got)
	}
	// A non-http destination leaves the previous host in place: about:blank
	// is not something we can be impolite to.
	p.pace("")
	if got := p.lastHost(); got != "other.example" {
		t.Errorf("lastHost after a blank pace = %q, want other.example", got)
	}
}

func TestDetectChallengeFindsInterstitials(t *testing.T) {
	cases := []struct {
		name   string
		title  string
		text   string
		assets []string
		kind   string
	}{
		{"cloudflare interstitial", "Just a moment...", "Enable JavaScript and cookies to continue", nil, "captcha"},
		{"turnstile widget", "Login", "", []string{"https://challenges.cloudflare.com/turnstile/v0/api.js"}, "captcha"},
		{"recaptcha widget", "Sign in", "", []string{"https://www.gstatic.com/recaptcha/releases/abc/recaptcha.js"}, "captcha"},
		{"google sorry page", "https://www.google.com/sorry/index", "Our systems have detected unusual traffic from your computer network.", nil, "captcha"},
		{"hcaptcha checkbox", "Verify", "Verify you are human by completing the action below.", nil, "captcha"},
		{"datadome", "Blocked", "", []string{"https://geo.captcha-delivery.com/captcha/?initialCid=x"}, "captcha"},
		{"http 429", "429 Too Many Requests", "Too many requests, please try again later.", nil, "ratelimit"},
		{"cloudflare 1015", "Access denied", "You are being rate limited. Error 1015", nil, "ratelimit"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := detectChallenge(c.title, c.text, c.assets)
			if got == nil {
				t.Fatalf("detectChallenge returned nil, want a %s", c.kind)
			}
			if got.Kind != c.kind {
				t.Errorf("Kind = %q, want %q", got.Kind, c.kind)
			}
			if got.Why == "" {
				t.Error("Why should name the signature that matched")
			}
		})
	}
}

// A false positive stalls a page that was working, so ordinary pages — and
// especially pages that merely talk about captchas and rate limits — must come
// back clean.
func TestDetectChallengeLeavesOrdinaryPagesAlone(t *testing.T) {
	cases := []struct {
		name   string
		title  string
		text   string
		assets []string
	}{
		{"plain article", "Go 1.25 release notes", "The Go team is pleased to announce the release.", []string{"https://go.dev/js/site.js"}},
		{"the phrase in prose", "Waiting well", "He said he would be along in just a moment, so we waited by the door.", nil},
		{"docs about rate limits", "API guide", "The endpoint allows 100 requests per minute; handle HTTP 429 responses with backoff.", nil},
		{"empty page", "", "", nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := detectChallenge(c.title, c.text, c.assets); got != nil {
				t.Errorf("detectChallenge flagged an ordinary page as %s (matched %q)", got.Kind, got.Why)
			}
		})
	}
}

// A rate limit and a human check need opposite responses, so each message has
// to name the right recovery.
func TestChallengeMessages(t *testing.T) {
	captcha := (&challenge{Kind: "captcha", Why: "verify you are human"}).message("example.com")
	if !strings.Contains(captcha, "example.com") || !strings.Contains(captcha, "wait_human") ||
		!strings.Contains(captcha, "ask them") {
		t.Errorf("captcha message should name the host and hand off to the user, got: %s", captcha)
	}
	rate := (&challenge{Kind: "ratelimit", Why: "too many requests"}).message("example.com")
	if !strings.Contains(rate, "rate-limiting") || !strings.Contains(rate, "Do not reload") {
		t.Errorf("ratelimit message should say to stop retrying, got: %s", rate)
	}
	if strings.Contains(rate, "wait_human") {
		t.Error("a rate limit is not something a human can solve in the window")
	}
	// With no host known the message still has to read as a sentence.
	if got := (&challenge{Kind: "captcha", Why: "x"}).message(""); !strings.HasPrefix(got, "This site") {
		t.Errorf("message with no host = %q, want it to start with a stand-in subject", got)
	}
}

func TestChallengeProbeJS(t *testing.T) {
	js := challengeProbeJS()
	for _, want := range []string{"document.title", "innerText", "iframe[src]", "script[src]", "JSON.stringify"} {
		if !strings.Contains(js, want) {
			t.Errorf("challengeProbeJS should collect %q", want)
		}
	}
}
