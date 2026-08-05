package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"

	"golang.org/x/oauth2"
)

// storedAuth is everything needed to keep talking to an OAuth-protected MCP
// server across restarts: the token itself plus the endpoint configuration
// that was discovered during the first authorization. Without the endpoint
// config we could not exchange a refresh token on the next launch.
type storedAuth struct {
	ClientID     string        `json:"client_id"`
	ClientSecret string        `json:"client_secret,omitempty"`
	AuthURL      string        `json:"auth_url,omitempty"`
	TokenURL     string        `json:"token_url"`
	RedirectURL  string        `json:"redirect_url,omitempty"`
	Scopes       []string      `json:"scopes,omitempty"`
	Token        *oauth2.Token `json:"token"`
}

func (s *storedAuth) oauthConfig() *oauth2.Config {
	return &oauth2.Config{
		ClientID:     s.ClientID,
		ClientSecret: s.ClientSecret,
		RedirectURL:  s.RedirectURL,
		Scopes:       s.Scopes,
		Endpoint: oauth2.Endpoint{
			AuthURL:  s.AuthURL,
			TokenURL: s.TokenURL,
		},
	}
}

// mcpAuthStore is the on-disk credential file, keyed by server name.
type mcpAuthStore struct {
	Servers map[string]*storedAuth `json:"servers"`
}

// authFileMu serializes read-modify-write cycles on the credential file.
var authFileMu sync.Mutex

// mcpAuthPath is the credential file. It is written 0600 inside the atlas
// data dir. Note this is a permission-protected file, not an OS keychain —
// anything running as this user can read it, same as ~/.aws/credentials or
// gh's hosts.yml.
func mcpAuthPath() (string, error) {
	base, err := atlasDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(base, "mcp-auth.json"), nil
}

func readAuthStore() (*mcpAuthStore, error) {
	st := &mcpAuthStore{Servers: map[string]*storedAuth{}}
	p, err := mcpAuthPath()
	if err != nil {
		return st, err
	}
	data, err := os.ReadFile(p)
	if err != nil {
		if os.IsNotExist(err) {
			return st, nil
		}
		return st, err
	}
	if err := json.Unmarshal(data, st); err != nil {
		return st, fmt.Errorf("parse %s: %w", p, err)
	}
	if st.Servers == nil {
		st.Servers = map[string]*storedAuth{}
	}
	return st, nil
}

func writeAuthStore(st *mcpAuthStore) error {
	p, err := mcpAuthPath()
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return err
	}
	// Write via a 0600 temp file and rename so a crash can't leave a
	// half-written credential file, and so the secret is never briefly
	// world-readable.
	tmp := p + ".tmp"
	if err := os.WriteFile(tmp, data, 0600); err != nil {
		return err
	}
	if err := os.Rename(tmp, p); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return os.Chmod(p, 0600)
}

// saveMCPAuth records the endpoint config and token for a server.
func saveMCPAuth(name string, oc *oauth2.Config, tok *oauth2.Token) {
	authFileMu.Lock()
	defer authFileMu.Unlock()
	st, err := readAuthStore()
	if err != nil {
		log.Printf("mcp oauth: reading credential store: %v", err)
	}
	st.Servers[name] = &storedAuth{
		ClientID:     oc.ClientID,
		ClientSecret: oc.ClientSecret,
		AuthURL:      oc.Endpoint.AuthURL,
		TokenURL:     oc.Endpoint.TokenURL,
		RedirectURL:  oc.RedirectURL,
		Scopes:       oc.Scopes,
		Token:        tok,
	}
	if err := writeAuthStore(st); err != nil {
		log.Printf("mcp oauth: saving credentials for %q: %v", name, err)
	}
}

// updateMCPToken replaces just the token for a server, preserving the
// endpoint config recorded at authorization time.
func updateMCPToken(name string, tok *oauth2.Token) {
	authFileMu.Lock()
	defer authFileMu.Unlock()
	st, err := readAuthStore()
	if err != nil {
		log.Printf("mcp oauth: reading credential store: %v", err)
		return
	}
	sa, ok := st.Servers[name]
	if !ok {
		// Nothing recorded yet — a token with no endpoint config can't be
		// refreshed later, so there's no point storing it alone.
		return
	}
	sa.Token = tok
	if err := writeAuthStore(st); err != nil {
		log.Printf("mcp oauth: updating token for %q: %v", name, err)
	}
}

func loadMCPAuth(name string) (*storedAuth, error) {
	authFileMu.Lock()
	defer authFileMu.Unlock()
	st, err := readAuthStore()
	if err != nil {
		return nil, err
	}
	return st.Servers[name], nil
}

// deleteMCPAuth forgets a server's stored credentials. Reports whether
// anything was actually removed.
func deleteMCPAuth(name string) (bool, error) {
	authFileMu.Lock()
	defer authFileMu.Unlock()
	st, err := readAuthStore()
	if err != nil {
		return false, err
	}
	if _, ok := st.Servers[name]; !ok {
		return false, nil
	}
	delete(st.Servers, name)
	return true, writeAuthStore(st)
}
