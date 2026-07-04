// Package provider abstracts the upstream LLM API. The interface exists from
// day one (ROADMAP.md §4) so additional backends can be added without
// touching the proxy; Phase 0 ships a single OpenAI-compatible implementation.
package provider

import (
	"errors"
	"net/http"
	"net/url"
	"strings"
)

// Provider abstracts an upstream, OpenAI-format LLM API.
type Provider interface {
	// Target returns the upstream base URL requests are forwarded to.
	Target() *url.URL
	// Authorize stamps upstream credentials onto an outbound request.
	Authorize(r *http.Request)
	// Name identifies the provider in logs (never includes secrets).
	Name() string
}

// OpenAICompatible forwards to any OpenAI-format endpoint (OpenAI, Together,
// Groq, a local vLLM, etc.) using a bearer token.
type OpenAICompatible struct {
	base   *url.URL
	apiKey string
}

// NewOpenAICompatible builds a provider pointed at baseURL. The API key may be
// empty (e.g. a keyless local endpoint), in which case no Authorization header
// is added.
func NewOpenAICompatible(baseURL, apiKey string) (*OpenAICompatible, error) {
	if baseURL == "" {
		return nil, errors.New("provider base URL is required")
	}
	u, err := url.Parse(strings.TrimRight(baseURL, "/"))
	if err != nil {
		return nil, err
	}
	if u.Scheme == "" || u.Host == "" {
		return nil, errors.New("provider base URL must be absolute (scheme + host)")
	}
	return &OpenAICompatible{base: u, apiKey: apiKey}, nil
}

func (p *OpenAICompatible) Target() *url.URL { return p.base }

func (p *OpenAICompatible) Name() string { return "openai-compatible" }

func (p *OpenAICompatible) Authorize(r *http.Request) {
	if p.apiKey != "" {
		r.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
}
