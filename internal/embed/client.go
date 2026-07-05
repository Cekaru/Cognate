// Package embed is the Go client for the BGE-M3 Python embedding sidecar.
// The proxy never runs the model in-process; it calls the sidecar over
// localhost/HTTP.
package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/kaanrumin/polyglot-cache/internal/cache"
)

// Dim is the BGE-M3 output dimensionality.
const Dim = cache.EmbeddingDim

// maxRespBytes caps the sidecar response we will read, so a misbehaving
// sidecar can't exhaust proxy memory.
const maxRespBytes = 64 << 20 // 64 MiB

// Client talks to the embedding sidecar.
type Client struct {
	baseURL string
	http    *http.Client
}

// Option customizes a Client.
type Option func(*Client)

// WithHTTPClient overrides the underlying HTTP client (e.g. in tests).
func WithHTTPClient(h *http.Client) Option {
	return func(c *Client) { c.http = h }
}

// NewClient returns a sidecar client pointed at baseURL (e.g. http://sidecar:8000).
func NewClient(baseURL string, opts ...Option) *Client {
	c := &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		http:    &http.Client{Timeout: 30 * time.Second},
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

type embedRequest struct {
	Texts []string `json:"texts"`
}

type embedResponse struct {
	Model      string      `json:"model"`
	Dim        int         `json:"dim"`
	Embeddings [][]float32 `json:"embeddings"`
}

// Embed returns BGE-M3 embeddings for the given texts, each in its original
// language (no translate-then-embed). The returned slice is
// index-aligned with texts and every vector has length Dim.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	payload, err := json.Marshal(embedRequest{Texts: texts})
	if err != nil {
		return nil, fmt.Errorf("embed: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/embed", bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("embed: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embed: call sidecar: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxRespBytes))
	if err != nil {
		return nil, fmt.Errorf("embed: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embed: sidecar returned %d: %s", resp.StatusCode, snippet(body))
	}

	var out embedResponse
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, fmt.Errorf("embed: decode response: %w", err)
	}
	if len(out.Embeddings) != len(texts) {
		return nil, fmt.Errorf("embed: sidecar returned %d embeddings for %d texts", len(out.Embeddings), len(texts))
	}
	for i, vec := range out.Embeddings {
		if len(vec) != Dim {
			return nil, fmt.Errorf("embed: vector %d has dim %d, want %d", i, len(vec), Dim)
		}
	}
	return out.Embeddings, nil
}

func snippet(b []byte) string {
	const max = 256
	if len(b) > max {
		return string(b[:max]) + "…"
	}
	return string(b)
}
