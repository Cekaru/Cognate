package proxy

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/kaanrumin/polyglot-cache/internal/cache/engine"
	"github.com/kaanrumin/polyglot-cache/internal/provider"
)

// fakeEmbedder maps message content to fixed vectors for the HTTP-level test.
type fakeEmbedder struct{ vecs map[string][]float32 }

func (f *fakeEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = f.vecs[t]
	}
	return out, nil
}

func nearVec(jitter float32) []float32 { return []float32{1, jitter, 0, 0} }

const (
	esBody     = `{"model":"gpt-test","messages":[{"role":"user","content":"¿Cuál es la capital de Francia?"}]}`
	trBody     = `{"model":"gpt-test","messages":[{"role":"user","content":"Fransa'nın başkenti neresidir?"}]}`
	streamBody = `{"model":"gpt-test","stream":true,"messages":[{"role":"user","content":"¿Cuál es la capital de Francia?"}]}`
)

func newTestHandler(t *testing.T, upstream http.Handler) http.Handler {
	t.Helper()
	up := httptest.NewServer(upstream)
	t.Cleanup(up.Close)

	prov, err := provider.NewOpenAICompatible(up.URL, "upstream-key")
	if err != nil {
		t.Fatal(err)
	}
	emb := &fakeEmbedder{vecs: map[string][]float32{
		"¿Cuál es la capital de Francia?": nearVec(0.0),
		"Fransa'nın başkenti neresidir?":  nearVec(0.03),
	}}
	eng := engine.New(emb, engine.Config{L1Capacity: 100, Threshold: 0.85},
		slog.New(slog.NewTextHandler(io.Discard, nil)))
	return New(prov, eng, slog.New(slog.NewTextHandler(io.Discard, nil)))
}

func post(t *testing.T, h http.Handler, body string) *httptest.ResponseRecorder {
	t.Helper()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}

// TestChatCrossLingualHitOverHTTP drives the full handler: a Spanish request
// misses and hits upstream; the equivalent Turkish request is served from cache
// with the cross-lingual headers set.
func TestChatCrossLingualHitOverHTTP(t *testing.T) {
	var upstreamCalls int
	h := newTestHandler(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		if got := r.Header.Get("Authorization"); got != "Bearer upstream-key" {
			t.Errorf("upstream Authorization = %q; want the stamped provider key", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"Paris"}}]}`)
	}))

	// Spanish: miss.
	es := post(t, h, esBody)
	if es.Code != 200 || es.Header().Get("X-Polyglot-Cache") != "MISS" {
		t.Fatalf("ES: code=%d cache=%q; want 200/MISS", es.Code, es.Header().Get("X-Polyglot-Cache"))
	}

	// Turkish: cross-lingual L2 hit, no second upstream call.
	tr := post(t, h, trBody)
	if tr.Code != 200 || tr.Header().Get("X-Polyglot-Cache") != "L2" {
		t.Fatalf("TR: code=%d cache=%q; want 200/L2", tr.Code, tr.Header().Get("X-Polyglot-Cache"))
	}
	if !strings.Contains(tr.Body.String(), "Paris") {
		t.Fatalf("TR body = %q; want the ES-seeded answer", tr.Body.String())
	}
	if tr.Header().Get("X-Polyglot-Prompt-Lang") != "tr" || tr.Header().Get("X-Polyglot-Entry-Lang") != "es" {
		t.Fatalf("langs = %q/%q; want tr/es", tr.Header().Get("X-Polyglot-Prompt-Lang"), tr.Header().Get("X-Polyglot-Entry-Lang"))
	}
	if upstreamCalls != 1 {
		t.Fatalf("upstream called %d times; want 1", upstreamCalls)
	}
}

// TestStreamingFallsThrough: a stream=true request bypasses the cache and is
// proxied straight to the upstream.
func TestStreamingFallsThrough(t *testing.T) {
	var upstreamCalls int
	h := newTestHandler(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		_, _ = io.WriteString(w, "data: chunk\n\n")
	}))

	w := post(t, h, streamBody)
	if w.Code != 200 {
		t.Fatalf("stream: code=%d; want 200", w.Code)
	}
	if got := w.Header().Get("X-Polyglot-Cache"); got != "" {
		t.Fatalf("stream set cache header %q; streaming must bypass the cache", got)
	}
	if upstreamCalls != 1 {
		t.Fatalf("upstream called %d times; want 1 (passthrough)", upstreamCalls)
	}
}

// TestNonChatPathPassesThrough: other /v1 paths are never cached.
func TestNonChatPathPassesThrough(t *testing.T) {
	h := newTestHandler(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Errorf("unexpected upstream path %q", r.URL.Path)
		}
		_, _ = io.WriteString(w, `{"data":[]}`)
	}))

	r := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != 200 || w.Header().Get("X-Polyglot-Cache") != "" {
		t.Fatalf("models: code=%d cache=%q; want 200 with no cache header", w.Code, w.Header().Get("X-Polyglot-Cache"))
	}
}
