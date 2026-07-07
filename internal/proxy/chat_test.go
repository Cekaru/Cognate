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
	"github.com/kaanrumin/polyglot-cache/internal/tenant"
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
	return newTestHandlerTenancy(t, upstream, Tenancy{})
}

func newTestHandlerTenancy(t *testing.T, upstream http.Handler, tenancy Tenancy) http.Handler {
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
	return New(prov, eng, tenancy, slog.New(slog.NewTextHandler(io.Discard, nil)))
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

// TestIsolationOptOut: an isolated request neither reads the shared cache nor
// leaks its own entries into it.
func TestIsolationOptOut(t *testing.T) {
	var upstreamCalls int
	h := newTestHandler(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"Paris"}}]}`)
	}))

	// Alice seeds the shared cache with the Spanish prompt.
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(esBody))
	r.Header.Set("Authorization", "Bearer sk-alice")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Header().Get("X-Polyglot-Cache") != "MISS" {
		t.Fatalf("seed = %q; want MISS", w.Header().Get("X-Polyglot-Cache"))
	}

	// Bob asks the equivalent Turkish prompt but opts out of sharing: the
	// shared entry must be invisible to him.
	r = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(trBody))
	r.Header.Set("Authorization", "Bearer sk-bob")
	r.Header.Set("X-Polyglot-Isolation", "tenant")
	w = httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Header().Get("X-Polyglot-Cache") != "MISS" {
		t.Fatalf("isolated request = %q; must not read the shared cache", w.Header().Get("X-Polyglot-Cache"))
	}
	if upstreamCalls != 2 {
		t.Fatalf("upstream calls = %d; want 2", upstreamCalls)
	}

	// A shared-scope Turkish request still hits the shared entry.
	tr := post(t, h, trBody)
	if tr.Header().Get("X-Polyglot-Cache") != "L2" {
		t.Fatalf("shared request = %q; want L2", tr.Header().Get("X-Polyglot-Cache"))
	}
	if upstreamCalls != 2 {
		t.Fatalf("upstream calls = %d; want 2 (shared hit served from cache)", upstreamCalls)
	}
}

// TestTenantRateLimit: a tenant that exhausts its byte budget gets 429; other
// tenants are unaffected.
func TestTenantRateLimit(t *testing.T) {
	h := newTestHandlerTenancy(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"Paris"}}]}`)
	}), Tenancy{Quotas: tenant.NewQuotas(tenant.Config{RequestBytesPerSec: 1, RequestBurst: len(esBody)})})

	send := func(key string) int {
		r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(esBody))
		r.Header.Set("Content-Type", "application/json")
		r.Header.Set("Authorization", "Bearer "+key)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		return w.Code
	}

	if code := send("sk-alice"); code != 200 {
		t.Fatalf("first request = %d; want 200", code)
	}
	if code := send("sk-alice"); code != http.StatusTooManyRequests {
		t.Fatalf("second request = %d; want 429 (burst spent)", code)
	}
	if code := send("sk-bob"); code != 200 {
		t.Fatalf("other tenant = %d; want 200 (independent bucket)", code)
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
