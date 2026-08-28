package engine

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"testing"

	"github.com/kaanrumin/polyglot-cache/internal/cache"
	"github.com/kaanrumin/polyglot-cache/internal/cache/semantic"
	"github.com/kaanrumin/polyglot-cache/internal/crypto"
)

// fakeEmbedder maps known prompt text to fixed vectors and counts calls, so a
// test can assert both cross-lingual matching and that a prompt is embedded at
// most once per request.
type fakeEmbedder struct {
	vecs  map[string][]float32
	calls int
	err   error
}

func (f *fakeEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	f.calls++
	if f.err != nil {
		return nil, f.err
	}
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = f.vecs[t]
	}
	return out, nil
}

func semVec(primary int, jitter float32) []float32 {
	v := make([]float32, 8)
	v[primary] = 1
	v[(primary+1)%8] = jitter
	return v
}

// A Spanish and a Turkish "capital of France" prompt sit next to each other in
// vector space (the whole thesis); an English "weather" prompt is orthogonal.
const (
	esCapital = "¿Cuál es la capital de Francia?"
	trCapital = "Fransa'nın başkenti neresidir?"
	enWeather = "What is the weather today?"
)

func newTestEngine(t *testing.T, emb Embedder, threshold float64) *Engine {
	t.Helper()
	return New(emb, Config{L1Capacity: 100, Threshold: threshold}, slog.New(slog.NewTextHandler(io.Discard, nil)))
}

func okUpstream(body string) (*UpstreamResp, error) {
	return &UpstreamResp{Status: http.StatusOK, Body: []byte(body)}, nil
}

func query(text, lang string) Query {
	return Query{
		TenantScope: cache.TenantScopeShared,
		Model:       "gpt-test",
		ExactText:   "user\x1f" + text,
		EmbedText:   text,
		Lang:        lang,
	}
}

// TestCrossLingualHit is the thesis in a unit test: a Spanish prompt
// seeds the cache; an equivalent Turkish prompt is served from that entry.
func TestCrossLingualHit(t *testing.T) {
	emb := &fakeEmbedder{vecs: map[string][]float32{
		esCapital: semVec(0, 0.0),
		trCapital: semVec(0, 0.05), // ~0.999 cosine with the ES vector
		enWeather: semVec(3, 0.0),  // orthogonal
	}}
	e := newTestEngine(t, emb, 0.85)
	ctx := context.Background()

	var upstreamCalls int
	up := func(body string) UpstreamFunc {
		return func(context.Context) (*UpstreamResp, error) {
			upstreamCalls++
			return okUpstream(body)
		}
	}

	// 1) Spanish prompt: full miss, calls upstream, stores.
	r1, err := e.Serve(ctx, query(esCapital, "es"), up("Paris (from ES)"))
	if err != nil {
		t.Fatal(err)
	}
	if r1.Tier != "MISS" {
		t.Fatalf("first ES request tier = %q; want MISS", r1.Tier)
	}

	// 2) Turkish equivalent: L2 cross-lingual hit, no upstream call.
	r2, err := e.Serve(ctx, query(trCapital, "tr"), up("should not be called"))
	if err != nil {
		t.Fatal(err)
	}
	if r2.Tier != "L2" {
		t.Fatalf("TR request tier = %q; want L2", r2.Tier)
	}
	if string(r2.Body) != "Paris (from ES)" {
		t.Fatalf("TR served body = %q; want the ES-seeded answer", r2.Body)
	}
	if r2.PromptLang != "tr" || r2.EntryLang != "es" {
		t.Fatalf("langs = %q/%q; want tr/es", r2.PromptLang, r2.EntryLang)
	}
	if r2.Similarity < 0.85 {
		t.Fatalf("similarity = %v; want >= threshold", r2.Similarity)
	}
	if upstreamCalls != 1 {
		t.Fatalf("upstream called %d times; want 1 (TR served from cache)", upstreamCalls)
	}

	// 3) Unrelated English prompt: full miss, upstream called again.
	r3, err := e.Serve(ctx, query(enWeather, "en"), up("Sunny"))
	if err != nil {
		t.Fatal(err)
	}
	if r3.Tier != "MISS" {
		t.Fatalf("EN request tier = %q; want MISS", r3.Tier)
	}
	if upstreamCalls != 2 {
		t.Fatalf("upstream called %d times; want 2", upstreamCalls)
	}
}

// TestL1ExactHitSkipsEmbedding proves an identical repeat is served from L1
// without touching the embedder.
func TestL1ExactHitSkipsEmbedding(t *testing.T) {
	emb := &fakeEmbedder{vecs: map[string][]float32{esCapital: semVec(0, 0)}}
	e := newTestEngine(t, emb, 0.85)
	ctx := context.Background()

	if _, err := e.Serve(ctx, query(esCapital, "es"), func(context.Context) (*UpstreamResp, error) {
		return okUpstream("Paris")
	}); err != nil {
		t.Fatal(err)
	}
	callsAfterMiss := emb.calls // one embed for the seeding miss

	r, err := e.Serve(ctx, query(esCapital, "es"), func(context.Context) (*UpstreamResp, error) {
		t.Fatal("upstream must not be called on an L1 hit")
		return nil, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if r.Tier != "L1" {
		t.Fatalf("tier = %q; want L1", r.Tier)
	}
	if emb.calls != callsAfterMiss {
		t.Fatalf("embedder called on L1 hit (%d -> %d); L1 must short-circuit before embedding", callsAfterMiss, emb.calls)
	}
}

// TestThresholdRejectsLooseMatch: a near-but-below-threshold neighbor must not
// be served; the request falls through to upstream.
func TestThresholdRejectsLooseMatch(t *testing.T) {
	emb := &fakeEmbedder{vecs: map[string][]float32{
		esCapital: semVec(0, 0.0),
		trCapital: semVec(0, 0.05),
	}}
	e := newTestEngine(t, emb, 0.9999) // stricter than the ~0.999 neighbor
	ctx := context.Background()

	if _, err := e.Serve(ctx, query(esCapital, "es"), func(context.Context) (*UpstreamResp, error) {
		return okUpstream("Paris")
	}); err != nil {
		t.Fatal(err)
	}

	upstreamCalled := false
	r, err := e.Serve(ctx, query(trCapital, "tr"), func(context.Context) (*UpstreamResp, error) {
		upstreamCalled = true
		return okUpstream("Paris (TR)")
	})
	if err != nil {
		t.Fatal(err)
	}
	if r.Tier != "MISS" || !upstreamCalled {
		t.Fatalf("tier = %q, upstreamCalled = %v; want MISS with upstream called", r.Tier, upstreamCalled)
	}
}

// TestGuardVetoesNumericNearMiss is the Phase 2 milestone in a unit test: two
// prompts that differ only in amount ($100 vs $1000) sit at ~0.999 cosine —
// far above any workable threshold — and the structural guard is the only
// thing standing between them. The match must be rejected and the request must
// reach the real LLM.
func TestGuardVetoesNumericNearMiss(t *testing.T) {
	const (
		es100  = "Transfiere $100 a mi cuenta de ahorros"
		tr1000 = "Tasarruf hesabıma 1000 $ aktar"
	)
	emb := &fakeEmbedder{vecs: map[string][]float32{
		es100:  semVec(0, 0.0),
		tr1000: semVec(0, 0.05), // ~0.999 cosine: an embedding-level near-miss
	}}
	e := newTestEngine(t, emb, 0.85)
	ctx := context.Background()

	if _, err := e.Serve(ctx, query(es100, "es"), func(context.Context) (*UpstreamResp, error) {
		return okUpstream("transferred $100")
	}); err != nil {
		t.Fatal(err)
	}

	upstreamCalled := false
	r, err := e.Serve(ctx, query(tr1000, "tr"), func(context.Context) (*UpstreamResp, error) {
		upstreamCalled = true
		return okUpstream("transferred $1000")
	})
	if err != nil {
		t.Fatal(err)
	}
	if r.Tier != "MISS" || !upstreamCalled {
		t.Fatalf("tier = %q, upstreamCalled = %v; guard must veto the $100/$1000 near-miss", r.Tier, upstreamCalled)
	}
	if string(r.Body) != "transferred $1000" {
		t.Fatalf("body = %q; want the fresh upstream answer", r.Body)
	}
}

// TestGuardAdmitsMatchingTokens: identical structural tokens across languages
// must NOT fire the guard — the cross-lingual hit still works when the amounts
// agree.
func TestGuardAdmitsMatchingTokens(t *testing.T) {
	const (
		es100 = "Transfiere $100 a mi cuenta de ahorros"
		tr100 = "Tasarruf hesabıma 100 $ aktar"
	)
	emb := &fakeEmbedder{vecs: map[string][]float32{
		es100: semVec(0, 0.0),
		tr100: semVec(0, 0.05),
	}}
	e := newTestEngine(t, emb, 0.85)
	ctx := context.Background()

	if _, err := e.Serve(ctx, query(es100, "es"), func(context.Context) (*UpstreamResp, error) {
		return okUpstream("transferred $100")
	}); err != nil {
		t.Fatal(err)
	}

	r, err := e.Serve(ctx, query(tr100, "tr"), func(context.Context) (*UpstreamResp, error) {
		t.Fatal("upstream must not be called when tokens match")
		return nil, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if r.Tier != "L2" || string(r.Body) != "transferred $100" {
		t.Fatalf("tier = %q body = %q; want the ES-seeded L2 hit", r.Tier, r.Body)
	}
}

// TestPerPairThreshold: a pair-specific cutoff overrides the default in both
// directions of the language pair.
func TestPerPairThreshold(t *testing.T) {
	emb := &fakeEmbedder{vecs: map[string][]float32{
		esCapital: semVec(0, 0.0),
		trCapital: semVec(0, 0.05), // ~0.999 cosine
	}}
	e := New(emb, Config{
		L1Capacity: 100,
		Threshold:  0.85,
		Thresholds: &Thresholds{
			Default: 0.85,
			Pairs:   map[string]float64{PairKey("tr", "es"): 0.9999}, // stricter than the neighbor
		},
	}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	ctx := context.Background()

	if _, err := e.Serve(ctx, query(esCapital, "es"), func(context.Context) (*UpstreamResp, error) {
		return okUpstream("Paris")
	}); err != nil {
		t.Fatal(err)
	}

	upstreamCalled := false
	r, err := e.Serve(ctx, query(trCapital, "tr"), func(context.Context) (*UpstreamResp, error) {
		upstreamCalled = true
		return okUpstream("Paris (TR)")
	})
	if err != nil {
		t.Fatal(err)
	}
	if r.Tier != "MISS" || !upstreamCalled {
		t.Fatalf("tier = %q; the es-tr pair threshold 0.9999 must reject a ~0.999 match", r.Tier)
	}
}

// TestNon200NotCached: an upstream error response must not poison the cache.
func TestNon200NotCached(t *testing.T) {
	emb := &fakeEmbedder{vecs: map[string][]float32{esCapital: semVec(0, 0)}}
	e := newTestEngine(t, emb, 0.85)
	ctx := context.Background()

	calls := 0
	up := func(context.Context) (*UpstreamResp, error) {
		calls++
		return &UpstreamResp{Status: http.StatusInternalServerError, Body: []byte(`{"error":"boom"}`)}, nil
	}

	r1, _ := e.Serve(ctx, query(esCapital, "es"), up)
	if r1.Status != http.StatusInternalServerError {
		t.Fatalf("status = %d; want 500 relayed", r1.Status)
	}
	// Identical repeat must hit upstream again (nothing was cached).
	if _, err := e.Serve(ctx, query(esCapital, "es"), up); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("upstream called %d times; want 2 (500 not cached)", calls)
	}
}

// TestEmbedFailureDegradesToUpstream: a sidecar error must not fail the request.
func TestEmbedFailureDegradesToUpstream(t *testing.T) {
	emb := &fakeEmbedder{err: errors.New("sidecar down")}
	e := newTestEngine(t, emb, 0.85)

	r, err := e.Serve(context.Background(), query(esCapital, "es"), func(context.Context) (*UpstreamResp, error) {
		return okUpstream("Paris")
	})
	if err != nil {
		t.Fatalf("embed failure should not error the request: %v", err)
	}
	if r.Tier != "MISS" || string(r.Body) != "Paris" {
		t.Fatalf("got tier %q body %q; want MISS/Paris via upstream", r.Tier, r.Body)
	}
}

// TestEncryptionAtRest wires a real AES-256-GCM cipher and proves two things at
// once: the response persisted to L2 is ciphertext (the plaintext answer never
// appears in the stored bytes), and a cross-lingual hit still serves the correct
// decrypted plaintext.
func TestEncryptionAtRest(t *testing.T) {
	t.Setenv(crypto.DefaultKeyEnv, base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{0x07}, 32)))
	cipher, err := crypto.NewEncryptor(crypto.EnvKeyProvider{})
	if err != nil {
		t.Fatalf("NewEncryptor: %v", err)
	}

	idx := semantic.NewMemoryIndex()
	emb := &fakeEmbedder{vecs: map[string][]float32{
		esCapital: semVec(0, 0.0),
		trCapital: semVec(0, 0.05),
	}}
	e := New(emb, Config{L1Capacity: 100, Threshold: 0.85, L2: idx, Cipher: cipher},
		slog.New(slog.NewTextHandler(io.Discard, nil)))
	ctx := context.Background()

	const answer = "Paris (from ES)"
	if _, err := e.Serve(ctx, query(esCapital, "es"), func(context.Context) (*UpstreamResp, error) {
		return okUpstream(answer)
	}); err != nil {
		t.Fatal(err)
	}

	// The bytes at rest in L2 must be ciphertext, not the plaintext answer.
	ent, _, err := idx.Search(ctx, semVec(0, 0.0), "gpt-test", cache.TenantScopeShared)
	if err != nil {
		t.Fatal(err)
	}
	if ent == nil {
		t.Fatal("nothing stored in L2")
	}
	if bytes.Contains(ent.Response, []byte(answer)) {
		t.Fatal("stored L2 response contains the plaintext answer; not encrypted at rest")
	}
	if pt, err := cipher.Open(ent.Response); err != nil || string(pt) != answer {
		t.Fatalf("stored blob did not decrypt to the answer: err=%v pt=%q", err, pt)
	}

	// A cross-lingual hit must still serve the decrypted plaintext.
	r, err := e.Serve(ctx, query(trCapital, "tr"), func(context.Context) (*UpstreamResp, error) {
		t.Fatal("upstream must not be called on an encrypted L2 hit")
		return nil, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if r.Tier != "L2" || string(r.Body) != answer {
		t.Fatalf("tier = %q body = %q; want an L2 hit serving the decrypted plaintext", r.Tier, r.Body)
	}
}

// TestUpstreamErrorPropagates: a transport error surfaces to the caller.
func TestUpstreamErrorPropagates(t *testing.T) {
	emb := &fakeEmbedder{vecs: map[string][]float32{esCapital: semVec(0, 0)}}
	e := newTestEngine(t, emb, 0.85)

	_, err := e.Serve(context.Background(), query(esCapital, "es"), func(context.Context) (*UpstreamResp, error) {
		return nil, errors.New("dial tcp: refused")
	})
	if err == nil {
		t.Fatal("expected upstream transport error to propagate")
	}
}
