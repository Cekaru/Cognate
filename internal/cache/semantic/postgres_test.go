package semantic

import (
	"context"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/kaanrumin/polyglot-cache/internal/cache"
)

// TestPostgresIndex is an integration test against a real pgvector database. It
// is skipped unless POLYGLOT_TEST_DATABASE_URL points at one — the default
// `go test ./...` stays dependency-free. To run it against the compose stack:
//
//	docker compose up -d postgres
//	POLYGLOT_TEST_DATABASE_URL='postgres://polyglot:polyglot@localhost:5432/polyglot?sslmode=disable' \
//	  go test ./internal/cache/semantic -run TestPostgresIndex -v
func TestPostgresIndex(t *testing.T) {
	url := os.Getenv("POLYGLOT_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("set POLYGLOT_TEST_DATABASE_URL to run the pgvector integration test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	ix, err := NewPostgresIndex(ctx, url, slog.New(slog.NewTextHandler(os.Stderr, nil)))
	if err != nil {
		t.Fatalf("NewPostgresIndex: %v", err)
	}
	defer ix.Close()

	// Isolate this run: a dedicated model, wiped up front so reruns are clean.
	const model = "pgtest-model"
	if _, err := ix.pool.Exec(ctx, `DELETE FROM l2_entries WHERE model = $1`, model); err != nil {
		t.Fatalf("cleanup: %v", err)
	}

	// A Spanish-seeded entry and a near-parallel Turkish query vector (the
	// cross-lingual thesis), plus an orthogonal one that must not match.
	esVec := genVec(0, 0.0)
	trVec := genVec(0, 0.02) // ~0.9998 cosine with esVec
	orthoVec := genVec(500, 0.0)

	if err := ix.Add(ctx, &Entry{
		Key:         "es-capital",
		TenantScope: cache.TenantScopeShared,
		Model:       model,
		Embedding:   esVec,
		Response:    []byte("Paris (from ES)"),
		Lang:        "es",
		CreatedAt:   time.Now(),
	}); err != nil {
		t.Fatalf("Add: %v", err)
	}

	// The Turkish vector should retrieve the Spanish entry with high similarity.
	got, sim, err := ix.Search(ctx, trVec, model, cache.TenantScopeShared)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if got == nil || string(got.Response) != "Paris (from ES)" {
		t.Fatalf("cross-lingual search returned %v; want the ES-seeded entry", got)
	}
	if got.Lang != "es" {
		t.Fatalf("entry lang = %q; want es", got.Lang)
	}
	if sim < 0.99 {
		t.Fatalf("similarity = %v; want ~1 for near-parallel vectors", sim)
	}

	// A different model must not see the entry.
	if got, _, err := ix.Search(ctx, trVec, "other-model", cache.TenantScopeShared); err != nil || got != nil {
		t.Fatalf("Search crossed models: got=%v err=%v", got, err)
	}

	// The orthogonal vector still returns the only row (threshold gating is the
	// engine's job), but at a low similarity that the engine would reject.
	if _, sim, err := ix.Search(ctx, orthoVec, model, cache.TenantScopeShared); err != nil || sim > 0.5 {
		t.Fatalf("orthogonal query similarity = %v (err=%v); want low", sim, err)
	}

	if n, err := ix.Len(ctx); err != nil || n < 1 {
		t.Fatalf("Len = %d (err=%v); want >= 1", n, err)
	}
}

// genVec builds a BGE-M3-dimensioned vector with a 1 at primary and a small
// jitter in the next slot, so two calls with different jitters are near-parallel.
func genVec(primary int, jitter float32) []float32 {
	v := make([]float32, cache.EmbeddingDim)
	v[primary] = 1
	v[(primary+1)%cache.EmbeddingDim] = jitter
	return v
}
