// Package semantic implements the L2 cache tier: similarity search over BGE-M3
// query embeddings, gated by a per-language-pair threshold and (in Phase 2) the
// structural token guard. Implemented in Phases 1–2 (ROADMAP.md §6, §9).
//
// This file is the in-memory index: an exhaustive cosine scan, dependency-free
// for tests and single-process runs. The pgvector-backed index (persistent,
// ANN) in postgres.go implements the same Index surface, so selecting a backend
// is a wiring choice in main.go and nothing in the engine changes.
package semantic

import (
	"context"
	"math"
	"sync"
	"time"
)

// Entry is one stored prompt/response record in the L2 index. The embedding is
// the query vector in the prompt's ORIGINAL language (ROADMAP.md §1).
type Entry struct {
	Key         string
	TenantScope string    // "shared" or a tenant hash
	Model       string    // matches are never served across models
	Embedding   []float32 // BGE-M3, unit-relevant; cosine handles non-unit norm
	Response    []byte    // the cached upstream response body
	Lang        string    // language of the seeding prompt
	CreatedAt   time.Time
}

// Index is the L2 vector-store surface the engine depends on. The in-memory and
// pgvector backends both satisfy it, so persistence is a wiring swap in main.go
// with no engine change (ROADMAP.md §6). Both methods take a context and return
// an error so the persistent backend can honor request cancellation and surface
// store failures; the engine degrades to passthrough on either (never fails a
// request on a cache-path error).
type Index interface {
	// Add stores an entry. A store failure must not corrupt the index.
	Add(ctx context.Context, e *Entry) error
	// Search returns the highest-cosine entry matching model and scopes, its
	// similarity in [-1, 1], or (nil, 0, nil) when no candidate exists.
	Search(ctx context.Context, vec []float32, model string, scopes ...string) (*Entry, float64, error)
}

// MemoryIndex is an in-memory, exhaustive-scan vector index. Safe for
// concurrent use.
type MemoryIndex struct {
	mu      sync.RWMutex
	entries []*Entry
}

var _ Index = (*MemoryIndex)(nil)

// NewMemoryIndex returns an empty in-memory index.
func NewMemoryIndex() *MemoryIndex {
	return &MemoryIndex{}
}

// Add appends an entry to the index. The context is unused (the in-memory store
// never blocks) but kept to satisfy Index. It never returns an error.
func (ix *MemoryIndex) Add(_ context.Context, e *Entry) error {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	ix.entries = append(ix.entries, e)
	return nil
}

// Len reports the number of stored entries.
func (ix *MemoryIndex) Len() int {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return len(ix.entries)
}

// Search returns the highest-cosine entry whose Model equals model and whose
// TenantScope is in scopes, along with its similarity in [-1, 1]. It returns
// (nil, 0, nil) when no candidate exists. Threshold gating is the caller's job —
// the index reports the best match and its score; policy lives in the engine.
// The context is unused and no error is ever returned; both exist to satisfy
// Index alongside the pgvector backend.
func (ix *MemoryIndex) Search(_ context.Context, vec []float32, model string, scopes ...string) (*Entry, float64, error) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()

	var (
		best    *Entry
		bestSim = math.Inf(-1)
		qNorm   = norm(vec)
	)
	if qNorm == 0 {
		return nil, 0, nil
	}
	for _, e := range ix.entries {
		if e.Model != model || !scopeAllowed(e.TenantScope, scopes) {
			continue
		}
		sim := cosineWithQueryNorm(vec, qNorm, e.Embedding)
		if sim > bestSim {
			bestSim, best = sim, e
		}
	}
	if best == nil {
		return nil, 0, nil
	}
	return best, bestSim, nil
}

func scopeAllowed(scope string, allowed []string) bool {
	if len(allowed) == 0 {
		return true
	}
	for _, a := range allowed {
		if scope == a {
			return true
		}
	}
	return false
}

// cosineWithQueryNorm computes cosine similarity given a precomputed norm for
// the query vector. Returns 0 for mismatched lengths or a zero candidate.
func cosineWithQueryNorm(q []float32, qNorm float64, e []float32) float64 {
	if len(q) != len(e) {
		return 0
	}
	var dot, eSum float64
	for i := range q {
		dot += float64(q[i]) * float64(e[i])
		eSum += float64(e[i]) * float64(e[i])
	}
	eNorm := math.Sqrt(eSum)
	if eNorm == 0 {
		return 0
	}
	return dot / (qNorm * eNorm)
}

func norm(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return math.Sqrt(s)
}
