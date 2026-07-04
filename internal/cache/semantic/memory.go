// Package semantic implements the L2 cache tier: similarity search over BGE-M3
// query embeddings, gated by a per-language-pair threshold and (in Phase 2) the
// structural token guard. Implemented in Phases 1–2 (ROADMAP.md §6, §9).
//
// This file is the Phase-1 in-memory index: an exhaustive cosine scan. It is
// correct and dependency-free for proving the thesis and for tests. The
// pgvector-backed index (persistent, ANN) implements the same Index behavior in
// a later Phase-1 step; keeping the surface small makes that swap mechanical.
package semantic

import (
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

// MemoryIndex is an in-memory, exhaustive-scan vector index. Safe for
// concurrent use.
type MemoryIndex struct {
	mu      sync.RWMutex
	entries []*Entry
}

// NewMemoryIndex returns an empty in-memory index.
func NewMemoryIndex() *MemoryIndex {
	return &MemoryIndex{}
}

// Add appends an entry to the index.
func (ix *MemoryIndex) Add(e *Entry) {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	ix.entries = append(ix.entries, e)
}

// Len reports the number of stored entries.
func (ix *MemoryIndex) Len() int {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return len(ix.entries)
}

// Search returns the highest-cosine entry whose Model equals model and whose
// TenantScope is in scopes, along with its similarity in [-1, 1]. It returns
// (nil, 0) when no candidate exists. Threshold gating is the caller's job — the
// index reports the best match and its score; policy lives in the engine.
func (ix *MemoryIndex) Search(vec []float32, model string, scopes ...string) (*Entry, float64) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()

	var (
		best    *Entry
		bestSim = math.Inf(-1)
		qNorm   = norm(vec)
	)
	if qNorm == 0 {
		return nil, 0
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
		return nil, 0
	}
	return best, bestSim
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
