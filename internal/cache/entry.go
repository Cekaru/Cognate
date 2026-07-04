// Package cache defines the shared cache-entry shape used by the L1 (exact)
// and L2 (semantic) tiers. See ROADMAP.md §3.
//
// Core principle: the query embedding is stored in the prompt's ORIGINAL
// language (no translate-then-embed). Responses are translated lazily, per
// language, on the first hit in that language.
package cache

import "time"

// TenantScopeShared marks an entry as visible to all tenants (cross-tenant
// sharing is where the cross-lingual hit rate lives; see ROADMAP.md §8).
const TenantScopeShared = "shared"

// EmbeddingDim is the BGE-M3 output dimensionality.
const EmbeddingDim = 1024

// CacheEntry is the record stored for a cached prompt/response pair.
type CacheEntry struct {
	ID               string
	TenantScope      string            // "shared" or a tenant hash
	QueryEmbedding   []float32         // BGE-M3, EmbeddingDim floats, original language
	CanonicalLang    string            // language the LLM answered in
	CanonicalText    string            // encrypted at rest
	Responses        map[string]string // lazily populated per language; encrypted at rest
	StructuralTokens StructuralTokens
	CreatedAt        time.Time
	TTL              time.Duration
	HitCount         int64
	LastSimilarity   float64
}

// StructuralTokens holds the locale-normalized structural elements the guard
// compares across a cross-lingual match, to reject numeric/ID near-misses that
// the embedding cannot separate. See ROADMAP.md §6 (Phase 2b).
type StructuralTokens struct {
	Numbers    []string
	IDs        []string
	Currencies []string
	Dates      []string
	CodeIdents []string
}
