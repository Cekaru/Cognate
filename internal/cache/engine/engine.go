// Package engine wires the L1 (exact) and L2 (semantic) tiers into a single
// cache lookup path and owns the store-on-miss logic. It is the piece the proxy
// calls; keeping it HTTP-free makes the whole cross-lingual pipeline testable
// with a fake embedder and a fake upstream (ROADMAP.md §3, §6).
package engine

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"log/slog"
	"net/http"
	"time"

	"github.com/kaanrumin/polyglot-cache/internal/cache"
	"github.com/kaanrumin/polyglot-cache/internal/cache/exact"
	"github.com/kaanrumin/polyglot-cache/internal/cache/semantic"
)

// Embedder returns BGE-M3 embeddings for texts, index-aligned. *embed.Client
// satisfies this; tests supply a fake.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// Query is a normalized cache lookup request built from an OpenAI chat request.
type Query struct {
	TenantScope string // "shared" in Phase 1; per-tenant hash in Phase 2
	Model       string
	ExactText   string // canonical string hashed for the L1 key (roles + content)
	EmbedText   string // semantic content embedded for the L2 lookup
	Lang        string // detected language of this prompt
}

func (q Query) key() string {
	h := sha256.New()
	// Domain-separate the fields so different splits can't collide.
	h.Write([]byte(q.TenantScope))
	h.Write([]byte{0})
	h.Write([]byte(q.Model))
	h.Write([]byte{0})
	h.Write([]byte(q.ExactText))
	return hex.EncodeToString(h.Sum(nil))
}

// UpstreamResp is what an UpstreamFunc returns after calling the real LLM.
type UpstreamResp struct {
	Status int
	Header http.Header
	Body   []byte
}

// UpstreamFunc performs the real upstream call on a full cache miss.
type UpstreamFunc func(ctx context.Context) (*UpstreamResp, error)

// Result is the outcome the proxy writes back to the client.
type Result struct {
	Body       []byte
	Status     int
	Header     http.Header // upstream headers on a MISS; nil on a hit
	Tier       string      // "L1", "L2", or "MISS"
	Similarity float64     // set for L2 hits
	PromptLang string      // language of the incoming prompt
	EntryLang  string      // language of the entry that served the answer
}

// record is the L1 payload.
type record struct {
	response  []byte
	lang      string
	createdAt time.Time
}

// Engine is the cache orchestrator.
type Engine struct {
	l1        *exact.LRU[*record]
	l2        semantic.Index
	embedder  Embedder
	threshold float64       // global cross-lingual cutoff (Phase 1; per-pair in Phase 2)
	ttl       time.Duration // 0 = no expiry
	logger    *slog.Logger
	now       func() time.Time
}

// Config configures an Engine.
type Config struct {
	L1Capacity int
	Threshold  float64
	TTL        time.Duration
	// L2 is the semantic backend. When nil an in-memory index is used, so tests
	// and single-process runs need no database; main.go supplies the pgvector
	// backend when DATABASE_URL is set.
	L2 semantic.Index
}

// New builds an Engine. embedder may be nil, in which case the L2 tier is
// disabled and only the exact tier operates.
func New(embedder Embedder, cfg Config, logger *slog.Logger) *Engine {
	if logger == nil {
		logger = slog.Default()
	}
	l2 := cfg.L2
	if l2 == nil {
		l2 = semantic.NewMemoryIndex()
	}
	return &Engine{
		l1:        exact.New[*record](cfg.L1Capacity),
		l2:        l2,
		embedder:  embedder,
		threshold: cfg.Threshold,
		ttl:       cfg.TTL,
		logger:    logger,
		now:       time.Now,
	}
}

// Serve runs the full lookup: L1 exact, then L2 semantic, then (on a miss) the
// upstream call, storing a successful response in both tiers. The embedding
// computed for the L2 lookup is reused when storing, so a prompt is embedded at
// most once per request.
func (e *Engine) Serve(ctx context.Context, q Query, up UpstreamFunc) (*Result, error) {
	key := q.key()

	// L1 — exact hash.
	if rec, ok := e.l1.Get(key); ok && !e.expired(rec.createdAt) {
		e.logger.Info("cache hit", "tier", "L1", "model", q.Model, "prompt_lang", q.Lang)
		return &Result{Body: rec.response, Status: http.StatusOK, Tier: "L1", PromptLang: q.Lang, EntryLang: rec.lang}, nil
	}

	// Embed once (reused for the L2 lookup and, on a miss, for the L2 store).
	var vec []float32
	if e.embedder != nil {
		vecs, err := e.embedder.Embed(ctx, []string{q.EmbedText})
		if err != nil {
			// Embedding failure degrades to passthrough; never fail the request
			// on a cache-path error.
			e.logger.Warn("embed failed; bypassing L2", "err", err)
		} else if len(vecs) == 1 {
			vec = vecs[0]
		}
	}

	// L2 — semantic. Cross-tenant sharing is intentional (ROADMAP.md §8); the
	// shared scope is where cross-lingual hits live.
	if vec != nil {
		ent, sim, err := e.l2.Search(ctx, vec, q.Model, q.TenantScope, cache.TenantScopeShared)
		switch {
		case err != nil:
			// A store-level failure degrades to passthrough, like an embed error;
			// never fail the request on a cache-path error.
			e.logger.Warn("l2 search failed; bypassing L2", "err", err)
		case ent != nil && !e.expired(ent.CreatedAt):
			// Log every semantic lookup's score (ROADMAP.md §6 Phase 1).
			hit := sim >= e.threshold
			e.logger.Info("semantic lookup",
				"similarity", sim,
				"threshold", e.threshold,
				"hit", hit,
				"prompt_lang", q.Lang,
				"entry_lang", ent.Lang,
				"model", q.Model,
			)
			if hit {
				// Promote to L1 so an identical repeat skips embedding.
				e.l1.Put(key, &record{response: ent.Response, lang: ent.Lang, createdAt: e.now()})
				return &Result{
					Body:       ent.Response,
					Status:     http.StatusOK,
					Tier:       "L2",
					Similarity: sim,
					PromptLang: q.Lang,
					EntryLang:  ent.Lang,
				}, nil
			}
		default:
			e.logger.Debug("semantic lookup: no candidate", "model", q.Model, "prompt_lang", q.Lang)
		}
	}

	// Full miss — call upstream.
	resp, err := up(ctx)
	if err != nil {
		return nil, err
	}

	// Only cache a clean success.
	if resp.Status == http.StatusOK {
		now := e.now()
		e.l1.Put(key, &record{response: resp.Body, lang: q.Lang, createdAt: now})
		if vec != nil {
			if err := e.l2.Add(ctx, &semantic.Entry{
				Key:         key,
				TenantScope: q.TenantScope,
				Model:       q.Model,
				Embedding:   vec,
				Response:    resp.Body,
				Lang:        q.Lang,
				CreatedAt:   now,
			}); err != nil {
				// The response is still returned to the client; only persistence
				// failed. L1 already holds it for identical repeats.
				e.logger.Warn("l2 store failed", "err", err)
			}
		}
	}

	return &Result{
		Body:       resp.Body,
		Status:     resp.Status,
		Header:     resp.Header,
		Tier:       "MISS",
		PromptLang: q.Lang,
		EntryLang:  q.Lang,
	}, nil
}

func (e *Engine) expired(createdAt time.Time) bool {
	return e.ttl > 0 && e.now().Sub(createdAt) > e.ttl
}
