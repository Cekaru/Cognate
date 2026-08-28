// Package engine wires the L1 (exact) and L2 (semantic) tiers into a single
// cache lookup path and owns the store-on-miss logic. It is the piece the proxy
// calls; keeping it HTTP-free makes the whole cross-lingual pipeline testable
// with a fake embedder and a fake upstream.
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
	"github.com/kaanrumin/polyglot-cache/internal/guard"
)

// Embedder returns BGE-M3 embeddings for texts, index-aligned. *embed.Client
// satisfies this; tests supply a fake.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// StoreQuota gates cache writes per tenant — the anti-flooding control.
// *tenant.Quotas satisfies this.
type StoreQuota interface {
	AllowStore(tenant string, n int) bool
}

// Cipher encrypts cache values at rest. It wraps the response body before it is
// persisted to L2 and unwraps it on a hit; the lookup key stays a plaintext hash
// so the index is unaffected. *crypto.Encryptor satisfies this. A nil Cipher
// disables encryption (responses are stored as-is).
type Cipher interface {
	Seal(plaintext []byte) ([]byte, error)
	Open(ciphertext []byte) ([]byte, error)
}

// Query is a normalized cache lookup request built from an OpenAI chat request.
type Query struct {
	// TenantScope is the namespace searched and stored: the shared scope by
	// default (cross-tenant sharing is where cross-lingual hits live), or the
	// tenant's own hash when the tenant opted out of sharing.
	TenantScope string
	// Tenant is the hashed tenant identity, always set; it keys the store
	// quota and appears in audit logs. Never a raw credential.
	Tenant    string
	Model     string
	ExactText string // canonical string hashed for the L1 key (roles + content)
	EmbedText string // semantic content embedded for the L2 lookup
	Lang      string // detected language of this prompt
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
	GuardFired bool        // a structural-token mismatch vetoed a candidate
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
	l1         *exact.LRU[*record]
	l2         semantic.Index
	embedder   Embedder
	thresholds *Thresholds
	ttl        time.Duration // 0 = no expiry
	quota      StoreQuota    // nil = unlimited
	cipher     Cipher        // nil = responses stored in plaintext
	logger     *slog.Logger
	now        func() time.Time
}

// Config configures an Engine.
type Config struct {
	L1Capacity int
	// Threshold is the global cosine cutoff, used directly when Thresholds is
	// nil and as the fallback default otherwise.
	Threshold float64
	// Thresholds is the calibrated per-language-pair cutoff table. Optional.
	Thresholds *Thresholds
	TTL        time.Duration
	// L2 is the semantic backend. When nil an in-memory index is used, so tests
	// and single-process runs need no database; main.go supplies the pgvector
	// backend when DATABASE_URL is set.
	L2 semantic.Index
	// Quota, when non-nil, is consulted before every L2 store; a denied write
	// is skipped (and logged) while the response is still served.
	Quota StoreQuota
	// Cipher, when non-nil, encrypts response bodies at rest in L2. On a store
	// the body is sealed before persistence; on a hit it is opened before use.
	// A seal or open failure fails closed (the entry is not stored / not served
	// from cache) rather than exposing plaintext.
	Cipher Cipher
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
	ths := cfg.Thresholds
	if ths == nil {
		ths = &Thresholds{Default: cfg.Threshold}
	}
	return &Engine{
		l1:         exact.New[*record](cfg.L1Capacity),
		l2:         l2,
		embedder:   embedder,
		thresholds: ths,
		ttl:        cfg.TTL,
		quota:      cfg.Quota,
		cipher:     cfg.Cipher,
		logger:     logger,
		now:        time.Now,
	}
}

// Serve runs the full lookup: L1 exact, then L2 semantic, then (on a miss) the
// upstream call, storing a successful response in both tiers. The embedding
// computed for the L2 lookup is reused when storing, so a prompt is embedded at
// most once per request.
func (e *Engine) Serve(ctx context.Context, q Query, up UpstreamFunc) (*Result, error) {
	key := q.key()

	// Tracks whether the structural guard vetoed an over-threshold candidate, so
	// the fall-through MISS can report it in the audit event.
	var guardFired bool

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

	// Structural tokens of the incoming prompt: compared by the guard on an
	// L2 hit and persisted alongside the embedding on a store.
	var qTokens cache.StructuralTokens
	if vec != nil {
		qTokens = guard.Extract(q.EmbedText, q.Lang)
	}

	// L2 — semantic, searched within the query's single scope: the shared
	// namespace by default (cross-tenant sharing is intentional; it is where
	// cross-lingual hits live), or the tenant's own namespace when isolated —
	// an isolated tenant neither reads nor seeds the shared pool.
	if vec != nil {
		ent, sim, err := e.l2.Search(ctx, vec, q.Model, q.TenantScope)
		switch {
		case err != nil:
			// A store-level failure degrades to passthrough, like an embed error;
			// never fail the request on a cache-path error.
			e.logger.Warn("l2 search failed; bypassing L2", "err", err)
		case ent != nil && !e.expired(ent.CreatedAt):
			threshold := e.thresholds.For(q.Lang, ent.Lang)
			hit := sim >= threshold
			// The guard is the safety net for near-misses the embedding cannot
			// separate ($100 vs $1000 at 0.98 cosine): a structural mismatch
			// vetoes the hit and the request falls through to the real LLM.
			guardOK, guardCat := true, ""
			if hit {
				guardOK, guardCat = guard.Compare(qTokens, ent.Tokens)
			}
			if hit && !guardOK {
				guardFired = true
			}
			// Log every semantic lookup's score.
			e.logger.Info("semantic lookup",
				"similarity", sim,
				"threshold", threshold,
				"hit", hit && guardOK,
				"guard_fired", hit && !guardOK,
				"guard_category", guardCat,
				"prompt_lang", q.Lang,
				"entry_lang", ent.Lang,
				"model", q.Model,
				"tenant", q.Tenant,
			)
			if hit && guardOK {
				// Decrypt at rest → plaintext for serving. A decrypt failure
				// (wrong key, tampered row) fails closed: skip the cache and
				// fall through to the real LLM rather than serve garbage.
				body, err := e.plaintext(ent.Response)
				if err != nil {
					e.logger.Warn("l2 decrypt failed; bypassing entry", "err", err, "model", q.Model)
				} else {
					// Promote to L1 (in-memory, plaintext) so an identical repeat
					// skips embedding.
					e.l1.Put(key, &record{response: body, lang: ent.Lang, createdAt: e.now()})
					return &Result{
						Body:       body,
						Status:     http.StatusOK,
						Tier:       "L2",
						Similarity: sim,
						PromptLang: q.Lang,
						EntryLang:  ent.Lang,
					}, nil
				}
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

	// Only cache a clean success, and only within the tenant's store quota —
	// the quota is what stops one tenant flooding the shared cache. The L1
	// LRU is bounded by capacity, so only the L2 store is gated.
	if resp.Status == http.StatusOK {
		now := e.now()
		e.l1.Put(key, &record{response: resp.Body, lang: q.Lang, createdAt: now})
		if vec != nil && e.quota != nil && !e.quota.AllowStore(q.Tenant, len(resp.Body)) {
			e.logger.Warn("l2 store skipped: tenant over store quota", "tenant", q.Tenant, "bytes", len(resp.Body))
			vec = nil
		}
		if vec != nil {
			// Encrypt the response at rest. A seal failure fails closed: skip the
			// L2 write (L1 still holds the plaintext for identical repeats) rather
			// than persist an unencrypted body when encryption was requested.
			body, err := e.ciphertext(resp.Body)
			if err != nil {
				e.logger.Warn("l2 store skipped: encrypt failed", "err", err)
			} else if err := e.l2.Add(ctx, &semantic.Entry{
				Key:         key,
				TenantScope: q.TenantScope,
				Model:       q.Model,
				Embedding:   vec,
				Response:    body,
				Lang:        q.Lang,
				Tokens:      qTokens,
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
		GuardFired: guardFired,
		PromptLang: q.Lang,
		EntryLang:  q.Lang,
	}, nil
}

// ciphertext seals a response body for storage when a cipher is configured, or
// returns it unchanged otherwise.
func (e *Engine) ciphertext(body []byte) ([]byte, error) {
	if e.cipher == nil {
		return body, nil
	}
	return e.cipher.Seal(body)
}

// plaintext opens a stored response body when a cipher is configured, or returns
// it unchanged otherwise.
func (e *Engine) plaintext(stored []byte) ([]byte, error) {
	if e.cipher == nil {
		return stored, nil
	}
	return e.cipher.Open(stored)
}

func (e *Engine) expired(createdAt time.Time) bool {
	return e.ttl > 0 && e.now().Sub(createdAt) > e.ttl
}
