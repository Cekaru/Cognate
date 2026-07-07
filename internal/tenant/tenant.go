package tenant

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"strings"
	"sync"
	"time"

	"golang.org/x/time/rate"
)

// Anonymous is the tenant ID assigned to requests carrying no credential.
// Anonymous callers share one bucket, so an unauthenticated flood throttles
// itself, not the identified tenants.
const Anonymous = "anon"

// IsolationHeader lets a single request opt out of the shared cache
// ("X-Polyglot-Isolation: tenant"). The shared default is a deliberate
// stance — cross-tenant sharing is where the cross-lingual hit rate lives —
// and the opt-out is the documented escape hatch (see THREAT_MODEL.md).
const IsolationHeader = "X-Polyglot-Isolation"

// FromRequest derives the tenant identity from the inbound credential:
// hex(SHA-256(key)) truncated to 16 bytes. The raw key is never stored or
// logged; the hash is the namespace, the audit identity, and the quota key.
func FromRequest(r *http.Request) string {
	key := strings.TrimSpace(strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer"))
	if key == "" {
		key = strings.TrimSpace(r.Header.Get("X-Api-Key"))
	}
	if key == "" {
		return Anonymous
	}
	sum := sha256.Sum256([]byte(key))
	return hex.EncodeToString(sum[:16])
}

// Isolated reports whether this request must be served from (and stored into)
// its own tenant namespace instead of the shared cache. defaultIsolated is the
// deployment-wide stance (TENANT_ISOLATION=isolated); the header opts a single
// request out of sharing, never into it — an isolated deployment stays
// isolated.
func Isolated(r *http.Request, defaultIsolated bool) bool {
	if defaultIsolated {
		return true
	}
	return strings.EqualFold(r.Header.Get(IsolationHeader), "tenant")
}

// Config sizes the per-tenant limiters. Rates are bytes/second with a burst
// ceiling; a byte is the honest unit here because it is what the attacker
// controls (token counts are only known after upstream tokenization).
type Config struct {
	// RequestBytesPerSec / RequestBurst throttle inbound prompt traffic.
	RequestBytesPerSec float64
	RequestBurst       int
	// StoreBytesPerSec / StoreBurst cap how fast one tenant can grow the
	// cache — the anti-flooding quota. The burst is the effective budget; the
	// refill lets a legitimate tenant recover over time.
	StoreBytesPerSec float64
	StoreBurst       int
}

// Quotas tracks per-tenant limiters. Limiters are created lazily and kept for
// the process lifetime; state is in-memory only, so multi-node deployments
// need a shared store (documented caveat, see THREAT_MODEL.md §6).
type Quotas struct {
	cfg Config

	mu    sync.Mutex
	reqs  map[string]*rate.Limiter
	store map[string]*rate.Limiter
}

// NewQuotas builds the limiter registry. Zero or negative rates disable the
// corresponding limit.
func NewQuotas(cfg Config) *Quotas {
	return &Quotas{
		cfg:   cfg,
		reqs:  map[string]*rate.Limiter{},
		store: map[string]*rate.Limiter{},
	}
}

// AllowRequest reports whether tenant may spend n more request bytes now.
func (q *Quotas) AllowRequest(tenant string, n int) bool {
	if q == nil || q.cfg.RequestBytesPerSec <= 0 {
		return true
	}
	return q.limiter(q.reqs, tenant, q.cfg.RequestBytesPerSec, q.cfg.RequestBurst).AllowN(time.Now(), n)
}

// AllowStore reports whether tenant may write n more bytes into the cache.
// A denial only skips the store; the request itself is still served.
func (q *Quotas) AllowStore(tenant string, n int) bool {
	if q == nil || q.cfg.StoreBytesPerSec <= 0 {
		return true
	}
	return q.limiter(q.store, tenant, q.cfg.StoreBytesPerSec, q.cfg.StoreBurst).AllowN(time.Now(), n)
}

func (q *Quotas) limiter(m map[string]*rate.Limiter, tenant string, perSec float64, burst int) *rate.Limiter {
	q.mu.Lock()
	defer q.mu.Unlock()
	l, ok := m[tenant]
	if !ok {
		l = rate.NewLimiter(rate.Limit(perSec), burst)
		m[tenant] = l
	}
	return l
}
