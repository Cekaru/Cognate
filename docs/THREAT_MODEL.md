# Threat Model

Polyglot Cache is a shared, cross-lingual semantic cache in front of an LLM. Its
value and its risk come from the *same* property: a prompt in one language can be
served an answer created for a prompt in another language, across tenants. This
document states the threats that property creates and what the code does about
each — including where a mitigation is shipped and where it is still planned.

Status legend: **[shipped]** implemented and tested · **[partial]** partly
implemented · **[planned]** designed, not yet built.

## Trust boundaries

- **Client → proxy.** Callers present an API credential. The proxy never logs it
  and never forwards it upstream; it derives a hashed tenant identity from it.
- **Proxy → embedding sidecar.** Local, trusted; prompts are sent in plaintext to
  be embedded. The sidecar holds no persistent state.
- **Proxy → L2 store (pgvector).** The persistence boundary — the "at rest" line.
- **Proxy → upstream LLM.** The proxy authenticates with its own provider key.

## 1. Cross-lingual cache poisoning (headline)

An attacker seeds an entry in language X so that a victim querying in language Y
is served attacker-influenced content — and the victim, reading a second
language, is less able to notice tampering than a monolingual user would be.

- **[shipped] Structural token guard.** On every semantic hit, locale-normalized
  numbers, IDs, currencies, dates, and code identifiers are compared between the
  two prompts; a mismatch vetoes the hit (`internal/guard`). This is what stops a
  `$100` prompt being answered from a `$1000` entry sitting at 0.98 cosine.
- **[shipped] Per-language-pair thresholds.** Cutoffs are calibrated to a
  precision target per pair (`eval/calibration`), not one global number, so the
  looser cross-lingual pairs cannot silently admit near-misses.
- **[shipped] TTL rotation.** Entries expire (`CACHE_TTL`), bounding how long a
  poisoned entry can live.
- **[partial] Tenant scoping.** Isolation is available (below); the *default* is
  shared, which is the intended trade-off, stated in §2.

## 2. Tenant isolation and the hit-rate / leak-surface tension

Cross-tenant sharing is where the cross-lingual hit rate lives — and it is also
the leak surface. This is the central honest tension of the design, not a bug.

- **[shipped] Documented shared-by-default.** The shared scope is deliberate and
  labeled as such in code and README.
- **[shipped] Isolation opt-out.** `TENANT_ISOLATION=isolated` gives each tenant a
  private namespace; a single request can opt out with `X-Polyglot-Isolation:
  tenant`. An isolated tenant neither reads nor seeds the shared pool.
- **Residual risk.** A tenant that stays in the shared pool accepts that its
  prompts can seed, and be served from, other tenants' equivalent prompts.

## 3. Timing side-channel

A hit (cache read) and a miss (full LLM call) differ in latency, so response time
leaks whether an equivalent prompt already exists in the shared cache.

- **[planned] Constant-time floor scoped only to shared semantic hits** — not a
  blanket delay, so isolated and L1 paths keep their latency. Random jitter is
  explicitly *not* treated as a fix (it is averaged out by repetition).

## 4. Data at rest

The L2 store persists responses; a database compromise must not hand over
plaintext answers.

- **[shipped] AES-256-GCM at rest.** When `POLYGLOT_ENCRYPTION_KEY` is set,
  response bodies are sealed with AES-256-GCM before persistence and opened on a
  hit (`internal/crypto`, wired in `internal/cache/engine`). A fresh random nonce
  per value means identical responses do not correlate at rest. Encryption fails
  closed: a bad key refuses startup, and a seal/open failure skips the cache
  rather than exposing plaintext.
- **[shipped] Pluggable KeyProvider.** The env-var provider is the single-node
  path; the interface is the seam for a KMS/Vault envelope provider.
- **Honest boundary.** The **lookup key stays a plaintext hash** so the index
  works — an attacker with the database learns *which* hashed prompts exist and
  their structural tokens (numbers/IDs are stored for the guard), just not the
  response text. Env-var mode protects against a stolen database, not against a
  host with the running process's environment.
- **[planned] KMS/Vault envelope provider** (KEK wraps a DEK) for production.

## 5. Cache flooding / noisy neighbour (DoS)

One tenant tries to evict others' entries or exhaust storage.

- **[shipped] Per-tenant request byte-rate limit** at the front door (429 on
  breach) and a **per-tenant L2 store-byte quota** that caps how much one tenant
  can write (`internal/tenant`). A denied write is skipped while the response is
  still served.

## 6. Horizontal-scaling caveat

- **[partial / stated] In-memory limiter state.** The quota and rate limiter hold
  state in process, so across multiple proxy instances the limits are per-instance,
  not global. A multi-node deployment needs shared state (e.g. Redis). Stated, not
  hidden.

## 7. Prompt / response confidentiality in logs

- **[shipped] Zero plaintext in the audit log.** The per-request audit event
  (`internal/telemetry`) carries only a hashed tenant id, model, cache tier,
  status, similarity, a guard-fired flag, languages, token counts, and latency —
  built with an explicit attribute list so no prompt or response body can reach
  the log. A test asserts the field set is closed.
- **[planned] Opt-in payload logging** would be encrypted and on a separate
  stream, never the default audit path.

## What this model deliberately does not claim

- It does not claim env-var encryption protects against a compromised host.
- It does not claim the shared cache is private; it claims the trade-off is
  explicit and opt-out-able.
- It does not claim timing is currently uniform; §3 is planned.
