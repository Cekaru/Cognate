# Polyglot Cache — Cross-Lingual Semantic Cache for LLMs

> The first open-source LLM-gateway component where a prompt in **any** language
> hits a cache entry created by a prompt in **another** language — one cached
> answer serves your whole multilingual userbase.

<p>
  <img alt="Go" src="https://img.shields.io/badge/Go-1.25-00ADD8?logo=go&logoColor=white">
  <img alt="Embeddings" src="https://img.shields.io/badge/embeddings-BGE--M3-6E56CF">
  <img alt="Vector store" src="https://img.shields.io/badge/vector%20store-pgvector-4169E1?logo=postgresql&logoColor=white">
  <img alt="Languages" src="https://img.shields.io/badge/languages-TR%20·%20ES%20·%20EN%20·%20ZH-2DA44E">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

A self-hostable, OpenAI-compatible sidecar in Go. Point an existing gateway
(LiteLLM, Bifrost) or your app at it, and semantically-equivalent prompts across
languages share a single cache entry — cutting token spend on multilingual
workloads that every English-only cache misses entirely.

**Core insight:** don't translate-then-embed. BGE-M3 places cross-lingual
equivalents near each other in the *same* vector space, so the raw prompt is
embedded in its original language; translation only ever touches the *response*,
lazily, on a hit.

---

## The one behavior everything rests on

A **Spanish** prompt seeds the cache from the real LLM. An equivalent **Turkish**
prompt is then answered from that Spanish entry — no second LLM call.

```bash
# 1) seeds the cache (real LLM call)
curl -sS -D - http://localhost:8080/v1/chat/completions -o /dev/null \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"¿Cuál es la capital de Francia?"}]}'
#   X-Polyglot-Cache: MISS

# 2) different language, same meaning → served from the Spanish entry
curl -sS -D - http://localhost:8080/v1/chat/completions -o /dev/null \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Fransa'\''nın başkenti neresidir?"}]}'
#   X-Polyglot-Cache:       L2
#   X-Polyglot-Prompt-Lang: tr
#   X-Polyglot-Entry-Lang:  es
#   X-Polyglot-Similarity:  0.9x
```

An English-only baseline cache scores **zero** cross-lingual hits here by
construction. Run it yourself: [`eval/demo/`](eval/demo/).

---

## Quick start

```bash
cp .env.example .env          # then set PROVIDER_API_KEY
docker compose up --build     # proxy + BGE-M3 sidecar + Postgres/pgvector
```

| Service | URL | Endpoints |
|---------|-----|-----------|
| Proxy | http://localhost:8080 | `/healthz`, OpenAI-compatible `/v1/*` |
| Sidecar (BGE-M3) | http://localhost:8000 | `/health`, `/embed` |
| Postgres / pgvector | localhost:5432 | — |

Smoke-test the passthrough (needs a real `PROVIDER_API_KEY`):

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"ping"}]}'
```

Local Go build:

```bash
go build ./... && go vet ./... && go test ./...
```

---

## How a request flows

```
client / gateway ──► Polyglot Cache (Go proxy)
   1. L1 exact-hash lookup ............. SHA-256 + LRU
   2. embed raw prompt ................. BGE-M3 sidecar, original language
   3. L2 vector search ................ pgvector (persistent) / in-memory
   4. per-language-pair threshold ..... calibrated cutoffs, not one global number
   5. structural token guard .......... locale-aware; a mismatch vetoes the hit
   6. miss → call upstream LLM ........ passthrough reverse proxy
   7. store {query embedding, response}
```

**The hot path never translates the query.** Streaming and non-chat requests it
can't cache pass straight through. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
for the package map.

---

## The safety layer (Phase 2 — the core)

Cross-lingual matching is riskier than monolingual matching: equivalents score
lower, and a naive guard silently fails on exactly the non-English inputs this
project exists to serve. Three defenses ship:

- **Structural token guard** — on every semantic hit, locale-normalized numbers,
  dates, currencies, IDs, and code identifiers are compared between the two
  prompts (`1.000,50` ≡ `1,000.50`, `一百` ≡ `100`, TR/ES day-first vs EN
  month-first dates). `transfer $100` never serves `transfer $1000`, no matter
  how high the cosine score.
- **Per-language-pair thresholds** — cutoffs are calibrated empirically per pair
  ([`eval/calibration/`](eval/calibration/)) and loaded via `THRESHOLDS_FILE`,
  because one global threshold can't be both safe and useful across pairs.
- **Tenant isolation & quotas** — tenants are hashed API-key namespaces. The
  cache is **cross-tenant shared by default** (that is where cross-lingual hits
  live — a deliberate, documented trade-off); set `TENANT_ISOLATION=isolated`,
  or opt a single request out with `X-Polyglot-Isolation: tenant`. Per-tenant
  byte-rate limits and a cache-write quota defend against flooding.

### Calibration results

40 intents × 4 languages, precision target 0.99. The engine always runs the
guard *after* the threshold, so cutoffs only have to separate what the guard
cannot catch — which is why the shipped "with guard" thresholds are lower and
recall is higher, at 100% measured precision.

| pair | AUC | recall (embed only) | recall (with guard) | precision | guard catch | false fires |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| en-es | 0.963 | 50% | **92%** | 100% | 68/68 | 0/40 |
| en-tr | 0.925 | 40% | **75%** | 100% | 68/68 | 0/40 |
| en-zh | 0.907 | 52% | **70%** | 100% | 68/68 | 0/40 |
| es-tr | 0.913 | 35% | **68%** | 100% | 68/68 | 0/40 |
| es-zh | 0.907 | 28% | **65%** | 100% | 68/68 | 0/40 |
| tr-zh | 0.886 | 20% | **55%** | 100% | 68/68 | 0/40 |

Regenerate with `go run ./eval/calibration` — raw scores in `scores/`, ROC
curves in `roc/`, full write-up in [`eval/calibration/WRITEUP.md`](eval/calibration/WRITEUP.md).

---

## Configuration

All config is 12-factor env vars (no config file). Common knobs:

| Variable | Default | Purpose |
|----------|---------|---------|
| `PROVIDER_BASE_URL` | `https://api.openai.com` | Upstream OpenAI-compatible endpoint |
| `PROVIDER_API_KEY` | — | Upstream credential (never logged) |
| `EMBED_SIDECAR_URL` | `http://sidecar:8000` | BGE-M3 embedding sidecar |
| `DATABASE_URL` | — | pgvector DSN; empty = in-memory L2 |
| `CACHE_ENABLED` | `true` | Master switch; `false` = pure passthrough |
| `SEMANTIC_THRESHOLD` | `0.85` | Global fallback cutoff when no pair table applies |
| `THRESHOLDS_FILE` | — | Calibrated per-pair threshold table (JSON) |
| `CACHE_TTL` | `24h` | Entry lifetime; `0` = no expiry |
| `TENANT_ISOLATION` | `shared` | `isolated` = private cache per tenant |
| `LOG_LEVEL` | `info` | `debug` \| `info` \| `warn` \| `error` |

See [`internal/config/config.go`](internal/config/config.go) for the full list
(byte-rate quotas, burst ceilings, L1 capacity).

---

## Project status

| Phase | Scope | State |
|-------|-------|:-----:|
| **1** | OpenAI-compatible proxy, L1 exact + L2 semantic cache, BGE-M3 sidecar, pgvector | ✅ Done |
| **2** | Structural guard, per-pair calibration, tenant isolation & quotas | ✅ Done |
| **3** | Multilingual benchmark: cross-lingual hit rate + dollar-savings report | 🚧 In progress |
| **4** | Threat model, AES-256-GCM at rest, observability, packaging | 🟡 Partial |

---

## Security

`.env` is gitignored — commit only `.env.example`. Audit logs are structured JSON
with **no plaintext prompts or responses**. The threat model (centered on
cross-lingual cache poisoning and the honest hit-rate-vs-leak-surface tension) is
drafted in [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

## License

[MIT](LICENSE).
