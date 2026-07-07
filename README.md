# Polyglot Cache — Cross-Lingual Semantic Cache for LLMs

> The first open-source LLM-gateway component where a prompt in **any** language
> hits a cache entry created by a prompt in **another** language — one cached
> answer serves your whole multilingual userbase.

A self-hostable, OpenAI-compatible sidecar in Go. Point an existing gateway
(LiteLLM, Bifrost) or your app at it, and semantically-equivalent prompts across
languages share a single cache entry — cutting token spend on multilingual
workloads that every English-only cache misses entirely.

**Core insight:** don't translate-then-embed. BGE-M3 places cross-lingual
equivalents near each other in the same vector space, so the raw prompt is
embedded in its original language; translation only ever touches the *response*,
lazily, on a hit.

Languages: **Turkish, Spanish, English, Chinese**.

---

## Status

`docker compose up` brings up **proxy + BGE-M3 sidecar + Postgres/pgvector**. The
proxy exposes an OpenAI-compatible surface and caches semantically-equivalent
prompts across languages; requests it can't cache (streaming, non-chat) pass
straight through to the upstream provider.

The matching layer is safety-gated (Phase 2):

- **Structural token guard** — on every semantic hit, locale-normalized
  numbers, dates, currencies, IDs, and code identifiers are compared between
  the two prompts (`1.000,50` ≡ `1,000.50`, `一百` ≡ `100`, TR/ES day-first vs
  EN month-first dates). A mismatch vetoes the hit: `transfer $100` never
  serves `transfer $1000`, no matter how high the cosine score.
- **Per-language-pair thresholds** — cross-lingual positives score lower than
  monolingual ones, by different margins per pair, so cutoffs are calibrated
  empirically per pair (`eval/calibration/`) and loaded via `THRESHOLDS_FILE`.
- **Tenant isolation & quotas** — tenants are hashed API-key namespaces.
  The cache is **cross-tenant shared by default** (that is where cross-lingual
  hits live — a deliberate, documented trade-off); a deployment can set
  `TENANT_ISOLATION=isolated`, or a single request can opt out with
  `X-Polyglot-Isolation: tenant`. Per-tenant byte-rate limits and a cache-write
  quota defend against flooding.

## Quick start

```bash
cp .env.example .env          # then set PROVIDER_API_KEY
docker compose up --build
```

- Proxy:   http://localhost:8080  (`/healthz`, OpenAI-compatible `/v1/*`)
- Sidecar: http://localhost:8000  (`/health`, `/embed`)
- Postgres/pgvector: localhost:5432

Smoke test the passthrough (needs a real `PROVIDER_API_KEY`):

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"ping"}]}'
```

## Local Go build

```bash
go build ./...
go vet ./...
```

## Layout

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the package map and the
target request path.

## Security

`.env` is gitignored; commit only `.env.example`. The threat model is drafted in
[`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).
