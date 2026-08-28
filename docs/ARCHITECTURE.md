# Architecture

> The as-built architecture of the proxy and its cache tiers.

## Request path

```
client / gateway ──► Polyglot Cache (Go proxy)
    1. L1 exact-hash lookup (SHA-256 + LRU)
    2. embed raw prompt via BGE-M3 sidecar
    3. L2 vector search (pgvector)
    4. per-language-pair threshold check
    5. structural token guard (locale-aware)
    6. on miss: call upstream LLM (passthrough)
    7. store {query embedding, response}
```

**The hot path never translates the query.** Translation only touches the
*response*, lazily, the first time an entry is served in a new language.

## Current status

- Go `net/http` proxy with an OpenAI-compatible `/v1/*` surface.
- L1 exact-hash cache and L2 cross-lingual semantic cache over BGE-M3 embeddings;
  requests that can't be cached (streaming, non-chat) pass straight through to the
  configured upstream provider (`internal/provider`).
- L2 is backed by pgvector when `DATABASE_URL` is set, so the cache survives a
  restart; with no database it falls back to a process-local in-memory index.
- `docker compose up` brings up **proxy + BGE-M3 sidecar + Postgres/pgvector**.
- Optional AES-256-GCM encryption at rest for L2 response bodies behind a
  pluggable `KeyProvider` (`internal/crypto`), enabled by `POLYGLOT_ENCRYPTION_KEY`;
  the lookup key stays a plaintext hash so the index is unaffected.
- Per-request structured JSON audit event (`internal/telemetry`): hashed tenant,
  tier, similarity, guard-fired flag, token counts, latency — no plaintext payloads.

## Package map

| Package                    | Responsibility                                   |
|----------------------------|--------------------------------------------------|
| `cmd/proxy`                | entrypoint, lifecycle                            |
| `internal/proxy`           | HTTP surface, passthrough reverse proxy          |
| `internal/provider`        | upstream LLM abstraction (OpenAI-compatible)     |
| `internal/config`          | 12-factor env config                             |
| `internal/telemetry`       | structured JSON logging + per-request audit event |
| `internal/cache` (+ `exact`, `semantic`) | L1/L2 tiers, `CacheEntry`          |
| `internal/embed`           | Go client to the BGE-M3 sidecar                  |
| `internal/guard`           | locale-aware structural token guard              |
| `internal/tenant`          | isolation + byte quota                           |
| `internal/crypto`          | AES-256-GCM, pluggable KeyProvider               |
