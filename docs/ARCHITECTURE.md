# Architecture

> The as-built architecture of the proxy and its cache tiers.

## Request path (target)

```
client / gateway ──► Polyglot Cache (Go proxy)
    1. L1 exact-hash lookup (SHA-256 + LRU)         ← Phase 1
    2. embed raw prompt via BGE-M3 sidecar          ← Phase 1
    3. L2 vector search (pgvector)                   ← Phase 1
    4. per-language-pair threshold check             ← Phase 2a
    5. structural token guard (locale-aware)         ← Phase 2b
    6. on miss: call upstream LLM (passthrough)      ← Phase 0 ✅
    7. store {query embedding, response}             ← Phase 1
```

**The hot path never translates the query.** Translation only touches the
*response*, lazily, the first time an entry is served in a new language.

## Current status — Phase 0

- Go `net/http` proxy with an OpenAI-compatible `/v1/*` surface.
- Pure passthrough to the configured upstream provider (`internal/provider`).
- `docker compose up` brings up **proxy + BGE-M3 sidecar + Postgres/pgvector**.
- Structured JSON audit logging (`internal/telemetry`); no plaintext payloads.

## Package map

| Package                    | Responsibility                                   | Phase |
|----------------------------|--------------------------------------------------|-------|
| `cmd/proxy`                | entrypoint, lifecycle                            | 0     |
| `internal/proxy`           | HTTP surface, passthrough reverse proxy          | 0     |
| `internal/provider`        | upstream LLM abstraction (OpenAI-compatible)     | 0     |
| `internal/config`          | 12-factor env config                             | 0     |
| `internal/telemetry`       | structured JSON logging                          | 0     |
| `internal/cache` (+ `exact`, `semantic`) | L1/L2 tiers, `CacheEntry`          | 1     |
| `internal/embed`           | Go client to the BGE-M3 sidecar                  | 1     |
| `internal/guard`           | locale-aware structural token guard              | 2b    |
| `internal/tenant`          | isolation + byte quota                           | 2c    |
| `internal/crypto`          | AES-256-GCM, pluggable KeyProvider               | 4b    |
