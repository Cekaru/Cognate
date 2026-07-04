# Demo — the cross-lingual hit

The whole project rests on one observable behavior:

> A **Turkish** prompt returns the answer cached from an equivalent **Spanish**
> prompt, end-to-end, with the similarity score logged.

## Run it

Bring the stack up with a real upstream key, then run the demo:

```bash
make up            # proxy + sidecar (BGE-M3) + pgvector
# in another shell:
PROXY=http://localhost:8080 MODEL=gpt-4o-mini make demo
```

`crosslingual_hit.sh` sends two requests and prints the response headers:

1. `¿Cuál es la capital de Francia?` → `X-Polyglot-Cache: MISS` (seeds the cache
   from the real LLM).
2. `Fransa'nın başkenti neresidir?` → `X-Polyglot-Cache: L2`,
   `X-Polyglot-Prompt-Lang: tr`, `X-Polyglot-Entry-Lang: es`, plus
   `X-Polyglot-Similarity`.

The second response is served from the Spanish entry without calling the LLM.
The proxy also logs the score on every semantic lookup:

```
docker compose logs proxy | grep 'semantic lookup'
# ... "similarity":0.9x "hit":true "prompt_lang":"tr" "entry_lang":"es" ...
```

> Note: the served answer is currently returned in the seeding language
> (Spanish). Lazy per-language translation of the response is a later step.

## 12-pair quality gut-check

`pairs.jsonl` holds 12 intents, each expressed in TR/ES/EN/ZH. Use it as a
sanity check: seed each intent in one language, query it in the other three, and
eyeball whether the served answer is actually right. This set also seeds the
threshold-calibration positives later on.

## What proves the mechanism without the stack

The deterministic tests already exercise the full pipeline with a fake embedder
and a fake upstream — no model download, no API key:

```bash
go test ./internal/cache/... ./internal/proxy/...
# TestCrossLingualHit, TestChatCrossLingualHitOverHTTP, ...
```
