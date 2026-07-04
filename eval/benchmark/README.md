# Benchmark

Multilingual hit-rate and cost-savings benchmark vs an English-only baseline
cache. The baseline gets ~0 cross-lingual hits by construction.

Planned contents:

- `queryset/` — ~200 support-bot-style prompts, each in 2–3 of TR/ES/EN/ZH,
  including ~20% near-miss pairs (same topic, different numbers/IDs) to prove
  the guard prevents false savings.
- Metric A — hit rate (Polyglot vs baseline).
- Metric B — net tokens/$ saved (after embedding + translation overhead).
- Metric C — false-positive rate with vs without the structural guard.
- Headline: net cost saved vs cache size / workload multilingualism.

**Honesty rule:** sell the win as **cost**, not latency — translated paths add
embedding + translation overhead.
