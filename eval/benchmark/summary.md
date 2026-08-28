# Cross-lingual benchmark

Reproducible offline from committed BGE-M3 scores (`eval/calibration/scores/`)
and the shipped thresholds (`BAAI/bge-m3`). Regenerate with `go run ./eval/benchmark`.

Workload: **1200** labeled cross-lingual queries across 6 language pairs, of which
**240** are true hit opportunities (semantically equivalent prompts) and the rest
are near-miss or unrelated candidates that must **not** be served.

## Headline

| Metric | Polyglot (shipped) | English-only baseline |
|--------|:------------------:|:---------------------:|
| Cross-lingual hit rate | **70.8%** | 0.0% (by construction) |
| Unsafe serves (leaks) | **0.0%** | 0.0% |
| Net cost saved / 1M queries | **$81.67** | $0 |

The baseline captures **zero** cross-lingual hits: an English-only cache cannot
match a Turkish prompt to a Spanish entry. Every cross-lingual hit Polyglot lands
is spend the baseline pays in full.

## What the guard is worth

Without the structural guard, the threshold alone serves near-miss candidates —
same topic, wrong number or ID — as if they were hits. Those are wrong answers.

| | Threshold only | Threshold + guard (shipped) |
|--|:--:|:--:|
| Unsafe serve rate | 8.8% | **0.0%** |
| Unsafe serves (count) | 84 | **0** |
| True hits kept | 170 | 170 |

The guard removes the leak while keeping the hits: the threshold can sit lower
*because* the guard is the backstop, which is exactly why the calibrated
with-guard cutoffs recover more recall at the precision target.

## Per-pair hit rate

| pair | hit opportunities | hits | hit rate | leaks (no guard → guard) |
|------|:--:|:--:|:--:|:--:|
| en-es | 40 | 37 | 92% | 16 → 0 |
| en-tr | 40 | 30 | 75% | 13 → 0 |
| en-zh | 40 | 28 | 70% | 23 → 0 |
| es-tr | 40 | 27 | 68% | 8 → 0 |
| es-zh | 40 | 26 | 65% | 10 → 0 |
| tr-zh | 40 | 22 | 55% | 14 → 0 |

## Cost model (illustrative, overridable)

Sold as **cost**, not latency: a translated path adds embedding overhead, so
total latency is higher — the win is dollars, not milliseconds.

| Assumption | Value |
|--|--|
| Prompt / completion tokens | 60 / 180 |
| Provider price (in / out, USD per 1M) | $0.15 / $0.60 |
| Embedding price (USD per 1M) | $0.02 |
| Avoided LLM call | $0.000117 each |
| Embedding overhead | $0.000001 per query |

At the measured **70.8%** hit rate over 1M cross-lingual queries:

- gross saved (avoided LLM calls): **$82.88**
- embedding overhead (paid on every query): **$1.20**
- **net saved: $81.67**

Embedding is self-hosted (BGE-M3 sidecar), so the real overhead trends below the
hosted rate used here; the net figure is deliberately conservative.
