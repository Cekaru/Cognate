# Cross-lingual embedding gut-check

Measures whether BGE-M3 actually places cross-lingual **equivalents** (the same
intent expressed in EN/ES/TR/ZH) closer together than **non-equivalents**
(different intents). This is the reality check the fake-embedder unit tests
can't give — it exercises the real sidecar on real prompts and validates the
project's central thesis directly.

## Run

With the embedding sidecar up (`docker compose up -d sidecar`):

```bash
go run ./eval/gutcheck
# options: -sidecar http://localhost:8000  -pairs eval/demo/pairs.jsonl  -threshold 0.85
```

The intents come from [`../demo/pairs.jsonl`](../demo/pairs.jsonl) (12 intents ×
4 languages).

## What it reports

- **Per-intent cross-lingual cosine** for all six language pairs, plus the weakest
  pair per intent.
- **Distributions** of equivalents vs. non-equivalents (min / p50 / mean / max).
- **Threshold gate**: at a given cutoff, how many true equivalents are admitted
  and how many non-equivalents leak.
- **Separability verdict**: whether the worst equivalent still beats the best
  non-equivalent on this set.

## Baseline artifact

[`baseline.txt`](baseline.txt) is the committed "before" snapshot at the default
global 0.85 threshold. Headline: equivalents and non-equivalents are separable
(worst equivalent 0.702 > best non-equivalent 0.556), and 0.85 leaks nothing
(0/132) yet admits only 72% of true equivalents — several ZH pairs and one hard
intent fall below the global cutoff.

That last gap is the point: a single global threshold is safe but leaves hits on
the table. The cosine distributions captured here are the seed dataset for
per-language-pair threshold calibration, whose ROC/AUC "after" artifact will sit
beside this one.
