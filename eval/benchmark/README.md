# Benchmark

The headline measurement: on a multilingual workload, how much does cross-lingual
caching win over an English-only cache — and does the structural guard keep that
win honest?

## Run it

```bash
go run ./eval/benchmark
```

No API key, no Docker, no network. It reads the real BGE-M3 cosine scores already
measured and committed by the calibration run (`../calibration/scores/*.csv`) and
the shipped per-pair thresholds (`../calibration/thresholds.json`), then writes
[`summary.md`](summary.md).

## What it models

Every row in the score CSVs is one incoming cross-lingual query paired with its
nearest cache candidate, carrying the real embedding score, the structural-guard
verdict, and a ground-truth label:

- **positive** — semantically equivalent prompts; serving is a correct hit.
- **hard-negative** — same topic, one structural token differs (`$100` vs
  `$1000`, order `#A` vs `#B`); serving is a **wrong answer**.
- **easy-negative** — unrelated intents; serving is a wrong answer.

Three systems are scored over that workload:

| System | Serves when |
|--------|-------------|
| `polyglot` (shipped) | `score ≥ pairThreshold` **and** the guard passes |
| `polyglot-noguard` | `score ≥ pairThreshold` (guard disabled) |
| `english-only` | never — an English-only cache can't match a cross-lingual query |

## Result (current run)

| Metric | Polyglot | English-only baseline |
|--------|:--------:|:---------------------:|
| Cross-lingual hit rate | **70.8%** | 0.0% (by construction) |
| Unsafe serves (leaks) | **0.0%** | 0.0% |
| Guard off → on, unsafe serves | 84 → **0** | — |
| Net cost saved / 1M queries | **~$82** | $0 |

Full per-pair breakdown and the cost model in [`summary.md`](summary.md).

**Honesty rule:** the win is sold as **cost**, not latency — a translated path
adds embedding overhead, so total latency is *higher*. The cost model is
illustrative and overridable via flags (`-usd-out-per-m`, `-completion-tokens`,
…); embedding is self-hosted, so the real overhead trends below the conservative
hosted rate used by default.
