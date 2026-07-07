# What the calibration shows

BGE-M3 cosine scores over 40 support-bot intents in TR/ES/EN/ZH; per language
pair: 40 cross-lingual positives, 80 hard negatives (same intent, one
structural token changed — or a semantic near-miss), 80 easy negatives
(unrelated intents). Precision target 0.99. Numbers below are from
`summary.md` / `scores/`; curves in `roc/`.

## 1. One global threshold cannot work

Cross-lingual positives score lower the further apart the languages are, and
the drift is large relative to the usable threshold band:

| pair  | positive mean | positive min |
|-------|---------------|--------------|
| en-es | 0.925         | 0.793        |
| en-tr | 0.886         | 0.704        |
| en-zh | 0.882         | 0.736        |
| es-tr | 0.880         | 0.707        |
| es-zh | 0.866         | 0.717        |
| tr-zh | 0.847         | 0.663        |

A cutoff tuned for en-es (0.878) throws away roughly a third of genuine tr-zh
hits; a cutoff tuned for tr-zh (0.848) admits en-es near-misses. Hence the
per-pair table in `thresholds.json`, loaded by the proxy via `THRESHOLDS_FILE`.

## 2. The embedding cannot see structural tokens

Easy negatives are trivially separable (max 0.62–0.69, far below every
threshold): *topic* separation is not the problem. Hard negatives are: change
one number in an otherwise identical prompt ($49.99 → $89.99, order #88213 →
#55107) and BGE-M3 still scores the pair 0.78–0.93 — **inside** the positive
range. The distributions overlap; no threshold separates them. At the 0.99
precision target the embedding alone must sit so high that recall collapses to
20–52% depending on the pair (see `summary.md`, "thr (embed only)").

That is the quantified version of the project's core safety claim: a semantic
cache that serves `transfer $1000` for `transfer $100` at 0.9 cosine is not a
cache, it is an incident.

## 3. The guard buys the recall back

The structural token guard (locale-normalized numbers, dates, currencies, IDs,
code identifiers) caught **68/68** token-kind hard negatives in every language
pair, with **0/40** false fires on positives. Because the engine runs the
guard after the threshold, guard-caught negatives are unservable and the
shipped threshold only has to separate what the guard cannot see:

| pair  | embed-only recall @0.99p | with guard | shipped threshold |
|-------|--------------------------|------------|-------------------|
| en-es | 50%                      | **92%**    | 0.8778            |
| en-tr | 40%                      | **75%**    | 0.8665            |
| en-zh | 52%                      | **70%**    | 0.8638            |
| es-tr | 35%                      | **68%**    | 0.8727            |
| es-zh | 28%                      | **65%**    | 0.8586            |
| tr-zh | 20%                      | **55%**    | 0.8476            |

Measured precision at these operating points is 100% on this set. The
threshold and the guard are not redundant layers; they are complementary
failure detectors — the threshold rejects *different intents*, the guard
rejects *same intent, different facts*.

## 4. Locale traps are real, not theoretical

Two concrete cases from building this set:

- **Article-一:** ZH 一个包裹 is "a package", not "1 package". A naive
  extractor turns every 一个/一件 into the number 1, mismatching the English
  side (which extracts nothing from "a") — it fired on 3/240 positives before
  the fix. English digit-word compounds (2-year) and Turkish apostrophe
  suffixes (#4821'e) are the same class of trap.
- **Date order is locale-bound:** 03/05/2024 is March 5 in English and May 3
  in Turkish. The guard normalizes with the *prompt's* locale, so the same
  written form correctly refuses to match across those locales.

## 5. Honest limits

- What the guard cannot see stays invisible: semantic near-misses (*cancel* vs
  *pause* subscription) pass the guard and are the reason the with-guard
  thresholds cannot drop further. They bound the achievable recall.
- 40 intents × 4 languages is a calibration set, not a benchmark; "100%
  precision" means 0 leaks in ~200 negatives per pair, which certifies the
  ~0.99 target, not more. Rerun `make calibrate` on a workload-shaped set
  before trusting the numbers elsewhere.
- Thresholds are model-bound (BAAI/bge-m3). A different embedder needs a
  fresh run.
