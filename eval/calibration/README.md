# Calibration

Per-language-pair threshold calibration — the densest single piece of technical
value in the project. Target language pairs are drawn from **TR / ES / EN / ZH**.

## Layout

- `intents.jsonl` — the labeled set: 40 support-bot intents, each expressed in
  all four languages, each with a **variant** (same intent, one structural
  token changed — `$49.99` → `$89.99` — or a semantic near-miss for the
  token-free intents). Kind `token` marks variants the structural guard should
  catch; `semantic` marks the ones it cannot.
- `main.go` — the harness. Embeds everything with the real BGE-M3 sidecar,
  scores positives / hard negatives / easy negatives per language pair, runs
  ROC analysis, picks the lowest threshold meeting the precision target, and
  measures what the guard adds on top of the threshold.
- `scores/` — raw labeled cosine scores per pair (CSV), committed.
- `roc/` — ROC curve per pair (SVG), committed.
- `thresholds.json` — the per-pair threshold table. The proxy loads it via
  `THRESHOLDS_FILE`; unknown pairs fall back to its `default` (the most
  conservative calibrated cutoff).
- `summary.md` — generated results table.
- `WRITEUP.md` — what the curves show and why one global threshold fails.

## Run

```sh
docker compose up -d sidecar   # BGE-M3 on localhost:8000
make calibrate                 # = go run ./eval/calibration
```

Flags: `-sidecar` (default `http://localhost:8000`), `-precision` (default
`0.99`), `-intents`, `-out`.
