# Calibration

Per-language-pair threshold calibration — the densest single piece of technical
value in the project. Target language pairs are drawn from **TR / ES / EN / ZH**.

Planned contents:

- `pairs/` — labeled positive / hard-negative / easy-negative prompt sets per
  language pair (e.g. `tr-en`, `es-zh`, ...).
- `scores/` — cosine-similarity distributions from BGE-M3.
- `roc/` — ROC curves + AUC per language pair.
- `thresholds.json` — committed per-pair precision-tuned threshold table.
- `WRITEUP.md` — what the curves show and why one global threshold fails.
