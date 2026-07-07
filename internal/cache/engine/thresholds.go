package engine

import (
	"encoding/json"
	"fmt"
	"os"
)

// Thresholds is the per-language-pair cosine cutoff table. Cross-lingual
// positives score systematically lower than monolingual ones, and by different
// margins per pair, so one global threshold cannot be both safe and useful —
// the calibration run in eval/calibration produces this table empirically.
type Thresholds struct {
	// Default applies to any pair not in Pairs (including unknown languages).
	Default float64 `json:"default"`
	// Pairs maps a normalized pair key (PairKey) to that pair's cutoff.
	Pairs map[string]float64 `json:"pairs,omitempty"`
}

// PairKey returns the canonical, order-independent key for a language pair,
// e.g. PairKey("tr", "en") == PairKey("en", "tr") == "en-tr". A monolingual
// lookup yields keys like "es-es".
func PairKey(a, b string) string {
	if b < a {
		a, b = b, a
	}
	return a + "-" + b
}

// For returns the cutoff for a lookup where the incoming prompt is in
// promptLang and the stored entry was seeded in entryLang.
func (t *Thresholds) For(promptLang, entryLang string) float64 {
	if t.Pairs != nil {
		if v, ok := t.Pairs[PairKey(promptLang, entryLang)]; ok {
			return v
		}
	}
	return t.Default
}

// LoadThresholds reads a calibration-produced JSON table
// ({"default": 0.85, "pairs": {"en-tr": 0.87, ...}}). When the file's default
// is zero it is backfilled with fallback so a pairs-only table stays safe.
func LoadThresholds(path string, fallback float64) (*Thresholds, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read thresholds: %w", err)
	}
	var t Thresholds
	if err := json.Unmarshal(data, &t); err != nil {
		return nil, fmt.Errorf("parse thresholds: %w", err)
	}
	if t.Default == 0 {
		t.Default = fallback
	}
	return &t, nil
}
