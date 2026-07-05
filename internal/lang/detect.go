// Package lang provides a lightweight, script-based language guess for the
// four target languages (TR/ES/EN/ZH).
//
// This is a placeholder used only to label log lines and cache entries so the
// cross-lingual hit is legible (e.g. "tr prompt served es-cached answer"). It
// is deliberately cheap and heuristic; the real locale-aware handling —
// including the structural-token guard and per-pair thresholds — comes later.
// Do not use this for a security decision.
package lang

import "unicode"

// Codes returned by Detect.
const (
	TR      = "tr"
	ES      = "es"
	EN      = "en"
	ZH      = "zh"
	Unknown = "und"
)

// Detect returns a best-effort language code for s. The heuristic order is:
//   - any Han character  -> zh
//   - Turkish-specific letters (ı İ ğ ş) -> tr
//   - Spanish-specific marks (ñ ¿ ¡)     -> es
//   - otherwise, if it contains letters  -> en
//   - empty / letterless                 -> und
func Detect(s string) string {
	var hasLetter bool
	for _, r := range s {
		switch {
		case unicode.Is(unicode.Han, r):
			return ZH
		case r == 'ı' || r == 'İ' || r == 'ğ' || r == 'Ğ' || r == 'ş' || r == 'Ş':
			return TR
		case r == 'ñ' || r == 'Ñ' || r == '¿' || r == '¡':
			return ES
		}
		if unicode.IsLetter(r) {
			hasLetter = true
		}
	}
	if hasLetter {
		return EN
	}
	return Unknown
}
