// Package telemetry configures structured JSON logging and the per-request audit
// event. Per the threat model the audit log MUST NOT contain plaintext prompts
// or responses — only hashed identifiers, counters, and timings. Those same
// fields double as the signal that feeds threshold calibration.
package telemetry

import (
	"context"
	"log/slog"
	"os"
)

// NewLogger returns a JSON slog logger writing to stdout at the given level.
func NewLogger(level slog.Level) *slog.Logger {
	return slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: level}))
}

// AuditEvent is the per-request audit record. Every field is a hashed
// identifier, a counter, or a timing — never a prompt or a response. The JSON
// handler stamps the timestamp, so it is not carried here.
type AuditEvent struct {
	Tenant     string  // hashed tenant identity, never a raw credential
	Model      string  // requested model id
	Tier       string  // L1 | L2 | MISS
	Status     int     // HTTP status served
	Similarity float64 // cosine similarity on an L2 hit; 0 otherwise
	GuardFired bool    // a structural-token mismatch vetoed a candidate
	PromptLang string  // detected language of the incoming prompt
	EntryLang  string  // language of the entry that served the answer
	TokensIn   int     // provider prompt tokens (from the response usage block)
	TokensOut  int     // provider completion tokens
	LatencyMS  int64   // wall-clock time to serve the request
}

// Audit writes one AuditEvent as a structured JSON line. It uses LogAttrs with
// an explicit attribute list so the record can only ever contain these fields —
// there is no path for a prompt or response body to reach the log.
func Audit(logger *slog.Logger, e AuditEvent) {
	if logger == nil {
		return
	}
	logger.LogAttrs(context.Background(), slog.LevelInfo, "audit",
		slog.String("tenant", e.Tenant),
		slog.String("model", e.Model),
		slog.String("tier", e.Tier),
		slog.Int("status", e.Status),
		slog.Float64("similarity", e.Similarity),
		slog.Bool("guard_fired", e.GuardFired),
		slog.String("prompt_lang", e.PromptLang),
		slog.String("entry_lang", e.EntryLang),
		slog.Int("tokens_in", e.TokensIn),
		slog.Int("tokens_out", e.TokensOut),
		slog.Int64("latency_ms", e.LatencyMS),
	)
}
