// Package telemetry configures structured JSON logging. Per the threat model
// the audit log MUST NOT contain plaintext prompts or responses — only hashed
// identifiers, counters, and timings.
package telemetry

import (
	"log/slog"
	"os"
)

// NewLogger returns a JSON slog logger writing to stdout at the given level.
func NewLogger(level slog.Level) *slog.Logger {
	return slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: level}))
}
