package telemetry

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"sort"
	"testing"
)

// TestAuditFieldsAreClosedSet is the threat-model guarantee in a test: an audit
// line can carry only hashed identifiers, counters, and timings. If someone
// later adds a prompt or response field to AuditEvent and routes it through
// Audit, this fails.
func TestAuditFieldsAreClosedSet(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&buf, nil))

	Audit(logger, AuditEvent{
		Tenant:     "sha256:deadbeef",
		Model:      "gpt-4o-mini",
		Tier:       "L2",
		Status:     200,
		Similarity: 0.91,
		GuardFired: false,
		PromptLang: "tr",
		EntryLang:  "es",
		TokensIn:   42,
		TokensOut:  128,
		LatencyMS:  7,
	})

	var got map[string]any
	if err := json.Unmarshal(buf.Bytes(), &got); err != nil {
		t.Fatalf("audit line is not valid JSON: %v", err)
	}

	want := []string{
		"time", "level", "msg",
		"tenant", "model", "tier", "status", "similarity",
		"guard_fired", "prompt_lang", "entry_lang",
		"tokens_in", "tokens_out", "latency_ms",
	}
	sort.Strings(want)

	keys := make([]string, 0, len(got))
	for k := range got {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	if len(keys) != len(want) {
		t.Fatalf("audit keys = %v; want exactly %v", keys, want)
	}
	for i := range want {
		if keys[i] != want[i] {
			t.Fatalf("audit keys = %v; want exactly %v", keys, want)
		}
	}

	if got["msg"] != "audit" {
		t.Fatalf("msg = %v; want \"audit\"", got["msg"])
	}
}

// TestAuditNilLoggerIsSafe: a nil logger must be a no-op, not a panic.
func TestAuditNilLoggerIsSafe(t *testing.T) {
	Audit(nil, AuditEvent{Tenant: "x"}) // must not panic
}
