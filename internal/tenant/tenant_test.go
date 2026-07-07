package tenant

import (
	"net/http/httptest"
	"testing"
)

func TestFromRequestHashesCredential(t *testing.T) {
	r1 := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r1.Header.Set("Authorization", "Bearer sk-alice")
	r2 := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r2.Header.Set("Authorization", "Bearer sk-bob")

	a, b := FromRequest(r1), FromRequest(r2)
	if a == b {
		t.Fatal("different keys must map to different tenants")
	}
	if a == "sk-alice" || len(a) != 32 {
		t.Fatalf("tenant id %q must be a truncated hash, never the raw key", a)
	}
	// Same key, either header form, same identity.
	r3 := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r3.Header.Set("X-Api-Key", "sk-alice")
	if got := FromRequest(r3); got != a {
		t.Fatalf("X-Api-Key identity %q != Authorization identity %q", got, a)
	}
}

func TestFromRequestAnonymous(t *testing.T) {
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	if got := FromRequest(r); got != Anonymous {
		t.Fatalf("no credential should map to %q, got %q", Anonymous, got)
	}
}

func TestIsolated(t *testing.T) {
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	if Isolated(r, false) {
		t.Fatal("shared deployment without header must not isolate")
	}
	r.Header.Set(IsolationHeader, "tenant")
	if !Isolated(r, false) {
		t.Fatal("header opt-out must isolate")
	}
	// The header can never opt an isolated deployment back into sharing.
	r2 := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r2.Header.Set(IsolationHeader, "shared")
	if !Isolated(r2, true) {
		t.Fatal("isolated deployment must stay isolated regardless of header")
	}
}

func TestAllowRequestThrottlesPerTenant(t *testing.T) {
	q := NewQuotas(Config{RequestBytesPerSec: 1, RequestBurst: 100})

	if !q.AllowRequest("t1", 100) {
		t.Fatal("first request within burst must pass")
	}
	if q.AllowRequest("t1", 100) {
		t.Fatal("t1 exhausted its burst; second request must be throttled")
	}
	// Another tenant has its own bucket.
	if !q.AllowRequest("t2", 100) {
		t.Fatal("t2 must not be throttled by t1's traffic")
	}
}

func TestAllowStoreQuota(t *testing.T) {
	q := NewQuotas(Config{StoreBytesPerSec: 1, StoreBurst: 1000})

	if !q.AllowStore("t1", 900) {
		t.Fatal("store within burst must pass")
	}
	if q.AllowStore("t1", 900) {
		t.Fatal("flooding beyond the byte budget must be denied")
	}
	if !q.AllowStore("t2", 900) {
		t.Fatal("t2's budget is independent of t1's")
	}
}

func TestZeroConfigDisablesLimits(t *testing.T) {
	q := NewQuotas(Config{})
	for i := 0; i < 100; i++ {
		if !q.AllowRequest("t", 1<<20) || !q.AllowStore("t", 1<<20) {
			t.Fatal("zero-valued config must disable limiting")
		}
	}
	// A nil registry (tenancy disabled entirely) also allows everything.
	var nilQ *Quotas
	if !nilQ.AllowRequest("t", 1) || !nilQ.AllowStore("t", 1) {
		t.Fatal("nil Quotas must allow everything")
	}
}
