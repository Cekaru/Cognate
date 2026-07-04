package semantic

import (
	"math"
	"testing"
)

func vec(vals ...float32) []float32 { return vals }

func TestSearchReturnsNearestByCosine(t *testing.T) {
	ix := NewMemoryIndex()
	ix.Add(&Entry{Key: "far", Model: "m", TenantScope: "shared", Embedding: vec(0, 1, 0)})
	ix.Add(&Entry{Key: "near", Model: "m", TenantScope: "shared", Embedding: vec(1, 0.05, 0)})

	got, sim := ix.Search(vec(1, 0, 0), "m", "shared")
	if got == nil || got.Key != "near" {
		t.Fatalf("Search picked %v; want near", got)
	}
	if sim < 0.99 {
		t.Fatalf("similarity = %v; want ~1 for near-parallel vectors", sim)
	}
}

func TestSearchIsScaleInvariant(t *testing.T) {
	ix := NewMemoryIndex()
	ix.Add(&Entry{Key: "a", Model: "m", TenantScope: "shared", Embedding: vec(2, 0, 0)})
	_, sim := ix.Search(vec(9, 0, 0), "m", "shared")
	if math.Abs(sim-1.0) > 1e-6 {
		t.Fatalf("cosine of collinear vectors = %v; want 1", sim)
	}
}

func TestSearchFiltersByModel(t *testing.T) {
	ix := NewMemoryIndex()
	ix.Add(&Entry{Key: "other-model", Model: "gpt-x", TenantScope: "shared", Embedding: vec(1, 0, 0)})
	if got, _ := ix.Search(vec(1, 0, 0), "gpt-y", "shared"); got != nil {
		t.Fatalf("Search crossed models: got %v", got)
	}
}

func TestSearchFiltersByScope(t *testing.T) {
	ix := NewMemoryIndex()
	ix.Add(&Entry{Key: "tenant-a", Model: "m", TenantScope: "tenantA", Embedding: vec(1, 0, 0)})

	if got, _ := ix.Search(vec(1, 0, 0), "m", "tenantB", "shared"); got != nil {
		t.Fatalf("tenantB saw tenantA's entry: %v", got)
	}
	if got, _ := ix.Search(vec(1, 0, 0), "m", "tenantA", "shared"); got == nil {
		t.Fatal("tenantA should see its own entry")
	}
}

func TestSearchEmptyIndex(t *testing.T) {
	if got, sim := NewMemoryIndex().Search(vec(1, 0, 0), "m", "shared"); got != nil || sim != 0 {
		t.Fatalf("empty index returned %v, %v", got, sim)
	}
}

func TestSearchZeroQueryVector(t *testing.T) {
	ix := NewMemoryIndex()
	ix.Add(&Entry{Key: "a", Model: "m", TenantScope: "shared", Embedding: vec(1, 0, 0)})
	if got, sim := ix.Search(vec(0, 0, 0), "m", "shared"); got != nil || sim != 0 {
		t.Fatalf("zero query vector returned %v, %v", got, sim)
	}
}
