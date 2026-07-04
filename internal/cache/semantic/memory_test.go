package semantic

import (
	"context"
	"math"
	"testing"
)

func vec(vals ...float32) []float32 { return vals }

func TestSearchReturnsNearestByCosine(t *testing.T) {
	ctx := context.Background()
	ix := NewMemoryIndex()
	_ = ix.Add(ctx, &Entry{Key: "far", Model: "m", TenantScope: "shared", Embedding: vec(0, 1, 0)})
	_ = ix.Add(ctx, &Entry{Key: "near", Model: "m", TenantScope: "shared", Embedding: vec(1, 0.05, 0)})

	got, sim, err := ix.Search(ctx, vec(1, 0, 0), "m", "shared")
	if err != nil {
		t.Fatal(err)
	}
	if got == nil || got.Key != "near" {
		t.Fatalf("Search picked %v; want near", got)
	}
	if sim < 0.99 {
		t.Fatalf("similarity = %v; want ~1 for near-parallel vectors", sim)
	}
}

func TestSearchIsScaleInvariant(t *testing.T) {
	ctx := context.Background()
	ix := NewMemoryIndex()
	_ = ix.Add(ctx, &Entry{Key: "a", Model: "m", TenantScope: "shared", Embedding: vec(2, 0, 0)})
	_, sim, err := ix.Search(ctx, vec(9, 0, 0), "m", "shared")
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(sim-1.0) > 1e-6 {
		t.Fatalf("cosine of collinear vectors = %v; want 1", sim)
	}
}

func TestSearchFiltersByModel(t *testing.T) {
	ctx := context.Background()
	ix := NewMemoryIndex()
	_ = ix.Add(ctx, &Entry{Key: "other-model", Model: "gpt-x", TenantScope: "shared", Embedding: vec(1, 0, 0)})
	if got, _, _ := ix.Search(ctx, vec(1, 0, 0), "gpt-y", "shared"); got != nil {
		t.Fatalf("Search crossed models: got %v", got)
	}
}

func TestSearchFiltersByScope(t *testing.T) {
	ctx := context.Background()
	ix := NewMemoryIndex()
	_ = ix.Add(ctx, &Entry{Key: "tenant-a", Model: "m", TenantScope: "tenantA", Embedding: vec(1, 0, 0)})

	if got, _, _ := ix.Search(ctx, vec(1, 0, 0), "m", "tenantB", "shared"); got != nil {
		t.Fatalf("tenantB saw tenantA's entry: %v", got)
	}
	if got, _, _ := ix.Search(ctx, vec(1, 0, 0), "m", "tenantA", "shared"); got == nil {
		t.Fatal("tenantA should see its own entry")
	}
}

func TestSearchEmptyIndex(t *testing.T) {
	if got, sim, _ := NewMemoryIndex().Search(context.Background(), vec(1, 0, 0), "m", "shared"); got != nil || sim != 0 {
		t.Fatalf("empty index returned %v, %v", got, sim)
	}
}

func TestSearchZeroQueryVector(t *testing.T) {
	ctx := context.Background()
	ix := NewMemoryIndex()
	_ = ix.Add(ctx, &Entry{Key: "a", Model: "m", TenantScope: "shared", Embedding: vec(1, 0, 0)})
	if got, sim, _ := ix.Search(ctx, vec(0, 0, 0), "m", "shared"); got != nil || sim != 0 {
		t.Fatalf("zero query vector returned %v, %v", got, sim)
	}
}
