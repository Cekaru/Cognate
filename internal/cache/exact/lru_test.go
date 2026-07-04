package exact

import (
	"strconv"
	"sync"
	"testing"
)

func TestLRUGetPut(t *testing.T) {
	c := New[int](3)
	c.Put("a", 1)
	if v, ok := c.Get("a"); !ok || v != 1 {
		t.Fatalf("Get(a) = %v, %v; want 1, true", v, ok)
	}
	if _, ok := c.Get("missing"); ok {
		t.Fatal("Get(missing) reported a hit")
	}
}

func TestLRUUpdateInPlace(t *testing.T) {
	c := New[string](2)
	c.Put("k", "one")
	c.Put("k", "two")
	if v, _ := c.Get("k"); v != "two" {
		t.Fatalf("Get(k) = %q; want two", v)
	}
	if c.Len() != 1 {
		t.Fatalf("Len = %d; want 1 after in-place update", c.Len())
	}
}

func TestLRUEvictsLeastRecentlyUsed(t *testing.T) {
	c := New[int](2)
	c.Put("a", 1)
	c.Put("b", 2)
	// Touch "a" so "b" becomes the LRU victim.
	if _, ok := c.Get("a"); !ok {
		t.Fatal("expected a to be present")
	}
	c.Put("c", 3) // evicts "b"

	if _, ok := c.Get("b"); ok {
		t.Fatal("b should have been evicted")
	}
	if _, ok := c.Get("a"); !ok {
		t.Fatal("a should have survived")
	}
	if _, ok := c.Get("c"); !ok {
		t.Fatal("c should be present")
	}
	if c.Len() != 2 {
		t.Fatalf("Len = %d; want 2", c.Len())
	}
}

func TestLRURemove(t *testing.T) {
	c := New[int](2)
	c.Put("a", 1)
	c.Remove("a")
	if _, ok := c.Get("a"); ok {
		t.Fatal("a should be gone after Remove")
	}
	c.Remove("nope") // no-op, must not panic
}

func TestLRUUnboundedWhenCapacityZero(t *testing.T) {
	c := New[int](0)
	for i := 0; i < 1000; i++ {
		c.Put(strconv.Itoa(i), i)
	}
	if c.Len() != 1000 {
		t.Fatalf("Len = %d; want 1000 (unbounded)", c.Len())
	}
}

func TestLRUConcurrent(t *testing.T) {
	c := New[int](128)
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 500; i++ {
				k := strconv.Itoa((g*500 + i) % 200)
				c.Put(k, i)
				c.Get(k)
			}
		}(g)
	}
	wg.Wait()
	if c.Len() > 128 {
		t.Fatalf("Len = %d; exceeded capacity", c.Len())
	}
}
