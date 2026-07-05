// Package exact implements the L1 cache tier: a SHA-256 keyed, in-memory LRU
// giving a zero-embedding-cost fast path for byte-identical prompts.
package exact

import (
	"container/list"
	"sync"
)

// LRU is a fixed-capacity, least-recently-used cache keyed by string. It is
// safe for concurrent use. V is stored by value; use a pointer type for large
// payloads.
type LRU[V any] struct {
	mu       sync.Mutex
	capacity int
	ll       *list.List               // front = most recently used
	items    map[string]*list.Element // key -> element
}

type pair[V any] struct {
	key   string
	value V
}

// New returns an LRU holding at most capacity entries. A capacity <= 0 disables
// eviction (unbounded); callers that want a bound must pass a positive value.
func New[V any](capacity int) *LRU[V] {
	return &LRU[V]{
		capacity: capacity,
		ll:       list.New(),
		items:    make(map[string]*list.Element),
	}
}

// Get returns the value for key and marks it most-recently-used. The bool is
// false on a miss.
func (c *LRU[V]) Get(key string) (V, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.items[key]; ok {
		c.ll.MoveToFront(el)
		return el.Value.(*pair[V]).value, true
	}
	var zero V
	return zero, false
}

// Put inserts or updates key, marking it most-recently-used and evicting the
// least-recently-used entry if capacity is exceeded.
func (c *LRU[V]) Put(key string, value V) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.items[key]; ok {
		c.ll.MoveToFront(el)
		el.Value.(*pair[V]).value = value
		return
	}
	el := c.ll.PushFront(&pair[V]{key: key, value: value})
	c.items[key] = el
	if c.capacity > 0 && c.ll.Len() > c.capacity {
		c.evictOldest()
	}
}

// Remove deletes key if present.
func (c *LRU[V]) Remove(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.items[key]; ok {
		c.removeElement(el)
	}
}

// Len reports the number of entries currently held.
func (c *LRU[V]) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.ll.Len()
}

func (c *LRU[V]) evictOldest() {
	if el := c.ll.Back(); el != nil {
		c.removeElement(el)
	}
}

func (c *LRU[V]) removeElement(el *list.Element) {
	c.ll.Remove(el)
	delete(c.items, el.Value.(*pair[V]).key)
}
