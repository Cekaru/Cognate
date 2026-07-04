// Package exact implements the L1 cache tier: a SHA-256 keyed, in-memory LRU
// giving a zero-embedding-cost fast path for byte-identical prompts.
// Implemented in Phase 1 (ROADMAP.md §6).
package exact
