// Package tenant handles tenant isolation and quota. Namespacing is by hashed
// key (Hash(Hash(API_Key) + Request)); a per-tenant byte quota defends against
// cache flooding. The stance is cross-tenant-by-default, guarded, with an
// opt-out flag and full audit trail (ROADMAP.md §8, §10.2, §10.5).
// Implemented in Phase 2.
package tenant
