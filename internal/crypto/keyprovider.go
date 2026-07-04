// Package crypto provides AES-256-GCM encryption at rest behind a pluggable
// KeyProvider. Phase 0 defines the interface; the env-var provider ships in
// Phase 4, with KMS/Vault envelope encryption documented as the production
// design (ROADMAP.md §4, §10.4). The lookup key stays plaintext-hashed so the
// index still works; only the value is encrypted.
package crypto

// KeyProvider supplies the 32-byte data-encryption key used for AES-256-GCM.
type KeyProvider interface {
	// DataKey returns a 32-byte key for AES-256.
	DataKey() ([]byte, error)
	// Name identifies the provider for audit logs (never returns key material).
	Name() string
}
