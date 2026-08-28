package crypto

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"errors"
	"fmt"
	"io"
)

// Encryptor seals and opens cache values with AES-256-GCM. The 32-byte data key
// comes from a KeyProvider, so the same code path serves an env-var key on a
// single node and a KMS/Vault-wrapped key in production without change. Only the
// stored VALUE is encrypted; the lookup key stays a plaintext hash, so the index
// still works.
//
// The sealed blob is nonce || ciphertext || tag: a fresh random 12-byte nonce is
// prepended to every value, so identical responses encrypt to different bytes
// and equal-response correlation leaks nothing at rest.
type Encryptor struct {
	provider KeyProvider
	aead     cipher.AEAD
}

// NewEncryptor pulls the data key from provider and builds the AES-256-GCM AEAD.
// It fails if the key is not exactly 32 bytes, so a misconfigured key is caught
// at startup rather than corrupting the cache.
func NewEncryptor(provider KeyProvider) (*Encryptor, error) {
	if provider == nil {
		return nil, errors.New("crypto: nil key provider")
	}
	key, err := provider.DataKey()
	if err != nil {
		return nil, fmt.Errorf("crypto: data key: %w", err)
	}
	if len(key) != 32 {
		return nil, fmt.Errorf("crypto: data key must be 32 bytes for AES-256, got %d", len(key))
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("crypto: new cipher: %w", err)
	}
	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("crypto: new gcm: %w", err)
	}
	return &Encryptor{provider: provider, aead: aead}, nil
}

// Seal encrypts plaintext, returning nonce || ciphertext || tag.
func (e *Encryptor) Seal(plaintext []byte) ([]byte, error) {
	nonce := make([]byte, e.aead.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, fmt.Errorf("crypto: nonce: %w", err)
	}
	// Seal appends the ciphertext+tag to nonce, so the nonce is the prefix of the
	// returned blob and Open can slice it back off.
	return e.aead.Seal(nonce, nonce, plaintext, nil), nil
}

// Open reverses Seal. A blob shorter than the nonce, or one that fails the GCM
// authentication tag (tampered or wrong key), returns an error and no plaintext.
func (e *Encryptor) Open(blob []byte) ([]byte, error) {
	ns := e.aead.NonceSize()
	if len(blob) < ns {
		return nil, errors.New("crypto: ciphertext shorter than nonce")
	}
	nonce, ct := blob[:ns], blob[ns:]
	plaintext, err := e.aead.Open(nil, nonce, ct, nil)
	if err != nil {
		return nil, fmt.Errorf("crypto: open: %w", err)
	}
	return plaintext, nil
}

// ProviderName identifies the key source for audit logs; it never exposes key
// material.
func (e *Encryptor) ProviderName() string { return e.provider.Name() }
