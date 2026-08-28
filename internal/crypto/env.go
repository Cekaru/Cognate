package crypto

import (
	"encoding/base64"
	"errors"
	"fmt"
	"os"
)

// EnvKeyProvider is the single-node KeyProvider: it reads a base64-encoded
// 32-byte data key from an environment variable. It is the development and
// small-deployment path; production is expected to plug a KMS/Vault provider
// behind the same interface (envelope encryption — a KEK unwraps this DEK),
// which is why the value, not the lookup key, is all that is encrypted.
//
// The key is read on demand and never logged; only Name() is safe to print.
type EnvKeyProvider struct {
	// EnvVar is the environment variable holding the base64 key. Defaults to
	// DefaultKeyEnv when empty.
	EnvVar string
}

// DefaultKeyEnv is the environment variable EnvKeyProvider reads by default.
const DefaultKeyEnv = "POLYGLOT_ENCRYPTION_KEY"

// DataKey decodes the base64 key from the environment and verifies it is 32
// bytes. A missing or malformed key is an error, so encryption fails closed
// rather than silently degrading to plaintext.
func (p EnvKeyProvider) DataKey() ([]byte, error) {
	name := p.EnvVar
	if name == "" {
		name = DefaultKeyEnv
	}
	raw, ok := os.LookupEnv(name)
	if !ok || raw == "" {
		return nil, fmt.Errorf("crypto: %s is not set", name)
	}
	key, err := base64.StdEncoding.DecodeString(raw)
	if err != nil {
		return nil, fmt.Errorf("crypto: %s is not valid base64: %w", name, err)
	}
	if len(key) != 32 {
		return nil, errors.New("crypto: decoded key must be 32 bytes for AES-256")
	}
	return key, nil
}

// Name identifies the provider in audit logs.
func (p EnvKeyProvider) Name() string { return "env" }
