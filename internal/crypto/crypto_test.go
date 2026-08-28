package crypto

import (
	"bytes"
	"encoding/base64"
	"testing"
)

// staticProvider is a test KeyProvider with a fixed key.
type staticProvider struct {
	key  []byte
	name string
}

func (s staticProvider) DataKey() ([]byte, error) { return s.key, nil }
func (s staticProvider) Name() string             { return s.name }

func newTestEncryptor(t *testing.T) *Encryptor {
	t.Helper()
	enc, err := NewEncryptor(staticProvider{key: bytes.Repeat([]byte{0x2a}, 32), name: "test"})
	if err != nil {
		t.Fatalf("NewEncryptor: %v", err)
	}
	return enc
}

func TestSealOpenRoundTrip(t *testing.T) {
	enc := newTestEncryptor(t)
	for _, pt := range [][]byte{
		[]byte(""),
		[]byte("hello"),
		[]byte("Fransa'nın başkenti Paris'tir."),
		[]byte("¿Cuál es la capital? 一百"),
		bytes.Repeat([]byte{0xff}, 4096),
	} {
		blob, err := enc.Seal(pt)
		if err != nil {
			t.Fatalf("Seal: %v", err)
		}
		got, err := enc.Open(blob)
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		if !bytes.Equal(got, pt) {
			t.Fatalf("round trip mismatch: got %q want %q", got, pt)
		}
	}
}

func TestSealIsNondeterministic(t *testing.T) {
	enc := newTestEncryptor(t)
	pt := []byte("identical plaintext")
	a, err := enc.Seal(pt)
	if err != nil {
		t.Fatalf("Seal a: %v", err)
	}
	b, err := enc.Seal(pt)
	if err != nil {
		t.Fatalf("Seal b: %v", err)
	}
	if bytes.Equal(a, b) {
		t.Fatal("two seals of the same plaintext produced identical blobs; nonce not random")
	}
	// The ciphertext must not contain the plaintext.
	if bytes.Contains(a, pt) {
		t.Fatal("ciphertext leaks plaintext bytes")
	}
}

func TestOpenRejectsTampering(t *testing.T) {
	enc := newTestEncryptor(t)
	blob, err := enc.Seal([]byte("authentic"))
	if err != nil {
		t.Fatalf("Seal: %v", err)
	}
	blob[len(blob)-1] ^= 0x01 // flip a tag bit
	if _, err := enc.Open(blob); err == nil {
		t.Fatal("Open accepted a tampered blob; GCM tag not enforced")
	}
}

func TestOpenRejectsWrongKey(t *testing.T) {
	a, _ := NewEncryptor(staticProvider{key: bytes.Repeat([]byte{0x01}, 32), name: "a"})
	b, _ := NewEncryptor(staticProvider{key: bytes.Repeat([]byte{0x02}, 32), name: "b"})
	blob, err := a.Seal([]byte("secret"))
	if err != nil {
		t.Fatalf("Seal: %v", err)
	}
	if _, err := b.Open(blob); err == nil {
		t.Fatal("Open with the wrong key succeeded")
	}
}

func TestOpenRejectsShortBlob(t *testing.T) {
	enc := newTestEncryptor(t)
	if _, err := enc.Open([]byte{0x00, 0x01}); err == nil {
		t.Fatal("Open accepted a blob shorter than the nonce")
	}
}

func TestNewEncryptorRejectsBadKeyLength(t *testing.T) {
	if _, err := NewEncryptor(staticProvider{key: []byte("too short"), name: "x"}); err == nil {
		t.Fatal("NewEncryptor accepted a non-32-byte key")
	}
}

func TestEnvKeyProvider(t *testing.T) {
	key := bytes.Repeat([]byte{0x11}, 32)
	t.Setenv(DefaultKeyEnv, base64.StdEncoding.EncodeToString(key))
	got, err := EnvKeyProvider{}.DataKey()
	if err != nil {
		t.Fatalf("DataKey: %v", err)
	}
	if !bytes.Equal(got, key) {
		t.Fatal("DataKey returned the wrong key")
	}
	if (EnvKeyProvider{}).Name() != "env" {
		t.Fatal("unexpected provider name")
	}
}

func TestEnvKeyProviderErrors(t *testing.T) {
	t.Run("missing", func(t *testing.T) {
		t.Setenv(DefaultKeyEnv, "")
		if _, err := (EnvKeyProvider{}).DataKey(); err == nil {
			t.Fatal("expected error for unset key")
		}
	})
	t.Run("bad base64", func(t *testing.T) {
		t.Setenv(DefaultKeyEnv, "not!base64!")
		if _, err := (EnvKeyProvider{}).DataKey(); err == nil {
			t.Fatal("expected error for malformed base64")
		}
	})
	t.Run("wrong length", func(t *testing.T) {
		t.Setenv(DefaultKeyEnv, base64.StdEncoding.EncodeToString([]byte("16-byte-key-here")))
		if _, err := (EnvKeyProvider{}).DataKey(); err == nil {
			t.Fatal("expected error for non-32-byte key")
		}
	})
	t.Run("custom var", func(t *testing.T) {
		key := bytes.Repeat([]byte{0x22}, 32)
		t.Setenv("CUSTOM_KEY", base64.StdEncoding.EncodeToString(key))
		got, err := EnvKeyProvider{EnvVar: "CUSTOM_KEY"}.DataKey()
		if err != nil {
			t.Fatalf("DataKey: %v", err)
		}
		if !bytes.Equal(got, key) {
			t.Fatal("custom-var key mismatch")
		}
	})
}
