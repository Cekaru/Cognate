// Package config loads the proxy configuration from the environment
// (12-factor). No file parsing: every knob is an env var so the same binary
// runs identically in Docker Compose and in production.
package config

import (
	"log/slog"
	"os"
	"strconv"
	"time"
)

// Config holds all runtime configuration for the proxy.
type Config struct {
	ProxyAddr       string     // address the proxy listens on
	ProviderBaseURL string     // upstream OpenAI-compatible base URL
	ProviderAPIKey  string     // upstream credential (never logged)
	EmbedSidecarURL string     // BGE-M3 embedding sidecar base URL
	DatabaseURL     string     // pgvector connection string
	LogLevel        slog.Level // structured-log verbosity

	// Cache tiers.
	CacheEnabled      bool          // master switch; false = pure passthrough
	CacheL1Capacity   int           // max entries in the L1 exact LRU
	SemanticThreshold float64       // global cutoff; fallback when no per-pair table applies
	ThresholdsFile    string        // calibrated per-language-pair threshold table (JSON); empty = global only
	CacheTTL          time.Duration // entry lifetime; 0 = no expiry

	// Tenancy. Shared cache is the deliberate default (see THREAT_MODEL.md);
	// rates are bytes/second with a burst ceiling, 0 disables the limit.
	TenantIsolation       bool    // true = every tenant gets a private cache namespace
	TenantReqBytesPerSec  float64 // inbound request-byte rate per tenant
	TenantReqBurst        int
	TenantStoreBytesPerSec float64 // cache-write byte rate per tenant (anti-flooding)
	TenantStoreBurst       int
}

// Load reads configuration from the environment, applying sane defaults.
func Load() Config {
	return Config{
		ProxyAddr:       env("PROXY_ADDR", ":8080"),
		ProviderBaseURL: env("PROVIDER_BASE_URL", "https://api.openai.com"),
		ProviderAPIKey:  env("PROVIDER_API_KEY", ""),
		EmbedSidecarURL: env("EMBED_SIDECAR_URL", "http://sidecar:8000"),
		DatabaseURL:     env("DATABASE_URL", ""),
		LogLevel:        parseLevel(env("LOG_LEVEL", "info")),

		CacheEnabled:      envBool("CACHE_ENABLED", true),
		CacheL1Capacity:   envInt("CACHE_L1_CAPACITY", 10000),
		SemanticThreshold: envFloat("SEMANTIC_THRESHOLD", 0.85),
		ThresholdsFile:    env("THRESHOLDS_FILE", ""),
		CacheTTL:          envDuration("CACHE_TTL", 24*time.Hour),

		TenantIsolation:        env("TENANT_ISOLATION", "shared") == "isolated",
		TenantReqBytesPerSec:   envFloat("TENANT_REQ_BYTES_PER_SEC", 64<<10),  // 64 KiB/s sustained
		TenantReqBurst:         envInt("TENANT_REQ_BURST", 512<<10),           // 512 KiB burst
		TenantStoreBytesPerSec: envFloat("TENANT_STORE_BYTES_PER_SEC", 32<<10), // 32 KiB/s sustained
		TenantStoreBurst:       envInt("TENANT_STORE_BURST", 8<<20),            // 8 MiB burst budget
	}
}

func envBool(key string, def bool) bool {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			return b
		}
	}
	return def
}

func envInt(key string, def int) int {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envFloat(key string, def float64) float64 {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

func envDuration(key string, def time.Duration) time.Duration {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}

func env(key, def string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return def
}

func parseLevel(s string) slog.Level {
	switch s {
	case "debug", "DEBUG":
		return slog.LevelDebug
	case "warn", "WARN":
		return slog.LevelWarn
	case "error", "ERROR":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}
