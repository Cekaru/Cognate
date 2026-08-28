// Command proxy is the Polyglot Cache sidecar entrypoint. It stands up an
// OpenAI-compatible HTTP surface that serves requests from the L1 (exact) and
// L2 (semantic) cache tiers, falling back to the configured upstream provider
// on a miss.
package main

import (
	"context"
	"errors"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/kaanrumin/polyglot-cache/internal/cache/engine"
	"github.com/kaanrumin/polyglot-cache/internal/cache/semantic"
	"github.com/kaanrumin/polyglot-cache/internal/config"
	"github.com/kaanrumin/polyglot-cache/internal/crypto"
	"github.com/kaanrumin/polyglot-cache/internal/embed"
	"github.com/kaanrumin/polyglot-cache/internal/provider"
	"github.com/kaanrumin/polyglot-cache/internal/proxy"
	"github.com/kaanrumin/polyglot-cache/internal/telemetry"
	"github.com/kaanrumin/polyglot-cache/internal/tenant"
)

func main() {
	cfg := config.Load()
	logger := telemetry.NewLogger(cfg.LogLevel)

	prov, err := provider.NewOpenAICompatible(cfg.ProviderBaseURL, cfg.ProviderAPIKey)
	if err != nil {
		logger.Error("failed to initialize provider", "err", err)
		os.Exit(1)
	}

	// Per-tenant limiters: request-byte rate limiting at the front door and
	// the store quota inside the engine, both keyed by hashed credential.
	quotas := tenant.NewQuotas(tenant.Config{
		RequestBytesPerSec: cfg.TenantReqBytesPerSec,
		RequestBurst:       cfg.TenantReqBurst,
		StoreBytesPerSec:   cfg.TenantStoreBytesPerSec,
		StoreBurst:         cfg.TenantStoreBurst,
	})

	// Cache tiers. When disabled, proxy.New falls back to a pure
	// passthrough with a nil engine.
	var (
		eng   *engine.Engine
		pgIdx *semantic.PostgresIndex
	)
	if cfg.CacheEnabled {
		embedder := embed.NewClient(cfg.EmbedSidecarURL)

		// L2 backend: pgvector when a database is configured (persistent across
		// restarts), otherwise an in-memory index (process-local). A failed
		// pgvector init is logged and falls back to in-memory rather than
		// refusing to start.
		var l2 semantic.Index
		if cfg.DatabaseURL != "" {
			initCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			idx, err := semantic.NewPostgresIndex(initCtx, cfg.DatabaseURL, logger)
			cancel()
			if err != nil {
				logger.Error("pgvector init failed; using in-memory L2 (cache will not persist)", "err", err)
			} else {
				l2, pgIdx = idx, idx
			}
		} else {
			logger.Warn("DATABASE_URL unset; using in-memory L2 (cache will not persist)")
		}

		// Calibrated per-language-pair thresholds; a missing or bad file is a
		// warning, not a startup failure — the global cutoff still applies.
		var ths *engine.Thresholds
		if cfg.ThresholdsFile != "" {
			t, err := engine.LoadThresholds(cfg.ThresholdsFile, cfg.SemanticThreshold)
			if err != nil {
				logger.Warn("thresholds file unusable; using global threshold only", "file", cfg.ThresholdsFile, "err", err)
			} else {
				ths = t
				logger.Info("per-language-pair thresholds loaded", "file", cfg.ThresholdsFile, "pairs", len(t.Pairs), "default", t.Default)
			}
		}

		// Encryption at rest for L2 response bodies. Enabled when the data-key
		// env var is present; a configured-but-invalid key fails startup (fail
		// closed) rather than running without the encryption the operator asked
		// for. The key is read inside the provider and never stored in config.
		var cipher engine.Cipher
		if v, ok := os.LookupEnv(crypto.DefaultKeyEnv); ok && v != "" {
			enc, err := crypto.NewEncryptor(crypto.EnvKeyProvider{})
			if err != nil {
				logger.Error("encryption key present but unusable; refusing to start", "err", err)
				os.Exit(1)
			}
			cipher = enc
			logger.Info("response encryption at rest enabled", "provider", enc.ProviderName())
		} else {
			logger.Warn("encryption at rest disabled; L2 response bodies stored in plaintext", "set", crypto.DefaultKeyEnv)
		}

		eng = engine.New(embedder, engine.Config{
			L1Capacity: cfg.CacheL1Capacity,
			Threshold:  cfg.SemanticThreshold,
			Thresholds: ths,
			TTL:        cfg.CacheTTL,
			L2:         l2,
			Quota:      quotas,
			Cipher:     cipher,
		}, logger)
	}

	srv := &http.Server{
		Addr:         cfg.ProxyAddr,
		Handler:      proxy.New(prov, eng, proxy.Tenancy{Quotas: quotas, IsolatedDefault: cfg.TenantIsolation}, logger),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second,
		IdleTimeout:  90 * time.Second,
	}

	go func() {
		logger.Info("polyglot-cache proxy listening",
			"addr", cfg.ProxyAddr,
			"provider", prov.Name(),
			"upstream", cfg.ProviderBaseURL,
			"cache_enabled", cfg.CacheEnabled,
		)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("server error", "err", err)
			os.Exit(1)
		}
	}()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	<-ctx.Done()

	logger.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		logger.Error("graceful shutdown failed", "err", err)
	}
	if pgIdx != nil {
		pgIdx.Close()
	}
}
