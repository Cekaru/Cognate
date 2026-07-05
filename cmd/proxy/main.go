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
	"github.com/kaanrumin/polyglot-cache/internal/embed"
	"github.com/kaanrumin/polyglot-cache/internal/provider"
	"github.com/kaanrumin/polyglot-cache/internal/proxy"
	"github.com/kaanrumin/polyglot-cache/internal/telemetry"
)

func main() {
	cfg := config.Load()
	logger := telemetry.NewLogger(cfg.LogLevel)

	prov, err := provider.NewOpenAICompatible(cfg.ProviderBaseURL, cfg.ProviderAPIKey)
	if err != nil {
		logger.Error("failed to initialize provider", "err", err)
		os.Exit(1)
	}

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

		eng = engine.New(embedder, engine.Config{
			L1Capacity: cfg.CacheL1Capacity,
			Threshold:  cfg.SemanticThreshold,
			TTL:        cfg.CacheTTL,
			L2:         l2,
		}, logger)
	}

	srv := &http.Server{
		Addr:         cfg.ProxyAddr,
		Handler:      proxy.New(prov, eng, logger),
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
