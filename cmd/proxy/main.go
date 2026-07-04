// Command proxy is the Polyglot Cache sidecar entrypoint.
//
// Phase 0: it stands up an OpenAI-compatible HTTP surface that passes
// requests straight through to the configured upstream provider. Later
// phases insert the L1 (exact) and L2 (semantic) cache tiers ahead of the
// upstream call. See ROADMAP.md §6.
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

	// Cache tiers (Phase 1). When disabled, proxy.New falls back to a pure
	// passthrough with a nil engine.
	var eng *engine.Engine
	if cfg.CacheEnabled {
		embedder := embed.NewClient(cfg.EmbedSidecarURL)
		eng = engine.New(embedder, engine.Config{
			L1Capacity: cfg.CacheL1Capacity,
			Threshold:  cfg.SemanticThreshold,
			TTL:        cfg.CacheTTL,
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
}
