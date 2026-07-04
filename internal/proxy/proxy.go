// Package proxy is the OpenAI-compatible HTTP front door. In Phase 0 it is a
// pure passthrough reverse proxy; the L1/L2 cache lookup (ROADMAP.md §3) is
// inserted ahead of the upstream call in later phases.
package proxy

import (
	"log/slog"
	"net/http"
	"net/http/httputil"
	"time"

	"github.com/kaanrumin/polyglot-cache/internal/cache/engine"
	"github.com/kaanrumin/polyglot-cache/internal/provider"
)

// New builds the proxy HTTP handler. When eng is non-nil, POST
// /v1/chat/completions is served through the cache tiers; everything else on
// /v1/ (and streaming or unparseable chat requests) is a passthrough reverse
// proxy. A nil eng makes the whole surface a passthrough (ROADMAP.md §6).
func New(prov provider.Provider, eng *engine.Engine, logger *slog.Logger) http.Handler {
	rp := newReverseProxy(prov, logger)

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", healthz)
	if eng != nil {
		mux.Handle("/v1/chat/completions", newChatHandler(prov, eng, rp, logger))
	}
	// OpenAI-compatible surface. All other paths pass straight through.
	mux.Handle("/v1/", rp)
	return withLogging(logger, mux)
}

func healthz(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}

func newReverseProxy(prov provider.Provider, logger *slog.Logger) *httputil.ReverseProxy {
	target := prov.Target()
	return &httputil.ReverseProxy{
		Director: func(r *http.Request) {
			r.URL.Scheme = target.Scheme
			r.URL.Host = target.Host
			r.Host = target.Host
			// Strip the inbound client credential and stamp the upstream one.
			// (The client key becomes the tenant identity in Phase 2; for now
			// it is simply not forwarded.)
			r.Header.Del("Authorization")
			prov.Authorize(r)
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			logger.Error("upstream request failed", "err", err, "path", r.URL.Path)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadGateway)
			_, _ = w.Write([]byte(`{"error":{"message":"upstream unavailable","type":"bad_gateway"}}`))
		},
	}
}

// withLogging emits one structured audit line per request. No prompt or
// response bodies are logged (ROADMAP.md §10.7).
func withLogging(logger *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		sw := &statusWriter{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(sw, r)
		logger.Info("request",
			"method", r.Method,
			"path", r.URL.Path,
			"status", sw.status,
			"latency_ms", time.Since(start).Milliseconds(),
		)
	})
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}
