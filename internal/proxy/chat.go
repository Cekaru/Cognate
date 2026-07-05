package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"strconv"
	"strings"
	"time"

	"github.com/kaanrumin/polyglot-cache/internal/cache"
	"github.com/kaanrumin/polyglot-cache/internal/cache/engine"
	"github.com/kaanrumin/polyglot-cache/internal/lang"
	"github.com/kaanrumin/polyglot-cache/internal/provider"
)

// maxChatBody caps the request body we buffer for cache keying. Larger requests
// fall through to the passthrough proxy untouched.
const maxChatBody = 4 << 20 // 4 MiB

// chatHandler serves POST /v1/chat/completions through the cache. Streaming,
// oversized, or unparseable requests fall back to the passthrough reverse proxy.
type chatHandler struct {
	prov   provider.Provider
	eng    *engine.Engine
	rp     *httputil.ReverseProxy
	logger *slog.Logger
	client *http.Client
}

func newChatHandler(prov provider.Provider, eng *engine.Engine, rp *httputil.ReverseProxy, logger *slog.Logger) *chatHandler {
	return &chatHandler{
		prov:   prov,
		eng:    eng,
		rp:     rp,
		logger: logger,
		client: &http.Client{Timeout: 120 * time.Second},
	}
}

// chatRequest is the minimal slice of the OpenAI chat schema we need to key the
// cache. Unknown fields are ignored and forwarded verbatim in the raw body.
type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func (h *chatHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		h.rp.ServeHTTP(w, r)
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, maxChatBody+1))
	_ = r.Body.Close()
	if err != nil || len(body) > maxChatBody {
		h.fallback(w, r, body)
		return
	}

	var req chatRequest
	if err := json.Unmarshal(body, &req); err != nil || req.Stream || len(req.Messages) == 0 {
		// Not cacheable (streaming, malformed, or empty): pass straight through.
		h.fallback(w, r, body)
		return
	}

	embedText := embedText(req.Messages)
	q := engine.Query{
		TenantScope: cache.TenantScopeShared, // per-tenant isolation comes later
		Model:       req.Model,
		ExactText:   exactText(req.Messages),
		EmbedText:   embedText,
		Lang:        lang.Detect(embedText),
	}

	result, err := h.eng.Serve(r.Context(), q, func(ctx context.Context) (*engine.UpstreamResp, error) {
		return h.callUpstream(ctx, body)
	})
	if err != nil {
		h.logger.Error("upstream request failed", "err", err, "path", r.URL.Path)
		writeError(w, http.StatusBadGateway, "upstream unavailable", "bad_gateway")
		return
	}

	writeResult(w, result)
}

// fallback restores the buffered body and hands the request to the passthrough
// reverse proxy.
func (h *chatHandler) fallback(w http.ResponseWriter, r *http.Request, body []byte) {
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	h.rp.ServeHTTP(w, r)
}

// callUpstream forwards the original request body to the provider and buffers
// the full response so it can be cached. A fresh request is built so no inbound
// client credential is ever forwarded.
func (h *chatHandler) callUpstream(ctx context.Context, body []byte) (*engine.UpstreamResp, error) {
	target := h.prov.Target()
	u := *target
	u.Path = singleJoiningSlash(target.Path, "/v1/chat/completions")

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u.String(), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	h.prov.Authorize(req)

	resp, err := h.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, maxChatBody))
	if err != nil {
		return nil, err
	}
	return &engine.UpstreamResp{Status: resp.StatusCode, Header: resp.Header.Clone(), Body: respBody}, nil
}

func writeResult(w http.ResponseWriter, res *engine.Result) {
	h := w.Header()
	// Relay upstream headers on a miss; hits carry the stored JSON body.
	if res.Header != nil {
		copyHeader(h, res.Header)
	}
	if h.Get("Content-Type") == "" {
		h.Set("Content-Type", "application/json")
	}
	// Content-Length may be stale after buffering; let the server recompute.
	h.Del("Content-Length")
	h.Set("X-Polyglot-Cache", res.Tier)
	if res.Tier == "L2" {
		h.Set("X-Polyglot-Similarity", strconv.FormatFloat(res.Similarity, 'f', 4, 64))
		h.Set("X-Polyglot-Prompt-Lang", res.PromptLang)
		h.Set("X-Polyglot-Entry-Lang", res.EntryLang)
	}
	status := res.Status
	if status == 0 {
		status = http.StatusOK
	}
	w.WriteHeader(status)
	_, _ = w.Write(res.Body)
}

func writeError(w http.ResponseWriter, status int, msg, typ string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write([]byte(`{"error":{"message":"` + msg + `","type":"` + typ + `"}}`))
}

func copyHeader(dst, src http.Header) {
	for k, vs := range src {
		// Hop-by-hop and length are handled by the server, not relayed.
		if k == "Content-Length" || k == "Transfer-Encoding" || k == "Connection" {
			continue
		}
		for _, v := range vs {
			dst.Add(k, v)
		}
	}
}

// embedText is the semantic content embedded for the L2 lookup: the message
// contents joined, in their ORIGINAL language (no translate-then-embed).
func embedText(msgs []chatMessage) string {
	var b strings.Builder
	for i, m := range msgs {
		if i > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(m.Content)
	}
	return b.String()
}

// exactText is the canonical string hashed for the L1 key: role and content of
// each message, with separators that cannot appear in normal text.
func exactText(msgs []chatMessage) string {
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString(m.Role)
		b.WriteByte(0x1f) // unit separator
		b.WriteString(m.Content)
		b.WriteByte(0x1e) // record separator
	}
	return b.String()
}

// singleJoiningSlash joins a and b with exactly one slash between them.
func singleJoiningSlash(a, b string) string {
	aslash := strings.HasSuffix(a, "/")
	bslash := strings.HasPrefix(b, "/")
	switch {
	case aslash && bslash:
		return a + b[1:]
	case !aslash && !bslash:
		return a + "/" + b
	}
	return a + b
}
