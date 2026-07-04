#!/usr/bin/env bash
# Cross-lingual demo: a Turkish prompt is answered from a cache entry seeded by
# an equivalent Spanish prompt — with the similarity score logged and surfaced
# in response headers.
#
# Prereqs: the stack is up (`make up`) with a real upstream key, so the proxy
# can seed the first (miss) request from the real LLM. The sidecar embeds each
# prompt in its ORIGINAL language; no translation happens on the request path.
#
# Usage: PROXY=http://localhost:8080 MODEL=gpt-4o-mini bash eval/demo/crosslingual_hit.sh
set -euo pipefail

PROXY="${PROXY:-http://localhost:8080}"
MODEL="${MODEL:-gpt-4o-mini}"

ask() { # $1 = prompt text
  curl -sS -D - -o /dev/null \
    -X POST "$PROXY/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer demo-client-key' \
    -d "$(printf '{"model":"%s","messages":[{"role":"user","content":"%s"}]}' "$MODEL" "$1")"
}

echo "==> 1) Spanish prompt (expected: X-Polyglot-Cache: MISS — seeds the cache)"
ask '¿Cuál es la capital de Francia?' | grep -iE 'X-Polyglot-|HTTP/'

echo
echo "==> 2) Turkish equivalent (expected: X-Polyglot-Cache: L2, Entry-Lang: es)"
ask 'Fransa'\''nın başkenti neresidir?' | grep -iE 'X-Polyglot-|HTTP/'

echo
echo "The L2 line proves the thesis: a TR prompt served the ES-seeded answer."
echo "See the proxy's 'semantic lookup' log line for the similarity score:"
echo "  docker compose logs proxy | grep 'semantic lookup'"
