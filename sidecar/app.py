"""BGE-M3 embedding sidecar for Polyglot Cache.

Exposes a minimal HTTP API the Go proxy calls over localhost:

    GET  /health  -> readiness probe
    POST /embed   -> {"model", "dim", "embeddings": [[...1024 floats...], ...]}

Core principle (ROADMAP.md §1): we do NOT translate-then-embed. Each text is
embedded in its ORIGINAL language; BGE-M3 places cross-lingual equivalents near
each other in the same vector space.

The model is loaded lazily on the first /embed call so the container becomes
healthy immediately during `docker compose up` (Phase 0) without waiting on a
multi-gigabyte model download.
"""
from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024

app = FastAPI(title="Polyglot Cache Embedding Sidecar", version="0.1.0")

_model: Optional[object] = None


def get_model():
    """Load BGE-M3 on first use (l azy)."""
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel

        _model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)
    return _model


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    model: str
    dim: int
    embeddings: List[List[float]]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "loaded": _model is not None}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    model = get_model()
    out = model.encode(req.texts, batch_size=16, max_length=8192)
    embeddings = [vec.tolist() for vec in out["dense_vecs"]]
    return EmbedResponse(model=MODEL_NAME, dim=EMBEDDING_DIM, embeddings=embeddings)
