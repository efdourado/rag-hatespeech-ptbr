"""Escolhas compartilhadas pelos comandos e pelo serviço de demonstração."""

from __future__ import annotations

import os
from pathlib import Path

from rag_hatespeech_ptbr.openrouter import OpenRouterClient
from rag_hatespeech_ptbr.retrieval import (
    LOCAL_MODEL,
    DryRunEmbedder,
    Embedder,
    LocalSemanticEmbedder,
    OpenRouterEmbedder,
)


def make_embedder(
    *,
    live: bool,
    client: OpenRouterClient | None,
    cache_path: Path,
    backend: str | None = None,
    model: str | None = None,
    batch_size: int = 32,
) -> Embedder:
    if not live:
        return DryRunEmbedder()
    backend = backend or os.environ.get("EMBEDDING_BACKEND", "local")
    if backend == "local":
        return LocalSemanticEmbedder(
            model or os.environ.get("LOCAL_EMBEDDING_MODEL", LOCAL_MODEL),
            revision=os.environ.get("LOCAL_EMBEDDING_REVISION") or None,
        )
    if backend != "openrouter" or client is None:
        raise ValueError("Backend de embeddings desconhecido ou cliente OpenRouter ausente.")
    return OpenRouterEmbedder(
        client,
        model or os.environ.get("OPENROUTER_EMBEDDING_MODEL", ""),
        cache_path,
        provider=os.environ.get("OPENROUTER_EMBEDDING_PROVIDER") or None,
        batch_size=batch_size,
    )
