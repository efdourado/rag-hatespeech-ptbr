"""Embeddings com cache e índice local de cosseno restrito ao treino."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from rag_hatespeech_ptbr.openrouter import OpenRouterClient, provider_preferences

LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


class Embedder(Protocol):
    model: str
    dry_run: bool

    def embed(self, texts: list[str]) -> np.ndarray: ...


class DryRunEmbedder:
    """Vetores de hashing para testar encanamento; não são embeddings semânticos."""

    model = "dry-run/hashing-256-v1"
    dry_run = True

    def embed(self, texts: list[str]) -> np.ndarray:
        return (
            HashingVectorizer(
                n_features=256, alternate_sign=False, analyzer="char", ngram_range=(2, 4)
            )
            .transform(texts)
            .toarray()
            .astype(np.float32)
        )


class LocalSemanticEmbedder:
    """Modelo multilíngue local; baixa pesos na primeira execução real."""

    dry_run = False

    def __init__(self, model: str, *, revision: str | None = None) -> None:
        if not model:
            raise ValueError("Defina LOCAL_EMBEDDING_MODEL.")
        self.model = model
        self.revision = revision or (LOCAL_MODEL_REVISION if model == LOCAL_MODEL else None)
        self._encoder: Any = None
        self.encoding_audit: dict[str, Any] = {}

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "backend": "sentence-transformers",
            "revision": self.revision,
            "normalize_embeddings": True,
        }

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._encoder is None:
            cache_folder = Path("data/interim/model_cache").resolve()
            os.environ.setdefault("HF_HOME", str(cache_folder / "hf_home"))
            os.environ.setdefault("HF_XET_CACHE", str(cache_folder / "xet"))
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise RuntimeError(
                    "Instale os embeddings locais: pip install -e '.[local-embeddings]'"
                ) from None
            snapshot = (
                cache_folder
                / ("models--" + self.model.replace("/", "--"))
                / "snapshots"
                / (self.revision or "unfixed")
            )
            cached = all(
                (snapshot / file).is_file()
                for file in (
                    "modules.json",
                    "config.json",
                    "model.safetensors",
                    "tokenizer.json",
                    "1_Pooling/config.json",
                )
            )
            self._encoder = SentenceTransformer(
                self.model,
                revision=self.revision,
                trust_remote_code=False,
                cache_folder=str(cache_folder),
                device="cpu",
                local_files_only=cached,
            )
        lengths = [
            len(ids)
            for ids in self._encoder.tokenizer(texts, truncation=False, padding=False)["input_ids"]
        ]
        self.encoding_audit = {
            "max_sequence_tokens": self._encoder.max_seq_length,
            "texts_exceeding_token_limit": sum(n > self._encoder.max_seq_length for n in lengths),
            "maximum_input_tokens": max(lengths),
            "device": "cpu",
        }
        return _normalize(
            self._encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        )


class OpenRouterEmbedder:
    dry_run = False

    def __init__(
        self,
        client: OpenRouterClient,
        model: str,
        cache_path: Path,
        *,
        provider: str | None = None,
        batch_size: int = 32,
    ) -> None:
        if not model or batch_size < 1:
            raise ValueError("Modelo e batch_size positivo são obrigatórios.")
        self.client, self.model, self.cache_path = client, model, cache_path
        self.provider, self.batch_size = provider, batch_size

    @property
    def identity(self) -> dict[str, Any]:
        return {"model": self.model, "provider": provider_preferences(self.provider)}

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Embeddings exigem textos não vazios.")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = [
            hashlib.sha256(
                json.dumps(
                    {**self.identity, "text": text}, sort_keys=True, ensure_ascii=False
                ).encode()
            ).hexdigest()
            for text in texts
        ]
        with sqlite3.connect(self.cache_path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
            )
            vectors: dict[str, list[float]] = {}
            missing: dict[str, str] = {}
            for key, text in zip(keys, texts):
                row = db.execute("SELECT vector FROM embeddings WHERE key=?", (key,)).fetchone()
                if row:
                    vectors[key] = json.loads(row[0])
                else:
                    missing[key] = text
            pending = list(missing)
            for start in range(0, len(pending), self.batch_size):
                batch = pending[start : start + self.batch_size]
                response = self.client.post(
                    "embeddings",
                    {
                        "model": self.model,
                        "input": [missing[key] for key in batch],
                        "encoding_format": "float",
                        "provider": provider_preferences(self.provider),
                    },
                ).payload
                data = response.get("data", [])
                if len(data) != len(batch) or sorted(
                    item.get("index", -1) for item in data
                ) != list(range(len(batch))):
                    raise ValueError("Embeddings: índices ausentes, repetidos ou inesperados.")
                ordered = sorted(data, key=lambda item: item["index"])
                array = _normalize([item["embedding"] for item in ordered])
                for key, vector in zip(batch, array.tolist()):
                    vectors[key] = vector
                    db.execute(
                        "INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (key, json.dumps(vector))
                    )
                db.commit()  # cada lote fica salvo mesmo se o seguinte falhar
        return _normalize([vectors[key] for key in keys])


def _normalize(vectors: Any) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("Embeddings devem formar uma matriz não vazia.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(array).all() or (norms <= 0).any():
        raise ValueError("Embeddings contêm valores não finitos ou vetores nulos.")
    return array / norms


@dataclass(frozen=True)
class RetrievedExample:
    id: int
    comment: str
    offensive_label: int
    rationales: list[str]
    score: float


@dataclass
class LocalIndex:
    ids: np.ndarray
    vectors: np.ndarray
    metadata: dict[str, Any]

    def validate(self, frame: pd.DataFrame, embedder: Embedder, provenance: dict[str, str]) -> None:
        expected_ids = sorted(frame.loc[frame.split.eq("train"), "id"].astype(int))
        if sorted(self.ids.tolist()) != expected_ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("Índice deve conter exatamente os IDs do treino, sem vazamento.")
        if self.metadata["embedding_model"] != embedder.model:
            raise ValueError("Modelo do índice difere do modelo de consulta.")
        if self.metadata["dry_run"] != embedder.dry_run:
            raise ValueError("Índice dry-run não pode ser usado em experimento real.")
        identity = getattr(embedder, "identity", {"model": embedder.model})
        if self.metadata["embedding_identity"] != identity:
            raise ValueError("Provedor/configuração dos embeddings difere do índice.")
        if any(self.metadata.get(key) != value for key, value in provenance.items()):
            raise ValueError("Proveniência do índice diverge do dataset/manifesto.")

    def search(self, query: np.ndarray, frame: pd.DataFrame, top_k: int) -> list[RetrievedExample]:
        if top_k < 1 or top_k > len(self.ids):
            raise ValueError("top_k fora do tamanho do índice.")
        query = _normalize(query)
        if query.shape != (1, self.vectors.shape[1]):
            raise ValueError("Dimensão da consulta difere do índice.")
        scores = self.vectors @ query[0]
        positions = np.lexsort((self.ids, -scores))[:top_k]
        rows = frame.set_index("id")
        examples = []
        for pos in positions:
            row = rows.loc[int(self.ids[pos])]
            if row["split"] != "train":
                raise ValueError("Recuperação de item fora do treino.")
            rationales = []
            for column in ("rationales_annotator1", "rationales_annotator2"):
                value = row.get(column)
                if pd.notna(value) and str(value).strip() and str(value) not in rationales:
                    rationales.append(str(value))
            examples.append(
                RetrievedExample(
                    int(self.ids[pos]),
                    str(row.comment),
                    int(row.offensive_label),
                    rationales,
                    float(scores[pos]),
                )
            )
        return examples

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Índice já existe: {path}; escolha outro caminho.")
        with path.open("xb") as handle:
            np.savez_compressed(
                handle,
                ids=self.ids,
                vectors=self.vectors,
                metadata=json.dumps(self.metadata, sort_keys=True),
            )

    @classmethod
    def load(cls, path: Path) -> LocalIndex:
        with np.load(path, allow_pickle=False) as data:
            ids = data["ids"].copy()
            vectors = _normalize(data["vectors"])
            metadata = json.loads(str(data["metadata"].item()))
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(ids) != len(vectors):
            raise ValueError("Índice corrompido: IDs/vetores incompatíveis.")
        return cls(ids, vectors, metadata)


def build_index(frame: pd.DataFrame, embedder: Embedder, provenance: dict[str, str]) -> LocalIndex:
    train = frame.loc[frame.split.eq("train")].sort_values("id")
    if train.empty or train.id.duplicated().any():
        raise ValueError("Treino vazio ou com IDs repetidos.")
    vectors = _normalize(embedder.embed(train.comment.tolist()))
    if len(vectors) != len(train):
        raise ValueError("Quantidade de embeddings diferente da quantidade de comentários.")
    return LocalIndex(
        train.id.to_numpy(dtype=np.int64),
        vectors,
        {
            **provenance,
            "source_split": "train",
            "train_rows": len(train),
            "embedding_model": embedder.model,
            "embedding_identity": getattr(embedder, "identity", {"model": embedder.model}),
            "dry_run": embedder.dry_run,
            "dimensions": vectors.shape[1],
            "metric": "cosine",
            "embedded_fields": ["comment"],
            "encoding_audit": getattr(embedder, "encoding_audit", {}),
        },
    )
