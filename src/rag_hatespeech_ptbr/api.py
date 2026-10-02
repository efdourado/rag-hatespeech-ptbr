"""Demonstração FastAPI local. Sem rede/inferência real até RAG_LIVE=1."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from rag_hatespeech_ptbr.data import load_labeled_frame
from rag_hatespeech_ptbr.llm import LLMClassifier, prediction_record
from rag_hatespeech_ptbr.openrouter import OpenRouterClient, OpenRouterError
from rag_hatespeech_ptbr.retrieval import LocalIndex
from rag_hatespeech_ptbr.runtime import make_embedder
from rag_hatespeech_ptbr.splits import calculate_sha256


class ClassificationRequest(BaseModel):
    comment: str = Field(min_length=1, max_length=10000)
    mode: Literal["baseline", "rag", "rag_no_rationales"] = "rag"
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("comment")
    @classmethod
    def reject_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Comentário vazio.")
        return value


def create_app(runtime: dict[str, Any] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if runtime is None:
            load_dotenv()
            live = os.environ.get("RAG_LIVE") == "1"
            dataset = Path(os.environ.get("RAG_DATASET", "data/raw/HateBRXplain.csv"))
            manifest = Path("data/processed/split_manifest.csv")
            frame = load_labeled_frame(dataset, manifest)
            client = (
                OpenRouterClient.from_env(
                    max_requests=int(os.environ.get("RAG_MAX_API_REQUESTS", "100"))
                )
                if live
                else None
            )
            embedder = make_embedder(
                live=live,
                client=client,
                cache_path=Path("data/interim/openrouter_embeddings.sqlite"),
            )
            index_path = Path(
                os.environ.get(
                    "RAG_INDEX",
                    "data/interim/train_index.npz"
                    if live
                    else "data/interim/train_index_dry_run.npz",
                )
            )
            index = LocalIndex.load(index_path)
            index.validate(
                frame,
                embedder,
                {
                    "dataset_source_sha256": calculate_sha256(dataset),
                    "manifest_sha256": calculate_sha256(manifest),
                },
            )
            state = {
                "frame": frame,
                "index": index,
                "embedder": embedder,
                "classifier": LLMClassifier(
                    os.environ.get("OPENROUTER_CHAT_MODEL", ""),
                    client=client,
                    dry_run=not live,
                    provider=os.environ.get("OPENROUTER_CHAT_PROVIDER") or None,
                ),
            }
        else:
            state = runtime
        app.state.runtime = state
        app.state.lock = Lock()  # demonstração serial: cache e teto de chamadas consistentes
        yield

    app = FastAPI(title="RAG HateBRXplain — demonstração", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        state = app.state.runtime
        return {
            "status": "ok",
            "dry_run": state["classifier"].dry_run,
            "train_rows": len(state["index"].ids),
        }

    @app.post("/classify")
    def classify(request: ClassificationRequest) -> dict[str, Any]:
        state = app.state.runtime
        if request.mode != "baseline" and request.top_k > len(state["index"].ids):
            raise HTTPException(422, "top_k excede o número de exemplos do treino.")
        try:
            with app.state.lock:
                examples = (
                    state["index"].search(
                        state["embedder"].embed([request.comment]), state["frame"], request.top_k
                    )
                    if request.mode != "baseline"
                    else []
                )
                prediction = state["classifier"].predict(
                    request.comment,
                    examples,
                    include_rationales=request.mode != "rag_no_rationales",
                )
        except (ValueError, OpenRouterError):
            raise HTTPException(502, "Falha de inferência; confira configuração e conta.") from None
        return {
            **prediction_record(prediction),
            "dry_run": state["classifier"].dry_run,
            "mode": request.mode,
            "retrieved": [{"id": e.id, "score": e.score} for e in examples],
        }

    return app


app = create_app()
