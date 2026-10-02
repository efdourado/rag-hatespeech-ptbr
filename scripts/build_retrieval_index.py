"""Cria índice de treino: dry-run por padrão, --live para embeddings OpenRouter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from rag_hatespeech_ptbr.data import load_labeled_frame
from rag_hatespeech_ptbr.experiments import git_state, write_json
from rag_hatespeech_ptbr.openrouter import OpenRouterClient
from rag_hatespeech_ptbr.retrieval import build_index
from rag_hatespeech_ptbr.runtime import make_embedder
from rag_hatespeech_ptbr.splits import calculate_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/split_manifest.csv"))
    parser.add_argument("--index", type=Path)
    parser.add_argument(
        "--cache", type=Path, default=Path("data/interim/openrouter_embeddings.sqlite")
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-backend", choices=["local", "openrouter"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-api-requests", type=int, default=200)
    args = parser.parse_args()
    load_dotenv()
    path = args.index or Path(
        "data/interim/train_index.npz" if args.live else "data/interim/train_index_dry_run.npz"
    )
    if path.exists():
        raise SystemExit("Índice já existe; use-o ou escolha outro --index, sem sobrescrever.")
    frame = load_labeled_frame(args.dataset, args.manifest)
    provenance = {
        "dataset_source_sha256": calculate_sha256(args.dataset),
        "manifest_sha256": calculate_sha256(args.manifest),
    }
    import os

    backend = args.embedding_backend or os.environ.get("EMBEDDING_BACKEND", "local")
    client = (
        OpenRouterClient.from_env(max_requests=args.max_api_requests)
        if args.live and backend == "openrouter"
        else None
    )
    embedder = make_embedder(
        live=args.live,
        client=client,
        model=args.embedding_model,
        backend=backend,
        cache_path=args.cache,
        batch_size=args.batch_size,
    )
    try:
        index = build_index(frame, embedder, provenance)
        index.validate(frame, embedder, provenance)
        index.save(path)
    finally:
        if client:
            # Uma execução interrompida mantém cache e rastreio do custo já incorrido.
            log = path.with_suffix(".api_calls.json")
            previous = json.loads(log.read_text()) if log.exists() else []
            write_json(log, previous + client.calls)
    report = {**index.metadata, "index_sha256": calculate_sha256(path), "git": git_state()}
    write_json(path.with_suffix(".audit.json"), report)
    print(json.dumps({"index": str(path), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
