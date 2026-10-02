"""Consulta o índice semântico local sem chamar o LLM/OpenRouter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from rag_hatespeech_ptbr.data import load_labeled_frame
from rag_hatespeech_ptbr.retrieval import LocalIndex
from rag_hatespeech_ptbr.runtime import make_embedder
from rag_hatespeech_ptbr.splits import calculate_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comment", required=True)
    parser.add_argument("--index", type=Path, default=Path("data/interim/train_index.npz"))
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/HateBRXplain.csv"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    load_dotenv()
    frame = load_labeled_frame(args.dataset, Path("data/processed/split_manifest.csv"))
    index = LocalIndex.load(args.index)
    identity = index.metadata["embedding_identity"]
    if not index.metadata["dry_run"] and identity.get("backend") != "sentence-transformers":
        parser.error("Este comando consulta somente embeddings locais, sem chamadas de API.")
    embedder = make_embedder(
        live=not index.metadata["dry_run"],
        client=None,
        backend="local",
        model=index.metadata["embedding_model"],
        cache_path=Path("data/interim/openrouter_embeddings.sqlite"),
    )
    index.validate(
        frame,
        embedder,
        {
            "dataset_source_sha256": calculate_sha256(args.dataset),
            "manifest_sha256": calculate_sha256(Path("data/processed/split_manifest.csv")),
        },
    )
    examples = index.search(embedder.embed([args.comment]), frame, args.top_k)
    print(
        json.dumps(
            {
                "embedding_model": embedder.model,
                "dry_run": embedder.dry_run,
                "matches": [{"id": e.id, "score": e.score} for e in examples],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
