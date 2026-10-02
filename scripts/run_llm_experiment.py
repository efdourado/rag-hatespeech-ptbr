"""Compara o mesmo LLM com e sem RAG; dry-run e validação por padrão."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

from rag_hatespeech_ptbr.data import get_split, load_labeled_frame
from rag_hatespeech_ptbr.experiments import MODES, load_sarcasm_labels, run_comparison
from rag_hatespeech_ptbr.llm import LLMClassifier
from rag_hatespeech_ptbr.openrouter import OpenRouterClient
from rag_hatespeech_ptbr.retrieval import LocalIndex
from rag_hatespeech_ptbr.runtime import make_embedder
from rag_hatespeech_ptbr.splits import calculate_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/split_manifest.csv"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--semantic-retrieval",
        action="store_true",
        help="RAG semântico local com LLM simulado, sem chave/API",
    )
    parser.add_argument("--model")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-backend", choices=["local", "openrouter"])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["baseline", "rag"])
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--confirm-final-test-run", action="store_true")
    parser.add_argument("--limit", type=int, default=20, help="0 = conjunto completo")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index", type=Path)
    parser.add_argument(
        "--cache", type=Path, default=Path("data/interim/openrouter_embeddings.sqlite")
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--max-api-requests", type=int, default=100)
    parser.add_argument("--annotations", nargs="+", type=Path, default=[])
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_final_test_run:
        parser.error("Teste exige --confirm-final-test-run; calibre somente na validação.")
    if args.split == "test" and args.limit != 0:
        parser.error("Avaliação final deve usar o teste completo: --limit 0.")
    if args.limit < 0:
        parser.error("--limit não pode ser negativo.")
    if args.semantic_retrieval and args.live:
        parser.error("--semantic-retrieval é um teste sem LLM real; use somente --live para API.")
    load_dotenv()
    frame = load_labeled_frame(args.dataset, args.manifest)
    evaluation = get_split(frame, args.split).sort_values("id").sample(frac=1, random_state=42)
    if args.limit:
        evaluation = evaluation.head(args.limit)
    provenance = {
        "dataset_source_sha256": calculate_sha256(args.dataset),
        "manifest_sha256": calculate_sha256(args.manifest),
    }
    client = OpenRouterClient.from_env(max_requests=args.max_api_requests) if args.live else None
    classifier = LLMClassifier(
        args.model or os.environ.get("OPENROUTER_CHAT_MODEL", ""),
        client=client,
        provider=os.environ.get("OPENROUTER_CHAT_PROVIDER") or None,
        dry_run=not args.live,
    )
    index, embedder, index_hash = None, None, None
    if any(mode != "baseline" for mode in args.modes):
        index_path = args.index or Path(
            "data/interim/train_index.npz"
            if (args.live or args.semantic_retrieval)
            else "data/interim/train_index_dry_run.npz"
        )
        index = LocalIndex.load(index_path)
        index_hash = calculate_sha256(index_path)
        embedder = make_embedder(
            live=args.live or args.semantic_retrieval,
            client=client,
            model=args.embedding_model,
            backend="local" if args.semantic_retrieval else args.embedding_backend,
            cache_path=args.cache,
        )
    labels, annotation_hashes = load_sarcasm_labels(args.annotations)
    run_dir = args.run_dir or Path(f"outputs/experiments/openrouter-{uuid.uuid4().hex[:8]}")
    summary = run_comparison(
        frame,
        evaluation,
        classifier,
        modes=args.modes,
        run_dir=run_dir,
        provenance=provenance,
        top_k=args.top_k,
        embedder=embedder,
        index=index,
        index_sha256=index_hash,
        sarcasm_labels=labels,
        annotation_hashes=annotation_hashes,
    )
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
