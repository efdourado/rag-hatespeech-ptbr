"""Treina e avalia o baseline determinístico (TF-IDF + regressão logística).

Por padrão avalia no conjunto de validação. Avaliar no conjunto de teste
exige --confirm-final-test-run, porque o protocolo de pesquisa
(docs/9_protocolo_pesquisa.md) prevê uma única execução final no teste; cada
uso no teste fica registrado em outputs/experiments/test_evaluations_ledger.jsonl
para tornar visível qualquer avaliação repetida.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

from rag_hatespeech_ptbr.baseline import MODEL_NAME, RANDOM_STATE, evaluate_baseline, train_baseline
from rag_hatespeech_ptbr.data import get_split, load_labeled_frame
from rag_hatespeech_ptbr.splits import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SOURCE_SHA256,
    calculate_sha256,
)

TEST_LEDGER_PATH = Path("outputs/experiments/test_evaluations_ledger.jsonl")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="HateBRXplain.csv bruto")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/split_manifest.csv"),
        help="manifesto local de splits (id, fold, split)",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="conjunto usado para avaliação; 'test' exige --confirm-final-test-run",
    )
    parser.add_argument(
        "--confirm-final-test-run",
        action="store_true",
        help="obrigatório para avaliar no split de teste (ver docstring do módulo)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="destino do relatório agregado (padrão: outputs/tables/baseline_<split>.json)",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("outputs/experiments"),
        help="diretório onde o registro bruto da execução é salvo (não versionado)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.split == "test" and not args.confirm_final_test_run:
        raise SystemExit(
            "Avaliação no split de teste requer --confirm-final-test-run. "
            "O protocolo de pesquisa prevê uma única execução final no teste "
            "(docs/9_protocolo_pesquisa.md); confirme explicitamente que esta é essa execução."
        )

    frame = load_labeled_frame(args.dataset, args.manifest)
    train_frame = get_split(frame, "train")
    eval_frame = get_split(frame, args.split)

    pipeline = train_baseline(train_frame, random_state=RANDOM_STATE)
    metrics = evaluate_baseline(pipeline, eval_frame)

    record = {
        "experiment_id": f"{MODEL_NAME}-{uuid.uuid4().hex[:8]}",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "model_name": MODEL_NAME,
        "hyperparameters": {
            "random_state": RANDOM_STATE,
            "vectorizer": "tfidf(ngram_range=(1,2), min_df=2, sublinear_tf=True, lowercase=True)",
            "classifier": "logistic_regression(class_weight=balanced, max_iter=1000)",
        },
        "split_evaluated": args.split,
        "train_rows": len(train_frame),
        "eval_rows": len(eval_frame),
        "dataset_source_sha256": calculate_sha256(args.dataset),
        "manifest_sha256": calculate_sha256(args.manifest),
        "expected_source_sha256": EXPECTED_SOURCE_SHA256,
        "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "metrics": metrics.to_dict(),
    }

    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_path = args.experiment_dir / f"{record['experiment_id']}.json"
    experiment_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report_path = args.report or Path(f"outputs/tables/baseline_{args.split}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.split == "test":
        TEST_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TEST_LEDGER_PATH.open("a", encoding="utf-8") as ledger:
            ledger.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"AVISO: execução no split de teste registrada em {TEST_LEDGER_PATH}")

    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"Registro bruto salvo em {experiment_path}")
    print(f"Relatório agregado salvo em {report_path}")


if __name__ == "__main__":
    main()
