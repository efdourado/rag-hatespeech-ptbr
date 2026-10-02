"""Execução pareada retomável: baseline LLM, RAG e ablação sem rationales."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from rag_hatespeech_ptbr.llm import LLMClassifier, prediction_record
from rag_hatespeech_ptbr.metrics import evaluate_predictions
from rag_hatespeech_ptbr.openrouter import OpenRouterError
from rag_hatespeech_ptbr.retrieval import Embedder, LocalIndex
from rag_hatespeech_ptbr.splits import calculate_sha256

MODES = ("baseline", "rag", "rag_no_rationales")


def git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def dependency_versions() -> dict[str, str | None]:
    result = {}
    for name in ("numpy", "pandas", "scikit-learn", "httpx", "sentence-transformers", "torch"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_sarcasm_labels(paths: list[Path]) -> tuple[dict[int, int], dict[str, str]]:
    # Usadas somente na análise posterior, nunca no prompt/índice/seleção dos itens.
    from rag_hatespeech_ptbr.sarcasm_pilot import load_annotation_workbook

    labels: dict[int, int] = {}
    hashes = {}
    for path in paths:
        table = load_annotation_workbook(path)
        numeric = pd.to_numeric(table.human_sarcasm_label, errors="coerce")
        if table.id.duplicated().any() or not numeric.isin([0, 1, 2]).all():
            raise ValueError("Anotação contém IDs repetidos ou rótulos inválidos/ausentes.")
        for identifier, label in zip(table.id, numeric):
            identifier = int(identifier)
            if identifier in labels:
                raise ValueError("ID repetido entre planilhas de anotação.")
            labels[identifier] = int(label)
        hashes[str(path)] = calculate_sha256(path)
    return labels, hashes


def run_comparison(
    frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    classifier: LLMClassifier,
    *,
    modes: list[str],
    run_dir: Path,
    provenance: dict[str, str],
    top_k: int = 5,
    embedder: Embedder | None = None,
    index: LocalIndex | None = None,
    index_sha256: str | None = None,
    sarcasm_labels: dict[int, int] | None = None,
    annotation_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not modes or len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError("Modos inválidos ou repetidos.")
    if eval_frame.empty or eval_frame.id.duplicated().any():
        raise ValueError("Avaliação vazia ou com IDs repetidos.")
    if eval_frame.split.nunique() != 1 or eval_frame.split.iloc[0] not in ("validation", "test"):
        raise ValueError("Avaliação deve usar somente validação ou teste.")
    use_rag = any(m != "baseline" for m in modes)
    if use_rag:
        if index is None or embedder is None:
            raise ValueError("RAG exige índice e embedder.")
        index.validate(frame, embedder, provenance)
        if not 1 <= top_k <= len(index.ids):
            raise ValueError("top_k fora do tamanho do treino.")
    source_files = [
        Path(__file__),
        Path(__file__).with_name("llm.py"),
        Path(__file__).with_name("retrieval.py"),
        Path(__file__).with_name("openrouter.py"),
        Path(__file__).with_name("runtime.py"),
        Path(__file__).with_name("metrics.py"),
    ]
    config = {
        **provenance,
        "classifier": classifier.configuration(),
        "modes": modes,
        "top_k": top_k if use_rag else None,
        "index_sha256": index_sha256 if use_rag else None,
        "embedding": index.metadata if use_rag and index is not None else None,
        "split_evaluated": str(eval_frame.split.iloc[0]),
        "eval_ids": eval_frame.id.astype(int).tolist(),
        "annotation_hashes": annotation_hashes or {},
        "source_hashes": {p.name: calculate_sha256(p) for p in source_files},
        "dependency_versions": dependency_versions(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path, results_path = run_dir / "config.json", run_dir / "predictions.jsonl"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Configuração mudou; use um novo --run-dir para outra execução.")
    else:
        if any(run_dir.iterdir()):
            raise ValueError("Diretório de execução não vazio e sem configuração.")
        write_json(config_path, config)
        write_json(
            run_dir / "environment.json",
            {
                "started_at_utc": datetime.now(UTC).isoformat(),
                "git": git_state(),
            },
        )
    records: dict[tuple[int, str], dict[str, Any]] = {}
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            record = json.loads(line)
            key = (record["id"], record["mode"])
            if key in records or key[0] not in config["eval_ids"] or key[1] not in modes:
                raise ValueError("Arquivo de predições contém registros repetidos/inesperados.")
            records[key] = record
    if config["split_evaluated"] == "test" and not classifier.dry_run and not records:
        ledger = Path("outputs/experiments/test_evaluations_ledger.jsonl")
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "started_at_utc": datetime.now(UTC).isoformat(),
                        "run_dir": str(run_dir),
                        "config": config,
                    }
                )
                + "\n"
            )
    previous_usage = (
        json.loads((run_dir / "api_calls.json").read_text())
        if (run_dir / "api_calls.json").exists()
        else []
    )
    initial_call_count = len(classifier.client.calls) if classifier.client else 0
    error: Exception | None = None
    try:
        for row in eval_frame.itertuples(index=False):
            pending = [mode for mode in modes if (int(row.id), mode) not in records]
            if not pending:
                continue
            examples = []
            if any(mode != "baseline" for mode in pending):
                assert index is not None and embedder is not None
                examples = index.search(embedder.embed([row.comment]), frame, top_k)
            for mode in pending:
                context = examples if mode != "baseline" else []
                prediction = classifier.predict(
                    row.comment, context, include_rationales=mode != "rag_no_rationales"
                )
                record = {
                    "id": int(row.id),
                    "mode": mode,
                    "y_true": int(row.offensive_label),
                    **prediction_record(prediction),
                    "retrieved": [{"id": e.id, "score": e.score} for e in context],
                }
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records[(int(row.id), mode)] = record
    except (ValueError, RuntimeError, OSError, OpenRouterError) as exc:
        error = exc
    finally:
        calls = previous_usage + (
            classifier.client.calls[initial_call_count:] if classifier.client else []
        )
        write_json(run_dir / "api_calls.json", calls)
    complete = len(records) == len(eval_frame) * len(modes)
    summary: dict[str, Any] = {
        "status": "complete" if complete else "partial",
        "dry_run": classifier.dry_run,
        "scientific_result": complete and not classifier.dry_run,
        "eval_rows": len(eval_frame),
        "completed_predictions": len(records),
        "expected_predictions": len(eval_frame) * len(modes),
        "metrics": None,
        "reported_cost_usd": sum(
            c["usage"]["cost"]
            for c in calls
            if isinstance(c.get("usage", {}).get("cost"), (int, float))
        )
        if any(isinstance(c.get("usage", {}).get("cost"), (int, float)) for c in calls)
        else None,
        "calls_without_reported_cost": sum(
            not isinstance(c.get("usage", {}).get("cost"), (int, float)) for c in calls
        ),
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }
    if complete and not classifier.dry_run:
        summary["metrics"] = {}
        ids = eval_frame.id.astype(int).tolist()
        truth = eval_frame.offensive_label.astype(int).tolist()
        for mode in modes:
            predicted = [records[(identifier, mode)]["offensive_label"] for identifier in ids]
            result = {"overall": evaluate_predictions(truth, predicted).to_dict()}
            if sarcasm_labels:
                result["sarcasm_strata"] = {}
                for label in (0, 1, 2):
                    mask = [sarcasm_labels.get(identifier) == label for identifier in ids]
                    result["sarcasm_strata"][str(label)] = (
                        evaluate_predictions(truth, predicted, mask=mask).to_dict()
                        if any(mask)
                        else None
                    )
                result["sarcasm_missing"] = sum(i not in sarcasm_labels for i in ids)
            summary["metrics"][mode] = result
    write_json(run_dir / "summary.json", summary)
    if error is not None:
        raise error
    return summary
