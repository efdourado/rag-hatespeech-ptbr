"""Inspeciona CSV/JSON/JSONL sem exibir ou copiar comentários."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Formato não suportado: {suffix}. Use CSV, JSON ou JSONL.")


def build_report(frame: pd.DataFrame, source: Path) -> dict[str, object]:
    nulls = frame.isna().sum()
    return {
        "source_filename": source.name,
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "column_types": {str(k): str(v) for k, v in frame.dtypes.items()},
        "null_count": {str(k): int(v) for k, v in nulls.items()},
        "exact_duplicate_rows": int(frame.duplicated().sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="arquivo local em CSV, JSON ou JSONL")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tables/dataset_audit.json"),
        help="destino do relatório agregado",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset não encontrado: {args.dataset}")

    report = build_report(load_dataset(args.dataset), args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Relatório salvo em {args.output}")


if __name__ == "__main__":
    main()
