"""Audita o HateBRXplain sem exibir ou copiar comentários."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_COLUMNS = [
    "id",
    "comment",
    "offensive_label",
    "link_post",
    "rationales_annotator1",
    "rationales_annotator2",
]
EXPECTED_ROWS = 7_000
EXPECTED_CLASS_DISTRIBUTION = {"0": 3_500, "1": 3_500}
EXPECTED_SHA256 = "dd2b669ab8c055805b4813b2628961d951273577fc637039f5eec1238cc9f8bb"
SOURCE_REVISION = "0ac05461f17c4d7b7655fbee0391ec325e61a83b"
SOURCE_URL = (
    "https://raw.githubusercontent.com/franciellevargas/HateBR/"
    f"{SOURCE_REVISION}/dataset/HateBRXplain.csv"
)


def load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Formato não suportado: {suffix}. Use CSV, JSON ou JSONL.")


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_for_comparison(value: object) -> str:
    """Normaliza somente para detectar duplicatas; não altera o corpus."""
    if pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(normalized.split())


def _label_key(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _value_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False)
    return {
        _label_key(key): int(value)
        for key, value in sorted(counts.items(), key=lambda item: str(item[0]))
    }


def _empty_string_count(series: pd.Series) -> int:
    present = series.dropna().astype("string")
    return int(present.str.strip().eq("").sum())


def _rationale_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for column in ("rationales_annotator1", "rationales_annotator2"):
        counts: dict[str, int] = {}
        for label, group in frame.groupby("offensive_label", dropna=False):
            counts[_label_key(label)] = int(group[column].notna().sum())
        result[column] = counts
    return result


def build_report(
    frame: pd.DataFrame,
    source: Path,
    *,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    nulls = frame.isna().sum()
    columns = [str(column) for column in frame.columns]
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in frame]
    extra_columns = [column for column in columns if column not in EXPECTED_COLUMNS]

    statistics: dict[str, Any] = {
        "exact_duplicate_rows": int(frame.duplicated().sum()),
    }

    checks: dict[str, bool] = {
        "source_sha256_matches": source_sha256 == EXPECTED_SHA256,
        "row_count_matches": len(frame) == EXPECTED_ROWS,
        "columns_match": columns == EXPECTED_COLUMNS,
    }

    if not missing_columns:
        labels = frame["offensive_label"]
        class_distribution = _value_counts(labels)
        normalized_comments = frame["comment"].map(normalize_for_comparison)
        normalized_label_counts = (
            pd.DataFrame({"normalized_comment": normalized_comments, "label": labels})
            .groupby("normalized_comment", dropna=False)["label"]
            .nunique(dropna=False)
        )

        rationale_counts = _rationale_counts(frame)
        post_sets = {
            _label_key(label): set(group["link_post"].dropna())
            for label, group in frame.groupby("offensive_label", dropna=False)
        }
        shared_posts = post_sets.get("0", set()) & post_sets.get("1", set())

        statistics.update(
            {
                "class_distribution": class_distribution,
                "unique_ids": int(frame["id"].nunique(dropna=True)),
                "exact_duplicate_comments": int(frame["comment"].duplicated().sum()),
                "normalized_duplicate_comments": int(normalized_comments.duplicated().sum()),
                "conflicting_normalized_label_groups": int(normalized_label_counts.gt(1).sum()),
                "unique_link_posts": int(frame["link_post"].nunique(dropna=True)),
                "unique_link_posts_by_class": {
                    label: len(posts) for label, posts in sorted(post_sets.items())
                },
                "link_posts_shared_between_classes": len(shared_posts),
                "rationales_non_null_by_class": rationale_counts,
            }
        )

        checks.update(
            {
                "ids_are_complete_and_unique": bool(
                    frame["id"].notna().all() and frame["id"].is_unique
                ),
                "comments_are_present": bool(
                    frame["comment"].notna().all() and _empty_string_count(frame["comment"]) == 0
                ),
                "labels_are_binary_and_balanced": (
                    class_distribution == EXPECTED_CLASS_DISTRIBUTION
                ),
                "link_posts_are_present": bool(
                    frame["link_post"].notna().all()
                    and _empty_string_count(frame["link_post"]) == 0
                ),
                "comments_have_no_exact_duplicates": bool(not frame["comment"].duplicated().any()),
                "comments_have_no_normalized_duplicates": bool(
                    not normalized_comments.duplicated().any()
                ),
                "normalized_duplicates_have_no_label_conflicts": bool(
                    not normalized_label_counts.gt(1).any()
                ),
                "rationales_follow_offensive_label": all(
                    counts.get("0", 0) == 0 and counts.get("1", 0) == 3_500
                    for counts in rationale_counts.values()
                ),
            }
        )

    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "audit_version": 1,
        "provenance": {
            "source_filename": source.name,
            "file_size_bytes": source.stat().st_size if source.is_file() else None,
            "sha256": source_sha256,
            "expected_sha256": EXPECTED_SHA256,
            "source_url": SOURCE_URL,
            "source_revision": SOURCE_REVISION,
            "license": "CC BY-NC 4.0",
        },
        "dataset": {
            "rows": len(frame),
            "columns": columns,
            "expected_columns": EXPECTED_COLUMNS,
            "missing_columns": missing_columns,
            "extra_columns": extra_columns,
            "column_types": {str(key): str(value) for key, value in frame.dtypes.items()},
            "null_count": {str(key): int(value) for key, value in nulls.items()},
            "empty_string_count": {
                str(column): _empty_string_count(frame[column]) for column in frame.columns
            },
        },
        "statistics": statistics,
        "checks": checks,
        "valid": not failed_checks,
        "failed_checks": failed_checks,
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

    source_sha256 = calculate_sha256(args.dataset)
    report = build_report(load_dataset(args.dataset), args.dataset, source_sha256=source_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Relatório salvo em {args.output}")
    if not report["valid"]:
        raise SystemExit("Auditoria reprovada; consulte failed_checks no relatório.")


if __name__ == "__main__":
    main()
