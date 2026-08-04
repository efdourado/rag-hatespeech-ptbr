"""Divisão reprodutível do HateBRXplain com isolamento por publicação."""

from __future__ import annotations

import hashlib
import platform
import unicodedata
from itertools import combinations
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

N_SPLITS = 10
RANDOM_STATE = 42
TEST_FOLD = 0
VALIDATION_FOLD = 7

ID_COLUMN = "id"
LABEL_COLUMN = "offensive_label"
GROUP_COLUMN = "link_post"
COMMENT_COLUMN = "comment"
FOLD_COLUMN = "fold"
SPLIT_COLUMN = "split"
SPLIT_NAMES = ("train", "validation", "test")

EXPECTED_SOURCE_SHA256 = "dd2b669ab8c055805b4813b2628961d951273577fc637039f5eec1238cc9f8bb"
EXPECTED_MANIFEST_SHA256 = "6a7212fbb42c26fc8e9c8f1ef3fe391fb9f651d5c14820fc3e2b521002208033"
EXPECTED_SPLIT_SIZES = {"train": 5_608, "validation": 706, "test": 686}
EXPECTED_CLASS_DISTRIBUTION = {
    "train": {"0": 2_804, "1": 2_804},
    "validation": {"0": 353, "1": 353},
    "test": {"0": 343, "1": 343},
}


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_input(frame: pd.DataFrame) -> None:
    required = {ID_COLUMN, LABEL_COLUMN, GROUP_COLUMN, COMMENT_COLUMN}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {', '.join(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("ID, rótulo e grupo não podem conter valores ausentes.")
    if not frame[ID_COLUMN].is_unique:
        raise ValueError("Os IDs precisam ser únicos antes da divisão.")
    if set(frame[LABEL_COLUMN].unique()) != {0, 1}:
        raise ValueError("offensive_label deve conter exatamente os valores 0 e 1.")


def canonicalize_group(value: object) -> str:
    """Remove variações de URL que não identificam uma publicação diferente."""
    parsed = urlsplit(str(value).strip())
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))


def normalize_for_comparison(value: object) -> str:
    """Normaliza texto somente para verificar vazamento entre partições."""
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(normalized.split())


def create_grouped_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Atribui cada ID a uma partição sem separar itens do mesmo link_post."""
    _validate_input(frame)
    working = frame.sort_values(ID_COLUMN, kind="stable").reset_index(drop=True).copy()
    canonical_groups = working[GROUP_COLUMN].map(canonicalize_group)

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    fold_by_index = pd.Series(-1, index=working.index, dtype="int64")
    for fold, (_, held_out_indices) in enumerate(
        splitter.split(working, working[LABEL_COLUMN], groups=canonical_groups)
    ):
        fold_by_index.iloc[held_out_indices] = fold

    if fold_by_index.lt(0).any():
        raise RuntimeError("Nem todos os registros receberam um fold.")

    split_by_fold = {TEST_FOLD: "test", VALIDATION_FOLD: "validation"}
    manifest = pd.DataFrame(
        {
            ID_COLUMN: working[ID_COLUMN],
            FOLD_COLUMN: fold_by_index,
            SPLIT_COLUMN: fold_by_index.map(lambda fold: split_by_fold.get(fold, "train")),
        }
    )
    return manifest.sort_values(ID_COLUMN, kind="stable").reset_index(drop=True)


def write_manifest(manifest: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False, lineterminator="\n")
    return calculate_sha256(path)


def _label_key(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _class_distribution(joined: pd.DataFrame) -> dict[str, dict[str, int]]:
    distribution: dict[str, dict[str, int]] = {}
    for split_name in SPLIT_NAMES:
        counts = joined.loc[joined[SPLIT_COLUMN].eq(split_name), LABEL_COLUMN].value_counts()
        distribution[split_name] = {
            _label_key(label): int(count)
            for label, count in sorted(counts.items(), key=lambda item: str(item[0]))
        }
    return distribution


def _pairwise_overlap(joined: pd.DataFrame, column: str) -> dict[str, int]:
    values = {
        split_name: set(joined.loc[joined[SPLIT_COLUMN].eq(split_name), column])
        for split_name in SPLIT_NAMES
    }
    return {
        f"{left}__{right}": len(values[left] & values[right])
        for left, right in combinations(SPLIT_NAMES, 2)
    }


def _group_concentration(joined: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for split_name in SPLIT_NAMES:
        split_rows = joined.loc[joined[SPLIT_COLUMN].eq(split_name)]
        group_sizes = split_rows.groupby("canonical_group").size().sort_values(ascending=False)
        result[split_name] = {
            "largest_group_rows": int(group_sizes.iloc[0]),
            "largest_group_fraction": round(float(group_sizes.iloc[0] / len(split_rows)), 6),
            "top_3_groups_fraction": round(float(group_sizes.head(3).sum() / len(split_rows)), 6),
        }
    return result


def build_split_report(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    source_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Produz apenas estatísticas agregadas; não inclui IDs, links ou textos."""
    _validate_input(frame)

    if list(manifest.columns) != [ID_COLUMN, FOLD_COLUMN, SPLIT_COLUMN]:
        raise ValueError("O manifesto deve conter apenas as colunas id, fold e split.")

    joined = frame[[ID_COLUMN, LABEL_COLUMN, GROUP_COLUMN, COMMENT_COLUMN]].merge(
        manifest,
        on=ID_COLUMN,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    joined["canonical_group"] = joined[GROUP_COLUMN].map(canonicalize_group)
    joined["normalized_comment"] = joined[COMMENT_COLUMN].map(normalize_for_comparison)
    split_sizes = {
        split_name: int(manifest[SPLIT_COLUMN].eq(split_name).sum())
        for split_name in SPLIT_NAMES
    }
    class_distribution = _class_distribution(joined)
    unique_groups = {
        split_name: int(
            joined.loc[joined[SPLIT_COLUMN].eq(split_name), "canonical_group"].nunique()
        )
        for split_name in SPLIT_NAMES
    }
    id_overlap = _pairwise_overlap(joined, ID_COLUMN)
    group_overlap = _pairwise_overlap(joined, "canonical_group")
    normalized_comment_overlap = _pairwise_overlap(joined, "normalized_comment")
    group_concentration = _group_concentration(joined)
    total_rows = len(frame)
    proportions = {
        split_name: round(size / total_rows, 6) for split_name, size in split_sizes.items()
    }

    checks = {
        "source_sha256_matches": source_sha256 == EXPECTED_SOURCE_SHA256,
        "manifest_sha256_matches": manifest_sha256 == EXPECTED_MANIFEST_SHA256,
        "manifest_ids_are_unique": bool(manifest[ID_COLUMN].is_unique),
        "manifest_has_only_known_splits": bool(
            set(manifest[SPLIT_COLUMN].unique()) == set(SPLIT_NAMES)
        ),
        "all_source_ids_are_covered_once": bool(
            len(joined) == len(frame)
            and joined["_merge"].eq("both").all()
            and manifest[ID_COLUMN].is_unique
        ),
        "ids_do_not_cross_splits": all(overlap == 0 for overlap in id_overlap.values()),
        "groups_do_not_cross_splits": all(overlap == 0 for overlap in group_overlap.values()),
        "normalized_comments_do_not_cross_splits": all(
            overlap == 0 for overlap in normalized_comment_overlap.values()
        ),
        "fold_mapping_is_consistent": bool(
            manifest.loc[manifest[SPLIT_COLUMN].eq("test"), FOLD_COLUMN].eq(TEST_FOLD).all()
            and manifest.loc[
                manifest[SPLIT_COLUMN].eq("validation"), FOLD_COLUMN
            ].eq(VALIDATION_FOLD).all()
            and manifest.loc[manifest[SPLIT_COLUMN].eq("train"), FOLD_COLUMN]
            .isin(set(range(N_SPLITS)) - {TEST_FOLD, VALIDATION_FOLD})
            .all()
        ),
        "split_sizes_match_frozen_design": split_sizes == EXPECTED_SPLIT_SIZES,
        "classes_match_frozen_design": class_distribution == EXPECTED_CLASS_DISTRIBUTION,
        "every_split_is_class_balanced": all(
            counts.get("0") == counts.get("1") for counts in class_distribution.values()
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]

    return {
        "split_audit_version": 1,
        "method": {
            "name": "StratifiedGroupKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "group_column": GROUP_COLUMN,
            "label_column": LABEL_COLUMN,
            "test_fold": TEST_FOLD,
            "validation_fold": VALIDATION_FOLD,
            "remaining_folds": "train",
            "rag_index_allowed_splits": ["train"],
        },
        "reproducibility": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "source_sha256": source_sha256,
            "manifest_sha256": manifest_sha256,
            "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        },
        "statistics": {
            "total_rows": total_rows,
            "total_unique_groups_raw": int(frame[GROUP_COLUMN].nunique()),
            "total_unique_groups_canonical": int(joined["canonical_group"].nunique()),
            "split_sizes": split_sizes,
            "split_proportions": proportions,
            "class_distribution": class_distribution,
            "unique_groups_by_split": unique_groups,
            "pairwise_id_overlap": id_overlap,
            "pairwise_group_overlap": group_overlap,
            "pairwise_normalized_comment_overlap": normalized_comment_overlap,
            "group_concentration": group_concentration,
        },
        "checks": checks,
        "valid": not failed_checks,
        "failed_checks": failed_checks,
    }
