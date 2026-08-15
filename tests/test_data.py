from pathlib import Path

import pandas as pd
import pytest

from rag_hatespeech_ptbr.data import DatasetIntegrityError, get_split, load_labeled_frame


def _write_synthetic_dataset(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "comment": ["a", "b", "c", "d"],
            "offensive_label": [0, 1, 0, 1],
            "link_post": ["https://example.test/1"] * 4,
            "rationales_annotator1": [None, "r1", None, "r1"],
            "rationales_annotator2": [None, "r2", None, "r2"],
        }
    )
    frame.to_csv(path, index=False)


def _write_synthetic_manifest(path: Path) -> None:
    manifest = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "fold": [0, 7, 1, 1],
            "split": ["test", "validation", "train", "train"],
        }
    )
    manifest.to_csv(path, index=False)


def test_load_labeled_frame_joins_dataset_and_manifest(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    manifest_path = tmp_path / "manifest.csv"
    _write_synthetic_dataset(dataset_path)
    _write_synthetic_manifest(manifest_path)

    frame = load_labeled_frame(dataset_path, manifest_path, verify_hashes=False)

    assert len(frame) == 4
    assert set(frame["split"]) == {"test", "validation", "train"}


def test_load_labeled_frame_rejects_mismatched_hash_by_default(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    manifest_path = tmp_path / "manifest.csv"
    _write_synthetic_dataset(dataset_path)
    _write_synthetic_manifest(manifest_path)

    with pytest.raises(DatasetIntegrityError, match="dataset bruto"):
        load_labeled_frame(dataset_path, manifest_path)


def test_load_labeled_frame_raises_when_manifest_missing_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    manifest_path = tmp_path / "manifest.csv"
    _write_synthetic_dataset(dataset_path)
    manifest = pd.DataFrame({"id": [1, 2], "fold": [0, 7], "split": ["test", "validation"]})
    manifest.to_csv(manifest_path, index=False)

    with pytest.raises(DatasetIntegrityError, match="manifesto"):
        load_labeled_frame(dataset_path, manifest_path, verify_hashes=False)


def test_get_split_filters_rows() -> None:
    frame = pd.DataFrame({"id": [1, 2, 3], "split": ["train", "validation", "test"]})

    train = get_split(frame, "train")

    assert train["id"].to_list() == [1]


def test_get_split_rejects_unknown_split() -> None:
    frame = pd.DataFrame({"id": [1], "split": ["train"]})

    with pytest.raises(ValueError, match="Split desconhecido"):
        get_split(frame, "bogus")
