import pandas as pd
import pytest

from rag_hatespeech_ptbr.sarcasm_pilot import select_pilot_sample, select_second_annotator_subset


def _synthetic_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    row_id = 1
    for split, count in (("train", 40), ("validation", 120), ("test", 40)):
        for index in range(count):
            rows.append(
                {
                    "id": row_id,
                    "comment": f"comentario sintetico {split} {index}",
                    "offensive_label": index % 2,
                    "split": split,
                }
            )
            row_id += 1
    return pd.DataFrame(rows)


def test_select_pilot_sample_never_uses_train_or_test() -> None:
    frame = _synthetic_frame()
    validation_ids = set(frame.loc[frame["split"] == "validation", "id"])

    pilot = select_pilot_sample(frame, n_per_class=10, random_state=1)

    assert set(pilot["id"]).issubset(validation_ids)


def test_select_pilot_sample_is_balanced_by_offensive_label() -> None:
    frame = _synthetic_frame()

    pilot = select_pilot_sample(frame, n_per_class=15, random_state=1)

    assert len(pilot) == 30
    merged = pilot.merge(frame[["id", "offensive_label"]], on="id")
    assert merged["offensive_label"].value_counts().to_dict() == {0: 15, 1: 15}


def test_select_pilot_sample_hides_offensive_label_and_is_deterministic() -> None:
    frame = _synthetic_frame()

    first = select_pilot_sample(frame, n_per_class=10, random_state=1)
    second = select_pilot_sample(frame, n_per_class=10, random_state=1)

    pd.testing.assert_frame_equal(first, second)
    assert "offensive_label" not in first.columns
    assert list(first.columns) == [
        "id",
        "comment",
        "human_sarcasm_label",
        "sarcasm_evidence",
        "annotation_notes",
        "review_status",
    ]
    assert (first["review_status"] == "pending").all()


def test_select_pilot_sample_has_no_duplicate_ids() -> None:
    frame = _synthetic_frame()

    pilot = select_pilot_sample(frame, n_per_class=20, random_state=2)

    assert pilot["id"].is_unique


def test_select_second_annotator_subset_is_subset_of_pilot() -> None:
    frame = _synthetic_frame()
    pilot = select_pilot_sample(frame, n_per_class=10, random_state=1)

    subset = select_second_annotator_subset(pilot, fraction=0.2, random_state=5)

    assert set(subset["id"]).issubset(set(pilot["id"]))
    assert len(subset) == 4


def test_select_pilot_sample_requires_enough_items_per_class() -> None:
    frame = _synthetic_frame()

    with pytest.raises(ValueError):
        select_pilot_sample(frame, n_per_class=1000, random_state=1)
