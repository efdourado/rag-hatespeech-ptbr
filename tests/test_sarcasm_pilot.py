import pandas as pd
import pytest

from rag_hatespeech_ptbr.sarcasm_pilot import (
    compute_agreement,
    select_additional_sample,
    select_pilot_sample,
    select_second_annotator_subset,
)


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


def test_select_additional_sample_excludes_already_used_ids_and_never_leaks() -> None:
    frame = _synthetic_frame()
    pilot = select_pilot_sample(frame, n_per_class=10, random_state=1)
    exclude_ids = set(pilot["id"])
    validation_ids = set(frame.loc[frame["split"] == "validation", "id"])

    additional = select_additional_sample(frame, exclude_ids=exclude_ids, random_state=2)

    assert set(additional["id"]).isdisjoint(exclude_ids)
    assert set(additional["id"]).issubset(validation_ids)
    assert len(additional) == len(validation_ids) - len(exclude_ids)


def test_select_additional_sample_respects_n_per_class_cap() -> None:
    frame = _synthetic_frame()

    additional = select_additional_sample(
        frame, exclude_ids=set(), n_per_class=5, random_state=2
    )

    assert len(additional) == 10


def test_compute_agreement_matches_manual_calculation() -> None:
    primary = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "comment": ["texto sensivel 1", "texto sensivel 2", "texto sensivel 3", "texto sensivel 4"],
            "human_sarcasm_label": [0, 0, 1, 2],
        }
    )
    second = pd.DataFrame(
        {
            "id": [1, 2, 3, 5],
            "comment": ["texto sensivel 1", "texto sensivel 2", "texto sensivel 3", "texto sensivel 5"],
            "human_sarcasm_label": [0, 1, 1, 0],
        }
    )

    result = compute_agreement(primary, second)

    assert result["n_common"] == 3
    assert result["exact_agreement"] == 2
    assert result["exact_agreement_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert result["disagreement_ids"] == [2]
    assert "texto sensivel" not in str(result)


def test_compute_agreement_raises_without_overlap() -> None:
    primary = pd.DataFrame({"id": [1], "human_sarcasm_label": [0]})
    second = pd.DataFrame({"id": [2], "human_sarcasm_label": [0]})

    with pytest.raises(ValueError):
        compute_agreement(primary, second)
