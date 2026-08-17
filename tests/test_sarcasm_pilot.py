import pandas as pd
import pytest

from rag_hatespeech_ptbr.sarcasm_pilot import (
    compute_agreement,
    compute_multi_rater_agreement,
    fleiss_kappa,
    load_labeled_pool,
    select_additional_sample,
    select_pilot_sample,
    select_reliability_check_subset,
    select_second_annotator_subset,
    write_annotation_workbook,
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


def _labeled_pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": list(range(1, 11)),
            "comment": [f"comentario sintetico {i}" for i in range(1, 11)],
            "human_sarcasm_label": [0, 0, 1, 2, 0, 0, 1, 0, 2, 0],
            "sarcasm_evidence": [""] * 10,
            "annotation_notes": [""] * 10,
            "review_status": ["reviewed"] * 10,
        }
    )


def test_select_reliability_check_subset_is_blind_and_deterministic() -> None:
    pool = _labeled_pool()

    first = select_reliability_check_subset(pool, n=4, random_state=7)
    second = select_reliability_check_subset(pool, n=4, random_state=7)

    pd.testing.assert_frame_equal(first, second)
    assert (first["human_sarcasm_label"] == "").all()
    assert set(first["id"]).issubset(set(pool["id"]))


def test_select_reliability_check_subset_excludes_already_checked_ids() -> None:
    pool = _labeled_pool()
    exclude_ids = {1, 2, 3}

    subset = select_reliability_check_subset(pool, n=4, random_state=7, exclude_ids=exclude_ids)

    assert set(subset["id"]).isdisjoint(exclude_ids)


def test_select_reliability_check_subset_raises_if_pool_too_small() -> None:
    pool = _labeled_pool()

    with pytest.raises(ValueError):
        select_reliability_check_subset(pool, n=100, random_state=7)


def test_fleiss_kappa_is_one_for_perfect_agreement() -> None:
    labels = pd.DataFrame({"rater_a": [0, 1, 2], "rater_b": [0, 1, 2], "rater_c": [0, 1, 2]})

    assert fleiss_kappa(labels) == pytest.approx(1.0, abs=1e-9)


def test_fleiss_kappa_matches_hand_computed_example() -> None:
    labels = pd.DataFrame(
        {
            "rater_a": [0, 1, 2],
            "rater_b": [0, 1, 2],
            "rater_c": [1, 1, 0],
        }
    )

    assert fleiss_kappa(labels) == pytest.approx(0.3078, abs=1e-3)


def test_fleiss_kappa_requires_at_least_two_raters() -> None:
    labels = pd.DataFrame({"rater_a": [0, 1, 2]})

    with pytest.raises(ValueError):
        fleiss_kappa(labels)


def test_compute_multi_rater_agreement_includes_fleiss_with_two_checkers() -> None:
    primary = pd.DataFrame({"id": [1, 2, 3], "human_sarcasm_label": [0, 1, 2]})
    checker_a = pd.DataFrame({"id": [1, 2, 3], "human_sarcasm_label": [0, 1, 2]})
    checker_b = pd.DataFrame({"id": [1, 2, 3], "human_sarcasm_label": [0, 1, 0]})

    result = compute_multi_rater_agreement(primary, {"amigo_a": checker_a, "amigo_b": checker_b})

    assert set(result["pairwise"].keys()) == {"amigo_a", "amigo_b"}
    assert result["pairwise"]["amigo_a"]["exact_agreement_rate"] == pytest.approx(1.0)
    assert "fleiss_kappa" in result
    assert result["fleiss_kappa_n_items"] == 3


def test_load_labeled_pool_survives_concat_with_still_blank_sheet(tmp_path) -> None:
    """Regressão: escrever um lote em branco (via write_annotation_workbook,
    como as planilhas reais) e concatenar com um lote já rotulado não pode
    fazer os rótulos válidos desaparecerem por causa da promoção int->float
    que o pandas faz ao juntar uma coluna inteira com uma coluna de NaN.
    """
    labeled = select_pilot_sample(_synthetic_frame(), n_per_class=5, random_state=1)
    labeled["human_sarcasm_label"] = [0, 1, 2, 0, 1, 2, 0, 1, 0, 1]
    still_blank = select_pilot_sample(_synthetic_frame(), n_per_class=5, random_state=2)

    labeled_path = tmp_path / "labeled.xlsx"
    blank_path = tmp_path / "blank.xlsx"
    write_annotation_workbook(labeled, labeled_path)
    write_annotation_workbook(still_blank, blank_path)

    pool = load_labeled_pool([labeled_path, blank_path])

    assert len(pool) == 10
    assert set(pool["human_sarcasm_label"].astype(int)) == {0, 1, 2}


def test_compute_multi_rater_agreement_skips_fleiss_with_single_checker() -> None:
    primary = pd.DataFrame({"id": [1, 2], "human_sarcasm_label": [0, 1]})
    checker_a = pd.DataFrame({"id": [1, 2], "human_sarcasm_label": [0, 1]})

    result = compute_multi_rater_agreement(primary, {"amigo_a": checker_a})

    assert "fleiss_kappa" not in result
