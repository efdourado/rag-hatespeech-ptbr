from pathlib import Path

import pandas as pd

from scripts.inspect_dataset import EXPECTED_SHA256, build_report, normalize_for_comparison


def test_build_report_returns_only_aggregate_metadata() -> None:
    frame = pd.DataFrame(
        {
            "id": [1, 2],
            "comment": ["texto sensível", "TEXTO   SENSÍVEL"],
            "offensive_label": [0, 1],
            "link_post": ["https://example.test/1", "https://example.test/2"],
            "rationales_annotator1": [None, "evidência privada"],
            "rationales_annotator2": [None, "outra evidência"],
        }
    )

    report = build_report(frame, Path("dataset.csv"))

    assert report["dataset"]["rows"] == 2
    assert report["statistics"]["exact_duplicate_comments"] == 0
    assert report["statistics"]["normalized_duplicate_comments"] == 1
    assert report["statistics"]["conflicting_normalized_label_groups"] == 1
    assert "texto sensível" not in str(report)
    assert "evidência privada" not in str(report)
    assert "https://example.test/1" not in str(report)


def test_build_report_accepts_expected_hatebrxplain_contract() -> None:
    total = 7_000
    labels = [0] * 3_500 + [1] * 3_500
    frame = pd.DataFrame(
        {
            "id": range(1, total + 1),
            "comment": [f"comentário sintético {index}" for index in range(total)],
            "offensive_label": labels,
            "link_post": [f"https://example.test/{index % 85}" for index in range(total)],
            "rationales_annotator1": [None] * 3_500 + ["evidência"] * 3_500,
            "rationales_annotator2": [None] * 3_500 + ["justificativa"] * 3_500,
        }
    )

    report = build_report(
        frame,
        Path("HateBRXplain.csv"),
        source_sha256=EXPECTED_SHA256,
    )

    assert report["valid"] is True
    assert report["failed_checks"] == []
    assert report["statistics"]["class_distribution"] == {"0": 3_500, "1": 3_500}


def test_normalize_for_comparison_preserves_only_comparison_equivalence() -> None:
    assert normalize_for_comparison("  Você  É\tÓTIMO ") == "você é ótimo"
