import pandas as pd
import pytest

from rag_hatespeech_ptbr.splits import (
    build_split_report,
    canonicalize_group,
    create_grouped_manifest,
)


def build_synthetic_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    row_id = 1
    for group in range(20):
        for item in range(10):
            rows.append(
                {
                    "id": row_id,
                    "comment": f"conteúdo privado {row_id}",
                    "offensive_label": item % 2,
                    "link_post": f"https://example.test/post/{group}",
                }
            )
            row_id += 1
    return pd.DataFrame(rows)


def test_grouped_manifest_is_deterministic_and_keeps_groups_together() -> None:
    frame = build_synthetic_frame()

    first = create_grouped_manifest(frame)
    second = create_grouped_manifest(frame.sample(frac=1, random_state=99))

    pd.testing.assert_frame_equal(first, second)
    joined = frame.merge(first, on="id", validate="one_to_one")
    assert joined.groupby("link_post")["split"].nunique().max() == 1
    assert set(first["split"]) == {"train", "validation", "test"}


def test_split_report_contains_only_aggregate_information() -> None:
    frame = build_synthetic_frame()
    manifest = create_grouped_manifest(frame)

    report = build_split_report(
        frame,
        manifest,
        source_sha256="source-test",
        manifest_sha256="manifest-test",
    )

    serialized = str(report)
    assert "conteúdo privado" not in serialized
    assert "https://example.test" not in serialized
    assert all(value == 0 for value in report["statistics"]["pairwise_group_overlap"].values())
    assert report["checks"]["groups_do_not_cross_splits"] is True


def test_grouped_manifest_rejects_duplicate_ids() -> None:
    frame = build_synthetic_frame()
    frame.loc[1, "id"] = frame.loc[0, "id"]

    with pytest.raises(ValueError, match="IDs precisam ser únicos"):
        create_grouped_manifest(frame)


def test_canonicalize_group_ignores_query_fragment_and_trailing_slash() -> None:
    first = "https://EXAMPLE.test/post/123/?tracking=abc#section"
    second = "https://example.test/post/123"

    assert canonicalize_group(first) == canonicalize_group(second)
