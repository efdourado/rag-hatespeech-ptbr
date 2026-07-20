from pathlib import Path

import pandas as pd

from scripts.inspect_dataset import build_report


def test_build_report_returns_only_aggregate_metadata() -> None:
    frame = pd.DataFrame(
        {"comment": ["texto sensível", "texto sensível"], "offensive_label": [1, 1]}
    )

    report = build_report(frame, Path("dataset.csv"))

    assert report["rows"] == 2
    assert report["exact_duplicate_rows"] == 1
    assert report["null_count"] == {"comment": 0, "offensive_label": 0}
    assert "texto sensível" not in str(report)
