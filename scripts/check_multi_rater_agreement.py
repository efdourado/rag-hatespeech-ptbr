"""Calcula concordância entre você e um ou mais verificadores adicionais.

Reporta o kappa par a par (você vs. cada verificador) e, com 2+
verificadores respondidos, o Fleiss' kappa conjunto na interseção dos
itens que todos rotularam.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_hatespeech_ptbr.sarcasm_pilot import (
    compute_multi_rater_agreement,
    load_annotation_workbook,
    load_labeled_pool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary",
        type=Path,
        nargs="+",
        default=[
            Path("data/annotations/sarcasm_pilot.xlsx"),
            Path("data/annotations/sarcasm_pilot_batch2.xlsx"),
        ],
        help="sua(s) planilha(s) de referência",
    )
    parser.add_argument(
        "--checker",
        type=Path,
        nargs="+",
        required=True,
        help="uma ou mais planilhas de verificadores já respondidas",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/tables/sarcasm_multi_rater_agreement.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    primary = load_labeled_pool(args.primary)
    checkers = {path.stem: load_annotation_workbook(path) for path in args.checker}

    result = compute_multi_rater_agreement(primary, checkers)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Relatório salvo em {args.report}")


if __name__ == "__main__":
    main()
