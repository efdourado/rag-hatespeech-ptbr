"""Calcula concordância entre duas planilhas de anotação de sarcasmo.

Compara os rótulos nos ids em comum entre a planilha principal e a do
segundo anotador: concordância exata, Cohen's kappa, matriz de confusão e
a lista de ids em desacordo (sem texto de comentário).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_hatespeech_ptbr.sarcasm_pilot import compute_agreement, load_annotation_workbook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary",
        type=Path,
        default=Path("data/annotations/sarcasm_pilot.xlsx"),
        help="planilha do anotador principal",
    )
    parser.add_argument(
        "--second",
        type=Path,
        default=Path("data/annotations/sarcasm_pilot_second_annotator.xlsx"),
        help="planilha do segundo anotador",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/tables/sarcasm_annotation_agreement.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    primary = load_annotation_workbook(args.primary)
    second = load_annotation_workbook(args.second)
    result = compute_agreement(primary, second)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Relatório salvo em {args.report}")
    print(
        "Interpretação (Landis & Koch, 1977): kappa < 0.20 = concordância "
        "leve; 0.21-0.40 = razoável; 0.41-0.60 = moderada; 0.61-0.80 = "
        "substancial; > 0.80 = quase perfeita. Com poucos itens e classes "
        "desbalanceadas, kappa pode ficar baixo mesmo com alta concordância "
        "bruta — isso é esperado, não indica erro no cálculo."
    )


if __name__ == "__main__":
    main()
