"""Prepara as planilhas do piloto de anotação de sarcasmo.

Sorteia 100 itens do conjunto de validação (50 ofensivos, 50 não
ofensivos), grava a planilha principal e uma segunda planilha (subconjunto)
para um segundo anotador, e registra um relatório agregado sem texto de
comentários.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_hatespeech_ptbr.data import ID_COLUMN, load_labeled_frame
from rag_hatespeech_ptbr.sarcasm_pilot import (
    PILOT_RANDOM_STATE,
    PILOT_SIZE_PER_CLASS,
    SECOND_ANNOTATOR_FRACTION,
    SECOND_ANNOTATOR_RANDOM_STATE,
    select_pilot_sample,
    select_second_annotator_subset,
    write_annotation_workbook,
)
from rag_hatespeech_ptbr.splits import calculate_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="HateBRXplain.csv bruto")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/split_manifest.csv"),
        help="manifesto local de splits (id, fold, split)",
    )
    parser.add_argument(
        "--pilot-output",
        type=Path,
        default=Path("data/annotations/sarcasm_pilot.xlsx"),
        help="planilha principal (100 itens) para o autor anotar",
    )
    parser.add_argument(
        "--second-annotator-output",
        type=Path,
        default=Path("data/annotations/sarcasm_pilot_second_annotator.xlsx"),
        help="planilha do subconjunto (~15%%) para o segundo anotador",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/tables/sarcasm_pilot_audit.json"),
        help="relatório agregado (ids e contagens, sem texto)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frame = load_labeled_frame(args.dataset, args.manifest)
    pilot = select_pilot_sample(frame, n_per_class=PILOT_SIZE_PER_CLASS, random_state=PILOT_RANDOM_STATE)
    second_annotator = select_second_annotator_subset(
        pilot, fraction=SECOND_ANNOTATOR_FRACTION, random_state=SECOND_ANNOTATOR_RANDOM_STATE
    )

    write_annotation_workbook(pilot, args.pilot_output)
    write_annotation_workbook(second_annotator, args.second_annotator_output)

    report = {
        "pilot_size": len(pilot),
        "pilot_size_per_class": PILOT_SIZE_PER_CLASS,
        "pilot_ids": sorted(int(value) for value in pilot[ID_COLUMN]),
        "second_annotator_subset_size": len(second_annotator),
        "second_annotator_ids": sorted(int(value) for value in second_annotator[ID_COLUMN]),
        "source_split": "validation",
        "random_state_pilot": PILOT_RANDOM_STATE,
        "random_state_second_annotator": SECOND_ANNOTATOR_RANDOM_STATE,
        "dataset_source_sha256": calculate_sha256(args.dataset),
        "manifest_sha256": calculate_sha256(args.manifest),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {key: value for key, value in report.items() if not key.endswith("_ids")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Planilha principal (100 itens) salva em {args.pilot_output}")
    print(f"Planilha do segundo anotador salva em {args.second_annotator_output}")
    print(f"Relatório agregado salvo em {args.report}")


if __name__ == "__main__":
    main()
