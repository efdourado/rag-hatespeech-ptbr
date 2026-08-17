"""Amplia a amostra de anotação de sarcasmo além do piloto de 100 itens.

Sorteia mais itens da validação, balanceados por ofensividade, excluindo
qualquer id que já esteja em alguma planilha existente (preenchida ou não).
Nunca toca o teste. Por padrão, pega tudo o que sobrar na validação.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_hatespeech_ptbr.data import ID_COLUMN, load_labeled_frame
from rag_hatespeech_ptbr.sarcasm_pilot import (
    PILOT_RANDOM_STATE,
    load_annotated_ids,
    select_additional_sample,
    write_annotation_workbook,
)
from rag_hatespeech_ptbr.splits import calculate_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="HateBRXplain.csv bruto")
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/split_manifest.csv")
    )
    parser.add_argument(
        "--existing",
        type=Path,
        nargs="*",
        default=[
            Path("data/annotations/sarcasm_pilot.xlsx"),
            Path("data/annotations/sarcasm_pilot_second_annotator.xlsx"),
        ],
        help="planilhas já em circulação, para não repetir ids",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/annotations/sarcasm_pilot_batch2.xlsx"),
        help="destino da nova planilha",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=None,
        help="itens por classe a sortear (padrão: tudo o que sobrar na validação)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/tables/sarcasm_sample_expansion_audit.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frame = load_labeled_frame(args.dataset, args.manifest)
    exclude_ids = load_annotated_ids(args.existing)
    additional = select_additional_sample(
        frame,
        exclude_ids=exclude_ids,
        n_per_class=args.n_per_class,
        random_state=PILOT_RANDOM_STATE,
    )

    write_annotation_workbook(additional, args.output)

    report = {
        "already_annotated_ids_excluded": len(exclude_ids),
        "new_batch_size": len(additional),
        "new_batch_ids": sorted(int(value) for value in additional[ID_COLUMN]),
        "source_split": "validation",
        "dataset_source_sha256": calculate_sha256(args.dataset),
        "manifest_sha256": calculate_sha256(args.manifest),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Itens já anotados/em circulação excluídos: {len(exclude_ids)}")
    print(f"Novo lote: {len(additional)} itens")
    print(f"Planilha salva em {args.output}")
    print(f"Relatório salvo em {args.report}")


if __name__ == "__main__":
    main()
