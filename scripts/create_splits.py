"""Cria e audita partições agrupadas do HateBRXplain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rag_hatespeech_ptbr.splits import (
    EXPECTED_SOURCE_SHA256,
    build_split_report,
    calculate_sha256,
    create_grouped_manifest,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="HateBRXplain.csv bruto")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/split_manifest.csv"),
        help="destino local do manifesto por ID",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/tables/split_audit.json"),
        help="destino do relatório agregado",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset não encontrado: {args.dataset}")

    source_sha256 = calculate_sha256(args.dataset)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise SystemExit("Hash do dataset diferente da revisão aprovada; divisão cancelada.")

    frame = pd.read_csv(args.dataset)
    manifest = create_grouped_manifest(frame)
    manifest_sha256 = write_manifest(manifest, args.manifest)
    report = build_split_report(
        frame,
        manifest,
        source_sha256=source_sha256,
        manifest_sha256=manifest_sha256,
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Manifesto local salvo em {args.manifest}")
    print(f"Relatório agregado salvo em {args.report}")
    if not report["valid"]:
        raise SystemExit("Divisão reprovada; consulte failed_checks no relatório.")


if __name__ == "__main__":
    main()
