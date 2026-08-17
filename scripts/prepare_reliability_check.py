"""Gera planilhas de checagem de confiabilidade para mais verificadores.

Sorteia N itens que você já rotulou (não anotação nova) e grava cópias em
branco, uma por verificador, para pessoas diferentes julgarem os mesmos
itens de forma independente. Serve para medir o quanto o seu critério se
sustenta com outra pessoa olhando o mesmo comentário — não amplia a base,
só mede confiabilidade.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_hatespeech_ptbr.sarcasm_pilot import (
    load_annotated_ids,
    load_labeled_pool,
    select_reliability_check_subset,
    write_annotation_workbook,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotated",
        type=Path,
        nargs="+",
        default=[
            Path("data/annotations/sarcasm_pilot.xlsx"),
            Path("data/annotations/sarcasm_pilot_batch2.xlsx"),
        ],
        help="planilhas já rotuladas por você, das quais sortear o subconjunto",
    )
    parser.add_argument(
        "--already-checked",
        type=Path,
        nargs="*",
        default=[Path("data/annotations/sarcasm_pilot_second_annotator.xlsx")],
        help="planilhas de verificadores anteriores, para não repetir os mesmos itens",
    )
    parser.add_argument(
        "--n-items", type=int, default=30, help="quantos itens sortear (padrão: 30)"
    )
    parser.add_argument(
        "--n-checkers", type=int, default=3, help="quantas cópias gerar (padrão: 3)"
    )
    parser.add_argument("--random-state", type=int, default=44)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("data/annotations/sarcasm_reliability_check"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/tables/sarcasm_reliability_check_audit.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pool = load_labeled_pool(args.annotated)
    exclude_ids = load_annotated_ids(args.already_checked)
    subset = select_reliability_check_subset(
        pool, n=args.n_items, random_state=args.random_state, exclude_ids=exclude_ids
    )

    output_paths = []
    for index in range(1, args.n_checkers + 1):
        path = Path(f"{args.output_prefix}_{index}.xlsx")
        write_annotation_workbook(subset, path)
        output_paths.append(str(path))

    report = {
        "n_items": len(subset),
        "n_checkers": args.n_checkers,
        "item_ids": sorted(int(value) for value in subset["id"]),
        "excluded_already_checked_ids": len(exclude_ids),
        "labeled_pool_size": len(pool),
        "random_state": args.random_state,
        "output_files": output_paths,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Poço de itens já rotulados por você: {len(pool)}")
    print(f"Itens já checados antes (excluídos): {len(exclude_ids)}")
    print(f"Subconjunto sorteado para checagem: {len(subset)} itens")
    for path in output_paths:
        print(f"  {path}")
    print(f"Relatório salvo em {args.report}")


if __name__ == "__main__":
    main()
