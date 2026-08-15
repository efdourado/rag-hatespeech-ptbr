"""Seleção determinística do piloto de anotação de sarcasmo.

Sorteia 100 itens do conjunto de validação (nunca do teste), balanceados
por `offensive_label`, e prepara as planilhas de anotação descritas em
11_diretrizes_anotacao.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rag_hatespeech_ptbr.data import COMMENT_COLUMN, ID_COLUMN, LABEL_COLUMN, get_split

PILOT_RANDOM_STATE = 42
PILOT_SIZE_PER_CLASS = 50
SECOND_ANNOTATOR_RANDOM_STATE = 43
SECOND_ANNOTATOR_FRACTION = 0.15

ANNOTATION_COLUMNS = [
    ID_COLUMN,
    COMMENT_COLUMN,
    "human_sarcasm_label",
    "sarcasm_evidence",
    "annotation_notes",
    "review_status",
]


def select_pilot_sample(
    frame: pd.DataFrame,
    *,
    n_per_class: int = PILOT_SIZE_PER_CLASS,
    random_state: int = PILOT_RANDOM_STATE,
) -> pd.DataFrame:
    """Amostra balanceada por ofensividade, exclusivamente do split de validação.

    Nunca inclui `offensive_label` na saída: a anotação de sarcasmo deve
    ocorrer sem o rótulo de ofensividade visível
    (11_diretrizes_anotacao.md, "Unidade de anotação").
    """
    validation = get_split(frame, "validation")
    parts = [
        validation.loc[validation[LABEL_COLUMN] == label].sample(
            n=n_per_class, random_state=random_state
        )
        for label in sorted(validation[LABEL_COLUMN].unique())
    ]
    sampled = pd.concat(parts, ignore_index=True)
    shuffled = sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

    pilot = shuffled[[ID_COLUMN, COMMENT_COLUMN]].copy()
    pilot["human_sarcasm_label"] = ""
    pilot["sarcasm_evidence"] = ""
    pilot["annotation_notes"] = ""
    pilot["review_status"] = "pending"
    return pilot[ANNOTATION_COLUMNS]


def select_second_annotator_subset(
    pilot: pd.DataFrame,
    *,
    fraction: float = SECOND_ANNOTATOR_FRACTION,
    random_state: int = SECOND_ANNOTATOR_RANDOM_STATE,
) -> pd.DataFrame:
    """Subconjunto do piloto para concordância entre anotadores (Cohen's kappa).

    Os mesmos `id`s do piloto principal, para que as duas anotações possam
    ser comparadas item a item.
    """
    n = max(1, round(len(pilot) * fraction))
    subset = pilot.sample(n=n, random_state=random_state).reset_index(drop=True)
    subset = subset.copy()
    subset["human_sarcasm_label"] = ""
    subset["sarcasm_evidence"] = ""
    subset["annotation_notes"] = ""
    subset["review_status"] = "pending"
    return subset[ANNOTATION_COLUMNS]


def write_annotation_workbook(table: pd.DataFrame, path: Path) -> None:
    """Grava a planilha de anotação em `.xlsx`, formatada para uso direto.

    Cabeçalho congelado, coluna de comentário com quebra de linha, e
    validação de dados (menu suspenso) nas colunas `human_sarcasm_label`
    e `review_status`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_excel(path, index=False, sheet_name="anotacao")
    _format_workbook(path, n_rows=len(table))


def _format_workbook(path: Path, *, n_rows: int) -> None:
    from openpyxl import load_workbook
    from openpyxl.worksheet.datavalidation import DataValidation

    workbook = load_workbook(path)
    sheet = workbook.active

    column_widths = {"A": 12, "B": 90, "C": 20, "D": 45, "E": 45, "F": 16}
    for column, width in column_widths.items():
        sheet.column_dimensions[column].width = width

    last_row = n_rows + 1
    for row in sheet.iter_rows(min_row=2, max_row=last_row, min_col=2, max_col=2):
        for cell in row:
            cell.alignment = cell.alignment.copy(wrap_text=True, vertical="top")

    label_validation = DataValidation(type="list", formula1='"0,1,2"', allow_blank=True)
    sheet.add_data_validation(label_validation)
    label_validation.add(f"C2:C{last_row}")

    status_validation = DataValidation(
        type="list", formula1='"pending,reviewed,adjudicated"', allow_blank=True
    )
    sheet.add_data_validation(status_validation)
    status_validation.add(f"F2:F{last_row}")

    sheet.freeze_panes = "A2"
    workbook.save(path)
