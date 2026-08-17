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


def select_additional_sample(
    frame: pd.DataFrame,
    *,
    exclude_ids: set[int],
    n_per_class: int | None = None,
    random_state: int = PILOT_RANDOM_STATE,
) -> pd.DataFrame:
    """Amostra adicional da validação, balanceada, excluindo itens já sorteados.

    `n_per_class=None` pega tudo o que sobrar de cada classe na validação
    (o teto é o tamanho do split de validação, nunca o teste). Serve para
    ampliar o piloto sem re-sortear itens já anotados e sem sair da
    amostragem aleatória (nenhuma curadoria manual de itens "que parecem
    sarcásticos").
    """
    validation = get_split(frame, "validation")
    remaining = validation.loc[~validation[ID_COLUMN].isin(exclude_ids)]

    parts = []
    for label in sorted(remaining[LABEL_COLUMN].unique()):
        candidates = remaining.loc[remaining[LABEL_COLUMN] == label]
        take = len(candidates) if n_per_class is None else min(n_per_class, len(candidates))
        parts.append(candidates.sample(n=take, random_state=random_state))
    sampled = pd.concat(parts, ignore_index=True)
    shuffled = sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

    additional = shuffled[[ID_COLUMN, COMMENT_COLUMN]].copy()
    additional["human_sarcasm_label"] = ""
    additional["sarcasm_evidence"] = ""
    additional["annotation_notes"] = ""
    additional["review_status"] = "pending"
    return additional[ANNOTATION_COLUMNS]


def load_annotation_workbook(path: Path) -> pd.DataFrame:
    """Lê uma planilha de anotação já preenchida e normaliza os tipos."""
    table = pd.read_excel(path, sheet_name="anotacao")
    missing = set(ANNOTATION_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"{path}: colunas ausentes: {sorted(missing)}")
    return table[ANNOTATION_COLUMNS]


def load_annotated_ids(paths: list[Path]) -> set[int]:
    """União dos `id`s presentes em uma ou mais planilhas (preenchidas ou não).

    Usado para não sortear de novo um item que já está em alguma planilha
    em circulação, preenchida ou pendente.
    """
    ids: set[int] = set()
    for path in paths:
        if path.is_file():
            ids.update(int(value) for value in load_annotation_workbook(path)[ID_COLUMN])
    return ids


def compute_agreement(primary: pd.DataFrame, second: pd.DataFrame) -> dict[str, object]:
    """Concordância entre duas anotações, nos itens que as duas cobrem.

    Não inclui texto de comentário no resultado — só `id`s e rótulos
    numéricos, seguro para versionar.
    """
    from sklearn.metrics import cohen_kappa_score

    merged = primary[[ID_COLUMN, "human_sarcasm_label"]].merge(
        second[[ID_COLUMN, "human_sarcasm_label"]],
        on=ID_COLUMN,
        how="inner",
        suffixes=("_primary", "_second"),
    )
    for column in ("human_sarcasm_label_primary", "human_sarcasm_label_second"):
        merged[column] = pd.to_numeric(merged[column], errors="raise").astype(int)

    if merged.empty:
        raise ValueError("Nenhum id em comum entre as duas planilhas.")

    agrees = merged["human_sarcasm_label_primary"] == merged["human_sarcasm_label_second"]
    disagreements = merged.loc[~agrees]

    confusion: dict[str, dict[str, int]] = {}
    for primary_label, group in merged.groupby("human_sarcasm_label_primary"):
        confusion[str(primary_label)] = (
            group["human_sarcasm_label_second"].value_counts().astype(int).to_dict()
        )
        confusion[str(primary_label)] = {
            str(key): int(value) for key, value in confusion[str(primary_label)].items()
        }

    return {
        "n_common": len(merged),
        "exact_agreement": int(agrees.sum()),
        "exact_agreement_rate": round(float(agrees.mean()), 4),
        "cohen_kappa": round(
            float(
                cohen_kappa_score(
                    merged["human_sarcasm_label_primary"], merged["human_sarcasm_label_second"]
                )
            ),
            4,
        ),
        "confusion_primary_rows_second_cols": confusion,
        "disagreement_ids": sorted(int(value) for value in disagreements[ID_COLUMN]),
    }


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
