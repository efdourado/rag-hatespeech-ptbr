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


def _is_labeled(series: pd.Series) -> pd.Series:
    """True onde o valor é um rótulo válido (0, 1 ou 2).

    Coage para numérico em vez de comparar strings: concatenar planilhas
    onde uma tem rótulos preenchidos (inteiros) e outra ainda está em
    branco (lida de volta do Excel como `NaN`) força a coluna inteira
    para `float64` — "0" preenchido vira `0.0`, que uma comparação de
    string ingênua ("0.0" != "0") deixaria passar como não rotulado.
    """
    return pd.to_numeric(series, errors="coerce").isin([0, 1, 2])


def load_labeled_pool(paths: list[Path]) -> pd.DataFrame:
    """Concatena planilhas já em uso e mantém só as linhas com rótulo preenchido.

    É o "poço" de itens já julgados por você mesmo, do qual um novo
    verificador de confiabilidade pode ser sorteado — nunca inclui linhas
    ainda `pending`/em branco, porque não há com o que comparar.
    """
    frames = [load_annotation_workbook(path) for path in paths if path.is_file()]
    if not frames:
        raise FileNotFoundError(f"Nenhuma planilha encontrada em {paths}")
    combined = pd.concat(frames, ignore_index=True)
    if combined[ID_COLUMN].duplicated().any():
        raise ValueError("Ids duplicados entre as planilhas informadas.")
    return combined.loc[_is_labeled(combined["human_sarcasm_label"])].reset_index(drop=True)


def select_reliability_check_subset(
    labeled_pool: pd.DataFrame,
    *,
    n: int,
    random_state: int,
    exclude_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Sorteia `n` itens já rotulados por você, em branco para um novo verificador.

    Usado para checar concordância com mais pessoas sobre o que você já
    anotou, sem gerar trabalho de anotação nova nem depender de terceiros
    para ampliar a base — só mede o quanto os rótulos que você já deu se
    sustentam com outra pessoa olhando o mesmo item.
    """
    exclude_ids = exclude_ids or set()
    candidates = labeled_pool.loc[~labeled_pool[ID_COLUMN].isin(exclude_ids)]
    if len(candidates) < n:
        raise ValueError(
            f"Só há {len(candidates)} itens rotulados disponíveis, mas foram pedidos {n}."
        )
    subset = candidates.sample(n=n, random_state=random_state).reset_index(drop=True)

    blind = subset[[ID_COLUMN, COMMENT_COLUMN]].copy()
    blind["human_sarcasm_label"] = ""
    blind["sarcasm_evidence"] = ""
    blind["annotation_notes"] = ""
    blind["review_status"] = "pending"
    return blind[ANNOTATION_COLUMNS]


def fleiss_kappa(labels: pd.DataFrame) -> float:
    """Fleiss' kappa para 3+ avaliadores nas mesmas colunas 0/1/2.

    `labels`: uma linha por item, uma coluna por avaliador, valores em
    {0, 1, 2}. Todas as colunas devem estar completamente preenchidas
    (sem valores ausentes) para os mesmos itens.
    """
    categories = [0, 1, 2]
    n_items, n_raters = labels.shape
    if n_raters < 2:
        raise ValueError("Fleiss' kappa exige pelo menos 2 avaliadores.")

    counts = pd.DataFrame(
        {
            category: (labels == category).sum(axis=1)
            for category in categories
        }
    )
    if not (counts.sum(axis=1) == n_raters).all():
        raise ValueError("Cada item precisa ter exatamente `n_raters` rótulos válidos.")

    p_item = (counts.pow(2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_item.mean()

    p_category = counts.sum(axis=0) / (n_items * n_raters)
    p_e = (p_category.pow(2)).sum()

    if p_e == 1:
        return 1.0
    return float((p_bar - p_e) / (1 - p_e))


def compute_multi_rater_agreement(
    primary: pd.DataFrame, checkers: dict[str, pd.DataFrame]
) -> dict[str, object]:
    """Concordância entre você e um ou mais verificadores adicionais.

    Reporta o kappa par a par (você vs. cada verificador) e, se houver
    2 ou mais verificadores com itens em comum, o Fleiss' kappa conjunto
    (você + todos) na interseção dos itens que todos rotularam.
    """
    pairwise = {
        name: compute_agreement(primary, checker) for name, checker in checkers.items()
    }

    result: dict[str, object] = {"pairwise": pairwise}

    if len(checkers) >= 2:
        merged = primary[[ID_COLUMN, "human_sarcasm_label"]].rename(
            columns={"human_sarcasm_label": "primary"}
        )
        for name, checker in checkers.items():
            merged = merged.merge(
                checker[[ID_COLUMN, "human_sarcasm_label"]].rename(
                    columns={"human_sarcasm_label": name}
                ),
                on=ID_COLUMN,
                how="inner",
            )
        rater_columns = ["primary", *checkers.keys()]
        for column in rater_columns:
            merged[column] = pd.to_numeric(merged[column], errors="raise").astype(int)

        if len(merged) > 0:
            result["fleiss_kappa"] = round(fleiss_kappa(merged[rater_columns]), 4)
            result["fleiss_kappa_n_items"] = len(merged)
            result["fleiss_kappa_raters"] = rater_columns
        else:
            result["fleiss_kappa"] = None
            result["fleiss_kappa_n_items"] = 0

    return result


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
