"""Métricas de classificação compartilhadas por baseline, LLM e RAG.

Centralizado aqui para que todos os modelos futuros (baseline determinístico,
LLM sem recuperação, RAG) sejam avaliados exatamente da mesma forma, incluindo
a futura análise por estrato de sarcasmo (via `mask`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

POSITIVE_LABEL = 1


@dataclass(frozen=True)
class ClassificationMetrics:
    n_samples: int
    positive_f1: float
    positive_precision: float
    positive_recall: float
    macro_f1: float
    confusion_matrix: list[list[int]]  # [[TN, FP], [FN, TP]], rótulos fixos em [0, 1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "positive_f1": self.positive_f1,
            "positive_precision": self.positive_precision,
            "positive_recall": self.positive_recall,
            "macro_f1": self.macro_f1,
            "confusion_matrix": self.confusion_matrix,
        }


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    mask: Sequence[bool] | None = None,
) -> ClassificationMetrics:
    """Calcula as métricas principais do projeto (F1 da classe ofensiva é a métrica-alvo).

    `mask` permite restringir a avaliação a um subconjunto (ex.: apenas
    comentários anotados como sarcásticos), sem duplicar esta função quando a
    análise por estrato de sarcasmo for implementada.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred devem ter o mesmo tamanho.")

    if mask is not None:
        if len(mask) != len(y_true):
            raise ValueError("mask deve ter o mesmo tamanho que y_true.")
        y_true = [value for value, keep in zip(y_true, mask) if keep]
        y_pred = [value for value, keep in zip(y_pred, mask) if keep]

    if len(y_true) == 0:
        raise ValueError("Não é possível calcular métricas com zero amostras.")

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return ClassificationMetrics(
        n_samples=len(y_true),
        positive_f1=float(
            f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        positive_precision=float(
            precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        positive_recall=float(
            recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        confusion_matrix=matrix.tolist(),
    )
