"""Baseline determinístico de classificação de ofensividade.

TF-IDF + regressão logística, sem LLM e sem recuperação. Serve como piso de
comparação para os experimentos de LLM e RAG futuros (ver
docs/7_arquitetura_pipeline.md, item 7). Usa apenas dependências já existentes
no projeto (scikit-learn).
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from rag_hatespeech_ptbr.data import COMMENT_COLUMN, LABEL_COLUMN
from rag_hatespeech_ptbr.metrics import ClassificationMetrics, evaluate_predictions

RANDOM_STATE = 42
MODEL_NAME = "tfidf_logreg_baseline"


def build_pipeline(*, random_state: int = RANDOM_STATE) -> Pipeline:
    """Monta o pipeline TF-IDF + regressão logística com hiperparâmetros fixos."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents=None,  # preserva acentos: sinal relevante em PT-BR
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_baseline(train_frame: pd.DataFrame, *, random_state: int = RANDOM_STATE) -> Pipeline:
    """Ajusta o pipeline exclusivamente sobre `train_frame`.

    O `Pipeline` do scikit-learn garante que o vocabulário do TF-IDF também é
    aprendido apenas nos dados passados aqui; `evaluate_baseline` só chama
    `.transform()` (via `.predict()`), nunca `.fit()`, sobre validação/teste.
    """
    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(train_frame[COMMENT_COLUMN], train_frame[LABEL_COLUMN])
    return pipeline


def evaluate_baseline(pipeline: Pipeline, frame: pd.DataFrame) -> ClassificationMetrics:
    """Avalia um pipeline já treinado sobre qualquer split (validação ou teste)."""
    predictions = pipeline.predict(frame[COMMENT_COLUMN])
    return evaluate_predictions(frame[LABEL_COLUMN].to_list(), predictions.tolist())
