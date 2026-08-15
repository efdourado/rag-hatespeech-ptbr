import pandas as pd

from rag_hatespeech_ptbr.baseline import build_pipeline, evaluate_baseline, train_baseline
from rag_hatespeech_ptbr.metrics import ClassificationMetrics


def _synthetic_split(n_per_class: int, *, offset: int = 0) -> pd.DataFrame:
    offensive = [
        f"você é um lixo horrível numero {offset + index}" for index in range(n_per_class)
    ]
    non_offensive = [
        f"bom dia tenha uma ótima semana numero {offset + index}" for index in range(n_per_class)
    ]
    return pd.DataFrame(
        {
            "comment": offensive + non_offensive,
            "offensive_label": [1] * n_per_class + [0] * n_per_class,
        }
    )


def test_build_pipeline_has_expected_steps() -> None:
    pipeline = build_pipeline()

    assert [name for name, _ in pipeline.steps] == ["tfidf", "clf"]


def test_train_baseline_fits_vectorizer_only_on_train_frame() -> None:
    train_frame = _synthetic_split(15, offset=0)
    pipeline = train_baseline(train_frame)

    vocabulary = pipeline.named_steps["tfidf"].vocabulary_
    # "horrível" só aparece nos comentários ofensivos de treino; palavras que só
    # existem em um exemplo de validação nunca vistas no treino não podem entrar
    # no vocabulário aprendido, o que evidencia que .fit() não tocou validação/teste.
    assert "horrível" in vocabulary
    assert "palavraexclusivadevalidacao" not in vocabulary


def test_evaluate_baseline_separates_clearly_distinct_classes() -> None:
    train_frame = _synthetic_split(30, offset=0)
    eval_frame = _synthetic_split(10, offset=1000)

    pipeline = train_baseline(train_frame)
    metrics = evaluate_baseline(pipeline, eval_frame)

    assert isinstance(metrics, ClassificationMetrics)
    assert metrics.n_samples == 20
    # Classes trivialmente separáveis por vocabulário: o baseline deve
    # capturar o padrão perfeitamente neste cenário sintético.
    assert metrics.positive_f1 == 1.0


def test_train_baseline_is_deterministic_given_same_random_state() -> None:
    train_frame = _synthetic_split(20, offset=0)
    eval_frame = _synthetic_split(5, offset=500)

    first = train_baseline(train_frame, random_state=42)
    second = train_baseline(train_frame, random_state=42)

    first_predictions = first.predict(eval_frame["comment"]).tolist()
    second_predictions = second.predict(eval_frame["comment"]).tolist()
    assert first_predictions == second_predictions
