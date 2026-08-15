import pytest

from rag_hatespeech_ptbr.metrics import evaluate_predictions


def test_evaluate_predictions_perfect_predictions() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]

    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics.positive_f1 == 1.0
    assert metrics.positive_precision == 1.0
    assert metrics.positive_recall == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.confusion_matrix == [[2, 0], [0, 2]]
    assert metrics.n_samples == 4


def test_evaluate_predictions_all_wrong() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [1, 0, 1, 0]

    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics.positive_f1 == 0.0
    assert metrics.confusion_matrix == [[0, 2], [2, 0]]


def test_evaluate_predictions_applies_mask_for_stratified_analysis() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    mask = [True, True, False, False]  # mantém só os dois primeiros itens

    metrics = evaluate_predictions(y_true, y_pred, mask=mask)

    assert metrics.n_samples == 2
    assert metrics.positive_f1 == 1.0


def test_evaluate_predictions_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="mesmo tamanho"):
        evaluate_predictions([0, 1], [0])


def test_evaluate_predictions_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="zero amostras"):
        evaluate_predictions([], [])


def test_evaluate_predictions_rejects_mismatched_mask_length() -> None:
    with pytest.raises(ValueError, match="mask deve ter o mesmo tamanho"):
        evaluate_predictions([0, 1], [0, 1], mask=[True])
