import numpy as np
import pytest

from laboratorio1.evaluation import F2ThresholdSelector, ModelEvaluator


def test_threshold_selector_finds_perfect_separation():
    labels = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.3, 0.9, 0.95])

    result = F2ThresholdSelector().select(labels, scores)

    assert result.threshold == pytest.approx(0.9)
    assert result.f2_score == pytest.approx(1.0)
    assert result.precision_at_threshold == pytest.approx(1.0)
    assert result.recall_at_threshold == pytest.approx(1.0)


def test_evaluator_computes_notebook_metrics():
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.6, 0.4, 0.8])

    metrics = ModelEvaluator().evaluate("model", labels, scores, threshold=0.5)

    assert metrics.confusion_matrix == ((2, 1), (1, 1))
    assert metrics.accuracy == pytest.approx(0.6)
    assert metrics.balanced_accuracy == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.specificity == pytest.approx(2 / 3)
    assert metrics.false_alerts_per_10k_legit == pytest.approx(10_000 / 3)


def test_evaluator_reports_captured_fraud_amount_percentage():
    metrics = ModelEvaluator().evaluate(
        "model",
        np.array([0, 1, 1]),
        np.array([0.1, 0.9, 0.2]),
        threshold=0.5,
        amounts=np.array([50.0, 100.0, 200.0]),
    )

    assert metrics.captured_fraud_amount_pct == pytest.approx(100 / 300 * 100)


def test_evaluator_handles_single_class():
    metrics = ModelEvaluator().evaluate(
        "all-fraud",
        np.ones(3),
        np.array([0.9, 0.8, 0.95]),
        threshold=0.5,
    )

    assert metrics.specificity == 0.0
    assert metrics.false_alerts_per_10k_legit == 0.0
    assert np.isnan(metrics.roc_auc)
    assert np.isnan(metrics.pr_auc)


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        (np.array([]), np.array([]), "vacíos"),
        (np.array([0, 1]), np.array([0.1]), "misma longitud"),
        (np.array([0, 2]), np.array([0.1, 0.2]), "solo puede contener"),
    ],
)
def test_evaluator_rejects_invalid_vectors(labels, scores, message):
    with pytest.raises(ValueError, match=message):
        ModelEvaluator().evaluate("model", labels, scores, threshold=0.5)


def test_threshold_selector_requires_both_classes():
    with pytest.raises(ValueError, match="ambas clases"):
        F2ThresholdSelector().select(np.ones(3), np.array([0.1, 0.2, 0.3]))
