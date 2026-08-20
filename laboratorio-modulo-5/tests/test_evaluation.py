import numpy as np
import pytest

from laboratorio1.evaluation import F2ThresholdSelector, ModelEvaluator


def test_threshold_selector_finds_perfect_separation():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.3, 0.9, 0.95])

    result = F2ThresholdSelector().select(y_true, scores)

    assert 0.3 < result.threshold <= 0.9
    assert result.f2_score == pytest.approx(1.0)
    assert result.precision_at_threshold == pytest.approx(1.0)
    assert result.recall_at_threshold == pytest.approx(1.0)


def test_evaluator_computes_expected_confusion_counts():
    y_true = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.6, 0.4, 0.8])
    threshold = 0.5

    metrics = ModelEvaluator().evaluate("test-model", y_true, scores, threshold)

    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.true_negatives == 2
    assert metrics.false_negatives == 1
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)


def test_evaluator_reports_captured_fraud_amount_percentage():
    y_true = np.array([0, 1, 1])
    scores = np.array([0.1, 0.9, 0.2])
    amounts = np.array([50.0, 100.0, 200.0])

    metrics = ModelEvaluator().evaluate(
        "test-model", y_true, scores, threshold=0.5, amounts=amounts
    )

    assert metrics.captured_fraud_amount_pct == pytest.approx(100 / 300 * 100)


def test_evaluator_handles_no_negatives_without_dividing_by_zero():
    y_true = np.array([1, 1, 1])
    scores = np.array([0.9, 0.8, 0.95])

    metrics = ModelEvaluator().evaluate("all-fraud", y_true, scores, threshold=0.5)

    assert metrics.specificity == 0.0
    assert metrics.false_alerts_per_10k_legit == 0.0