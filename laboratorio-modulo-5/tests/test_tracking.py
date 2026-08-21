import pytest

from laboratorio1.tracking import normalized_metrics


def test_normalized_metrics_maps_report_names_and_discards_text():
    result = normalized_metrics(
        {
            "Modelo": "MLP",
            "PR-AUC": 0.75,
            "F2": 0.8,
            "TP": 10,
            "Monto de fraude capturado (%)": None,
        }
    )

    assert result == {
        "pr_auc": pytest.approx(0.75),
        "f2": pytest.approx(0.8),
        "true_positives": pytest.approx(10.0),
    }
