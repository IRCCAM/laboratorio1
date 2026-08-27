from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from laboratorio1.api import create_app
from laboratorio1.modeling.remote import (
    MLPServiceResponseError,
    MLPServiceUnavailable,
)


class FakeFraudService:
    def __init__(self, *, ready=True, prediction_error=None):
        self.threshold = 0.5
        self.service_url = "http://mlp-test:8080"
        self.ready = ready
        self.prediction_error = prediction_error
        self.closed = False

    def is_ready(self):
        return self.ready

    def predict(self, transactions):
        if self.prediction_error is not None:
            raise self.prediction_error
        count = len(transactions)
        return pd.DataFrame(
            {
                "probabilidad_fraude": np.linspace(0.2, 0.8, count),
                "alerta_fraude": [0] * (count - 1) + [1],
            }
        )

    def close(self):
        self.closed = True


def transaction(value=0.0):
    return {
        "Time": 100.0,
        "Amount": 25.5,
        **{f"V{index}": value for index in range(1, 29)},
    }


def test_predict_returns_mlp_model_results():
    app = create_app(lambda: FakeFraudService())

    with TestClient(app) as client:
        response = client.post("/predict", json={"data": [transaction(), transaction(0.1)]})

    assert response.status_code == 200
    body = response.json()
    assert body["total_predicciones"] == 2
    assert body["umbral"] == 0.5
    assert body["resultados"][1] == {
        "index": 1,
        "probabilidad_fraude": 0.8,
        "alerta_fraude": 1,
    }


def test_predict_rejects_incomplete_transaction():
    app = create_app(lambda: FakeFraudService())

    with TestClient(app) as client:
        response = client.post("/predict", json={"data": [{"Time": 10, "Amount": 5}]})

    assert response.status_code == 422


def test_health_checks_remote_mlp_and_closes_client():
    service = FakeFraudService()
    app = create_app(lambda: service)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["servicio_mlp"] == "http://mlp-test:8080"
    assert service.closed is True


def test_health_reports_remote_mlp_unavailable():
    app = create_app(lambda: FakeFraudService(ready=False))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert "contenedor" in response.json()["detail"]["error"]


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (MLPServiceUnavailable("modelo desconectado"), 503),
        (MLPServiceResponseError("respuesta inválida"), 502),
    ],
)
def test_predict_maps_remote_model_errors(error, expected_status):
    app = create_app(lambda: FakeFraudService(prediction_error=error))

    with TestClient(app) as client:
        response = client.post("/predict", json={"data": [transaction()]})

    assert response.status_code == expected_status
    assert response.json()["detail"] == str(error)


def test_health_reports_model_load_failure():
    def fail_loading():
        raise FileNotFoundError("No se encontró el modelo")

    app = create_app(fail_loading)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert "No se encontró el modelo" in response.json()["detail"]["error"]
