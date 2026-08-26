import json

import httpx2
import joblib
import numpy as np
import pytest

from laboratorio1.config import ArtifactPaths
from laboratorio1.features import FraudPreprocessor
from laboratorio1.modeling.remote import (
    MLPServiceResponseError,
    MLPServiceUnavailable,
    RemoteMLPFraudDetectionService,
)


def build_service(transactions_factory, handler, threshold=0.5):
    preprocessor = FraudPreprocessor().fit(transactions_factory(20))
    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    return RemoteMLPFraudDetectionService(
        preprocessor=preprocessor,
        threshold=threshold,
        service_url="http://mlp-test:8080/",
        client=client,
    )


def test_remote_service_sends_32_features_and_returns_alerts(transactions_factory):
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["payload"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json={"predictions": [[0.2], [0.8]]},
            request=request,
        )

    service = build_service(transactions_factory, handler)
    transactions = transactions_factory(20).iloc[:2]

    result = service.predict(transactions)

    assert captured["path"] == "/invocations"
    assert len(captured["payload"]["inputs"]) == 2
    assert all(len(row) == 32 for row in captured["payload"]["inputs"])
    assert np.allclose(result["probabilidad_fraude"], [0.2, 0.8])
    assert result["alerta_fraude"].tolist() == [0, 1]


def test_remote_service_health_uses_model_container(transactions_factory):
    def handler(request):
        assert request.url.path == "/health"
        return httpx2.Response(200, request=request)

    service = build_service(transactions_factory, handler)

    assert service.is_ready() is True


def test_remote_service_maps_connection_failure(transactions_factory):
    def handler(request):
        raise httpx2.ConnectError("modelo apagado", request=request)

    service = build_service(transactions_factory, handler)

    with pytest.raises(MLPServiceUnavailable, match="No fue posible conectar"):
        service.predict(transactions_factory(20).iloc[:1])


@pytest.mark.parametrize(
    "response_payload",
    [
        {},
        {"predictions": [[0.1, 0.2]]},
        {"predictions": [["NaN"]]},
        {"predictions": [[1.5]]},
    ],
)
def test_remote_service_rejects_invalid_contract(
    transactions_factory,
    response_payload,
):
    def handler(request):
        return httpx2.Response(200, json=response_payload, request=request)

    service = build_service(transactions_factory, handler)

    with pytest.raises(MLPServiceResponseError):
        service.predict(transactions_factory(20).iloc[:1])


def test_remote_service_maps_http_error(transactions_factory):
    def handler(request):
        return httpx2.Response(500, request=request)

    service = build_service(transactions_factory, handler)

    with pytest.raises(MLPServiceResponseError, match="HTTP 500"):
        service.predict(transactions_factory(20).iloc[:1])


def test_from_artifacts_does_not_require_local_keras_model(
    tmp_path,
    transactions_factory,
):
    model_dir = tmp_path / "models"
    result_dir = tmp_path / "results"
    model_dir.mkdir()
    paths = ArtifactPaths(
        mlp_model=model_dir / "missing.keras",
        autoencoder_model=model_dir / "missing-autoencoder.keras",
        preprocessor=model_dir / "preprocessor.joblib",
        thresholds=model_dir / "thresholds.json",
        metrics=result_dir / "metrics.csv",
        provenance=result_dir / "provenance.json",
    )
    joblib.dump(FraudPreprocessor().fit(transactions_factory()), paths.preprocessor)
    paths.thresholds.write_text(json.dumps({"mlp": 0.5}), encoding="utf-8")

    service = RemoteMLPFraudDetectionService.from_artifacts(
        paths,
        service_url="http://mlp-test:8080",
        timeout_seconds=1,
    )

    assert service.threshold == 0.5
    assert service.service_url == "http://mlp-test:8080"
    service.close()
