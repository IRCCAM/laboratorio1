import json
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from laboratorio1.config import ArtifactPaths
from laboratorio1.features import FraudPreprocessor
from laboratorio1.modeling.predict import FraudDetectionService


class ScoreModel:
    def predict(self, features, **kwargs):
        del features, kwargs
        return np.array([[0.2], [0.8]], dtype=np.float32)


class OffsetAutoencoder:
    def predict(self, features, **kwargs):
        del kwargs
        offsets = np.array([0.1, 0.5], dtype=np.float32)[:, None]
        return features + offsets


def test_service_returns_both_model_scores_and_alerts(transactions_factory):
    transactions = transactions_factory(10).iloc[:2]
    preprocessor = FraudPreprocessor().fit(transactions_factory(10))
    service = FraudDetectionService(
        preprocessor,
        ScoreModel(),
        OffsetAutoencoder(),
        {"mlp": 0.5, "autoencoder": 0.3},
    )

    result = service.predict(transactions)

    assert result.columns.tolist() == [
        "probabilidad_mlp",
        "alerta_mlp",
        "error_autoencoder",
        "alerta_autoencoder",
    ]
    assert result["alerta_mlp"].tolist() == [0, 1]
    assert result["alerta_autoencoder"].tolist() == [0, 1]
    assert np.allclose(result["error_autoencoder"], [0.1, 0.5])


def test_service_requires_both_thresholds(transactions_factory):
    preprocessor = FraudPreprocessor().fit(transactions_factory())

    with pytest.raises(ValueError, match="Faltan umbrales"):
        FraudDetectionService(preprocessor, ScoreModel(), OffsetAutoencoder(), {"mlp": 0.5})


@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")
def test_service_loads_training_artifacts(
    tmp_path,
    monkeypatch,
    transactions_factory,
):
    model_dir = tmp_path / "models"
    result_dir = tmp_path / "results"
    paths = ArtifactPaths(
        mlp_model=model_dir / "mlp.keras",
        autoencoder_model=model_dir / "autoencoder.keras",
        preprocessor=model_dir / "preprocessor.joblib",
        thresholds=model_dir / "thresholds.json",
        metrics=result_dir / "metrics.csv",
        provenance=result_dir / "provenance.json",
    )
    model_dir.mkdir()
    preprocessor = FraudPreprocessor().fit(transactions_factory())
    joblib.dump(preprocessor, paths.preprocessor)
    paths.thresholds.write_text(
        json.dumps({"mlp": 0.5, "autoencoder": 0.3}),
        encoding="utf-8",
    )
    paths.mlp_model.touch()
    paths.autoencoder_model.touch()
    models = iter([ScoreModel(), OffsetAutoencoder()])
    fake_tensorflow = SimpleNamespace(
        keras=SimpleNamespace(models=SimpleNamespace(load_model=lambda path: next(models)))
    )
    monkeypatch.setitem(__import__("sys").modules, "tensorflow", fake_tensorflow)

    service = FraudDetectionService.from_artifacts(paths)

    assert isinstance(service.preprocessor, FraudPreprocessor)
    assert service.thresholds == {"mlp": 0.5, "autoencoder": 0.3}
