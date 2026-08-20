"""Servicio y CLI para inferencia de fraude."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import joblib
from loguru import logger
import numpy as np
import pandas as pd
import typer

from laboratorio1.config import ARTIFACT_PATHS, PROCESSED_DATA_DIR, ArtifactPaths
from laboratorio1.features import FraudPreprocessor
from laboratorio1.modeling.models import reconstruction_error

app = typer.Typer()


class PredictionModel(Protocol):
    """Interfaz mínima compartida por modelos Keras y dobles de prueba."""

    def predict(self, features: np.ndarray, **kwargs: object) -> np.ndarray: ...


class FraudDetectionService:
    """Aplica el preprocesamiento y ambos detectores con umbrales persistidos."""

    def __init__(
        self,
        preprocessor: FraudPreprocessor,
        mlp_model: PredictionModel,
        autoencoder_model: PredictionModel,
        thresholds: dict[str, float],
    ) -> None:
        missing = {"mlp", "autoencoder"} - set(thresholds)
        if missing:
            raise ValueError(f"Faltan umbrales requeridos: {sorted(missing)}")
        if not all(np.isfinite(thresholds[key]) for key in ("mlp", "autoencoder")):
            raise ValueError("Los umbrales deben ser finitos")
        self.preprocessor = preprocessor
        self.mlp_model = mlp_model
        self.autoencoder_model = autoencoder_model
        self.thresholds = {
            "mlp": float(thresholds["mlp"]),
            "autoencoder": float(thresholds["autoencoder"]),
        }

    @classmethod
    def from_artifacts(
        cls,
        artifact_paths: ArtifactPaths = ARTIFACT_PATHS,
    ) -> FraudDetectionService:
        """Carga los artefactos generados por ``FraudTrainingPipeline``."""
        import tensorflow as tf  # Importación diferida para facilitar pruebas unitarias.

        preprocessor = joblib.load(artifact_paths.preprocessor)
        if not isinstance(preprocessor, FraudPreprocessor):
            raise TypeError("El artefacto no contiene un FraudPreprocessor")
        thresholds = json.loads(artifact_paths.thresholds.read_text(encoding="utf-8"))
        return cls(
            preprocessor=preprocessor,
            mlp_model=tf.keras.models.load_model(artifact_paths.mlp_model),
            autoencoder_model=tf.keras.models.load_model(artifact_paths.autoencoder_model),
            thresholds=thresholds,
        )

    def predict(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Devuelve probabilidades, errores de reconstrucción y alertas."""
        features = self.preprocessor.transform(transactions)
        mlp_scores = np.asarray(self.mlp_model.predict(features, verbose=0)).ravel()
        reconstructed = np.asarray(
            self.autoencoder_model.predict(features, verbose=0),
            dtype=np.float32,
        )
        autoencoder_scores = reconstruction_error(features, reconstructed)
        if mlp_scores.shape != (len(transactions),):
            raise ValueError("La MLP devolvió una cantidad inesperada de predicciones")
        return pd.DataFrame(
            {
                "probabilidad_mlp": mlp_scores,
                "alerta_mlp": (mlp_scores >= self.thresholds["mlp"]).astype(np.int32),
                "error_autoencoder": autoencoder_scores,
                "alerta_autoencoder": (
                    autoencoder_scores >= self.thresholds["autoencoder"]
                ).astype(np.int32),
            },
            index=transactions.index,
        )


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "transacciones.csv",
    output_path: Path = PROCESSED_DATA_DIR / "predicciones.csv",
) -> None:
    """Ejecuta inferencia desde CSV usando los artefactos predeterminados."""
    transactions = pd.read_csv(input_path)
    predictions = FraudDetectionService.from_artifacts().predict(transactions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    logger.success(f"Se escribieron {len(predictions):,} predicciones en {output_path}")


if __name__ == "__main__":
    app()
