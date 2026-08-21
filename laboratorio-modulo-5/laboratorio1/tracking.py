"""Registro MLflow de los modelos producidos por el pipeline DVC."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from loguru import logger
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import tensorflow as tf
import typer

from laboratorio1.config import ARTIFACT_PATHS
from laboratorio1.pipeline import (
    METRICS_JSON_PATH,
    PARAMS_PATH,
    PREPARED_DATA_PATH,
    PipelineParameters,
    PreparedDataset,
)

MLFLOW_ARTIFACTS_DIR = Path("mlartifacts")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRIC_NAMES = {
    "Umbral": "threshold",
    "Accuracy": "accuracy",
    "Balanced Accuracy": "balanced_accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "Especificidad": "specificity",
    "F1": "f1",
    "F2": "f2",
    "ROC-AUC": "roc_auc",
    "PR-AUC": "pr_auc",
    "MCC": "mcc",
    "TP": "true_positives",
    "FP": "false_positives",
    "FN": "false_negatives",
    "TN": "true_negatives",
    "Falsas alertas / 10.000 legítimas": "false_alerts_per_10k_legit",
    "Monto de fraude capturado (%)": "captured_fraud_amount_pct",
}


def normalized_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Convierte las métricas del reporte a nombres estables aceptados por MLflow."""
    normalized: dict[str, float] = {}
    for source_name, target_name in METRIC_NAMES.items():
        value = metrics.get(source_name)
        if isinstance(value, (int, float)) and np.isfinite(value):
            normalized[target_name] = float(value)
    return normalized


def file_sha256(path: str | Path) -> str:
    """Calcula un identificador de contenido para archivos de linaje."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_lineage(project_root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Obtiene commit y estado del árbol sin hacer fallar el tracking fuera de Git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git.commit": "unknown", "git.dirty": "unknown"}
    return {"git.commit": commit, "git.dirty": str(dirty).lower()}


class FraudMLflowTracker:
    """Registra en MLflow los dos candidatos generados y evaluados por DVC."""

    def __init__(self, parameters: PipelineParameters) -> None:
        self.parameters = parameters
        mlflow.set_tracking_uri(parameters.mlflow.tracking_uri)
        client = MlflowClient(tracking_uri=parameters.mlflow.tracking_uri)
        experiment = client.get_experiment_by_name(parameters.mlflow.experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(
                parameters.mlflow.experiment_name,
                artifact_location=MLFLOW_ARTIFACTS_DIR.resolve().as_uri(),
            )
        else:
            experiment_id = experiment.experiment_id
        self.experiment_id = experiment_id

    def log_models(
        self,
        prepared: PreparedDataset,
        metrics: dict[str, dict[str, Any]],
        mlp_path: Path,
        autoencoder_path: Path,
        artifact_paths: list[Path],
    ) -> list[str]:
        """Crea un run completo para cada modelo candidato."""
        mlp = tf.keras.models.load_model(mlp_path)
        autoencoder = tf.keras.models.load_model(autoencoder_path)
        common_tags = {
            "project": "deteccion_fraude",
            "framework": "tensorflow-keras",
            "orchestrator": "dvc",
            "dataset.sha256": prepared.dataset_sha256,
            "dataset.rows": str(prepared.modeled_rows),
            "dvc.lock.sha256": file_sha256(PROJECT_ROOT / "dvc.lock"),
            **git_lineage(),
        }
        run_ids = [
            self._log_model(
                model_key="mlp",
                model=mlp,
                model_params=asdict(self.parameters.mlp),
                metrics=metrics["mlp"],
                prepared=prepared,
                tags={**common_tags, "model.type": "cost_sensitive_mlp"},
                artifact_paths=artifact_paths,
            ),
            self._log_model(
                model_key="autoencoder",
                model=autoencoder,
                model_params=asdict(self.parameters.autoencoder),
                metrics=metrics["autoencoder"],
                prepared=prepared,
                tags={**common_tags, "model.type": "denoising_autoencoder"},
                artifact_paths=artifact_paths,
            ),
        ]
        return run_ids

    def _log_model(
        self,
        *,
        model_key: str,
        model: tf.keras.Model,
        model_params: dict[str, Any],
        metrics: dict[str, Any],
        prepared: PreparedDataset,
        tags: dict[str, str],
        artifact_paths: list[Path],
    ) -> str:
        run_name = f"{model_key}-{prepared.dataset_sha256[:8]}"
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name) as run:
            mlflow.set_tags(tags)
            mlflow.log_params(
                {
                    **{
                        f"split.{key}": value
                        for key, value in asdict(self.parameters.split).items()
                    },
                    **{f"model.{key}": value for key, value in model_params.items()},
                }
            )
            mlflow.log_metrics(normalized_metrics(metrics))
            self._log_datasets(prepared)
            for artifact in artifact_paths:
                mlflow.log_artifact(artifact, artifact_path="pipeline")
            if self.parameters.mlflow.log_models:
                input_example = prepared.test_features[:5]
                predictions = model.predict(input_example, verbose=0)
                signature = infer_signature(input_example, predictions)
                mlflow.tensorflow.log_model(
                    model=model,
                    name="model",
                    signature=signature,
                    input_example=input_example,
                )
            logger.success(f"Run MLflow registrado: {model_key} ({run.info.run_id})")
            return run.info.run_id

    @staticmethod
    def _log_datasets(prepared: PreparedDataset) -> None:
        source = "data/raw/creditcard.csv"
        datasets = (
            (prepared.train_features, "training"),
            (prepared.validation_features, "validation"),
            (prepared.test_features, "testing"),
        )
        for values, context in datasets:
            dataset = mlflow.data.from_numpy(
                values,
                source=source,
                name=f"fraud_{context}",
            )
            mlflow.log_input(dataset, context=context)


def main(
    prepared_path: Path = PREPARED_DATA_PATH,
    mlp_path: Path = ARTIFACT_PATHS.mlp_model,
    autoencoder_path: Path = ARTIFACT_PATHS.autoencoder_model,
    metrics_path: Path = METRICS_JSON_PATH,
    params_path: Path = PARAMS_PATH,
) -> None:
    """Registra modelos y resultados; se ejecuta después de la evaluación DVC."""
    parameters = PipelineParameters.load(params_path)
    if not parameters.mlflow.enabled:
        logger.info("Tracking MLflow deshabilitado mediante params.yaml")
        return
    prepared = PreparedDataset.load(prepared_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    required_models = {"mlp", "autoencoder"}
    if not isinstance(metrics, dict) or required_models - set(metrics):
        raise ValueError("El reporte de métricas no contiene ambos modelos")
    artifacts = [
        params_path,
        PROJECT_ROOT / "dvc.yaml",
        PROJECT_ROOT / "dvc.lock",
        metrics_path,
        ARTIFACT_PATHS.metrics,
        ARTIFACT_PATHS.thresholds,
        ARTIFACT_PATHS.provenance,
    ]
    missing = [path for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Faltan artefactos del pipeline: {missing}")
    tracker = FraudMLflowTracker(parameters)
    run_ids = tracker.log_models(
        prepared,
        metrics,
        mlp_path,
        autoencoder_path,
        artifacts,
    )
    logger.success(f"Tracking MLflow completo: {', '.join(run_ids)}")


if __name__ == "__main__":
    typer.run(main)
