"""Promoción reproducible de la mejor MLP al Model Registry de MLflow."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient
import typer

from laboratorio1.config import ARTIFACT_PATHS
from laboratorio1.pipeline import PARAMS_PATH, PipelineParameters
from laboratorio1.tracking import file_sha256

REGISTERED_MLP_NAME = "deteccion_fraude_mlp"
REGISTERED_MLP_ALIAS = "champion"
MLP_MODEL_TYPE = "cost_sensitive_mlp"
SELECTION_METRIC = "f2"


def select_best_mlp_run(runs: Iterable[Any]) -> Any:
    """Selecciona la corrida MLP finalizada con mayor F2 y PR-AUC."""
    candidates = [
        run
        for run in runs
        if run.info.status == "FINISHED"
        and run.data.tags.get("model.type") == MLP_MODEL_TYPE
        and SELECTION_METRIC in run.data.metrics
    ]
    if not candidates:
        raise RuntimeError("No existe una corrida MLP finalizada con métrica F2")
    return max(
        candidates,
        key=lambda run: (
            float(run.data.metrics[SELECTION_METRIC]),
            float(run.data.metrics.get("pr_auc", float("-inf"))),
            int(run.info.start_time or 0),
        ),
    )


def select_logged_model(logged_models: Iterable[Any], run_id: str) -> Any:
    """Obtiene el modelo listo asociado a la corrida seleccionada."""
    candidates = [
        model
        for model in logged_models
        if model.source_run_id == run_id
        and getattr(model.status, "value", model.status) == "READY"
    ]
    if not candidates:
        raise RuntimeError(f"La corrida MLP {run_id} no contiene un modelo listo")
    return max(candidates, key=lambda model: int(model.creation_timestamp or 0))


def find_registered_version(versions: Iterable[Any], model_sha256: str) -> Any | None:
    """Busca una versión ya creada a partir del mismo archivo Keras."""
    return next(
        (version for version in versions if version.tags.get("model.sha256") == model_sha256),
        None,
    )


def ensure_registered_version(
    client: MlflowClient,
    *,
    model_uri: str,
    source_run_id: str,
    model_sha256: str,
    registered_model_name: str,
) -> tuple[Any, bool]:
    """Registra una versión solo cuando el artefacto aún no está promovido."""
    versions = client.search_model_versions(f"name = '{registered_model_name}'")
    existing = find_registered_version(versions, model_sha256)
    if existing is not None:
        return existing, False

    version = mlflow.register_model(
        model_uri,
        registered_model_name,
        await_registration_for=300,
        tags={
            "model.sha256": model_sha256,
            "model.role": "winner",
            "selection.metric": SELECTION_METRIC,
            "source.run_id": source_run_id,
        },
    )
    return version, True


def register_best_mlp(
    params_path: Path = PARAMS_PATH,
    model_path: Path = ARTIFACT_PATHS.mlp_model,
    registered_model_name: str = REGISTERED_MLP_NAME,
    alias: str = REGISTERED_MLP_ALIAS,
) -> Any:
    """Promueve la mejor corrida MLP y le asigna el alias operativo."""
    parameters = PipelineParameters.load(params_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"No existe el modelo MLP local; ejecute dvc pull: {model_path}")

    mlflow.set_tracking_uri(parameters.mlflow.tracking_uri)
    client = MlflowClient(tracking_uri=parameters.mlflow.tracking_uri)
    experiment = client.get_experiment_by_name(parameters.mlflow.experiment_name)
    if experiment is None:
        raise RuntimeError(f"No existe el experimento MLflow: {parameters.mlflow.experiment_name}")

    runs = client.search_runs([experiment.experiment_id])
    best_run = select_best_mlp_run(runs)
    logged_models = client.search_logged_models([experiment.experiment_id])
    logged_model = select_logged_model(logged_models, best_run.info.run_id)
    model_sha256 = file_sha256(model_path)
    version, created = ensure_registered_version(
        client,
        model_uri=logged_model.model_uri,
        source_run_id=best_run.info.run_id,
        model_sha256=model_sha256,
        registered_model_name=registered_model_name,
    )

    client.set_registered_model_tag(registered_model_name, "project", "deteccion_fraude")
    client.set_registered_model_alias(registered_model_name, alias, version.version)
    action = "creada" if created else "reutilizada"
    logger.success(
        f"Versión {version.version} {action}: {registered_model_name}@{alias} "
        f"(run {best_run.info.run_id})"
    )
    return version


def main(
    params_path: Path = PARAMS_PATH,
    model_path: Path = ARTIFACT_PATHS.mlp_model,
) -> None:
    """Registra la MLP ganadora sin volver a entrenar los modelos."""
    register_best_mlp(params_path=params_path, model_path=model_path)


if __name__ == "__main__":
    typer.run(main)
