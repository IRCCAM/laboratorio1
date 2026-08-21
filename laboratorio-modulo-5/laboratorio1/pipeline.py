"""Etapas reproducibles del pipeline DVC de detección de fraude."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import sys
from typing import Any

import joblib
from loguru import logger
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import typer
import yaml

from laboratorio1.config import (
    ARTIFACT_PATHS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    AutoencoderConfig,
    MLPConfig,
    SplitConfig,
)
from laboratorio1.dataset import FraudDatasetLoader, create_stratified_indices
from laboratorio1.evaluation import F2ThresholdSelector, ModelEvaluator
from laboratorio1.features import FraudPreprocessor
from laboratorio1.modeling.models import reconstruction_error
from laboratorio1.modeling.train import FraudTrainingPipeline

app = typer.Typer(help="Ejecuta de forma independiente las etapas declaradas en dvc.yaml.")

PARAMS_PATH = Path("params.yaml")
PREPARED_DATA_PATH = PROCESSED_DATA_DIR / "fraud_splits.npz"
METRICS_JSON_PATH = ARTIFACT_PATHS.metrics.with_suffix(".json")


@dataclass(frozen=True)
class PipelineParameters:
    """Configuración validada que DVC rastrea desde ``params.yaml``."""

    split: SplitConfig
    mlp: MLPConfig
    autoencoder: AutoencoderConfig

    @classmethod
    def load(cls, path: str | Path = PARAMS_PATH) -> PipelineParameters:
        """Carga los tres grupos de parámetros y rechaza configuraciones inválidas."""
        params_path = Path(path)
        raw = yaml.safe_load(params_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("params.yaml debe contener un objeto YAML")
        missing = {"split", "mlp", "autoencoder"} - set(raw)
        if missing:
            raise ValueError(f"Faltan secciones en params.yaml: {sorted(missing)}")
        try:
            parameters = cls(
                split=SplitConfig(**raw["split"]),
                mlp=MLPConfig(**raw["mlp"]),
                autoencoder=AutoencoderConfig(**raw["autoencoder"]),
            )
        except (TypeError, AttributeError) as error:
            raise ValueError(
                "params.yaml contiene parámetros desconocidos o mal formados"
            ) from error
        parameters.validate()
        return parameters

    def validate(self) -> None:
        """Valida proporciones y valores de entrenamiento antes de iniciar trabajo costoso."""
        split = self.split
        if (
            split.validation_size <= 0
            or split.test_size <= 0
            or split.validation_size + split.test_size >= 1
        ):
            raise ValueError("Las proporciones de validación y test no son válidas")
        if split.seed < 0:
            raise ValueError("split.seed no puede ser negativo")
        for name, config in (("mlp", self.mlp), ("autoencoder", self.autoencoder)):
            if config.learning_rate <= 0 or config.epochs <= 0 or config.batch_size <= 0:
                raise ValueError(f"Los parámetros principales de {name} deben ser positivos")
            if config.prediction_batch_size <= 0:
                raise ValueError(f"{name}.prediction_batch_size debe ser positivo")


@dataclass(frozen=True)
class PreparedDataset:
    """Particiones transformadas que conectan las etapas del grafo DVC."""

    train_features: np.ndarray
    validation_features: np.ndarray
    test_features: np.ndarray
    train_labels: np.ndarray
    validation_labels: np.ndarray
    test_labels: np.ndarray
    test_amounts: np.ndarray
    dataset_sha256: str
    modeled_rows: int

    def save(self, path: str | Path) -> None:
        """Persiste las particiones en un único artefacto comprimido."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            train_features=self.train_features,
            validation_features=self.validation_features,
            test_features=self.test_features,
            train_labels=self.train_labels,
            validation_labels=self.validation_labels,
            test_labels=self.test_labels,
            test_amounts=self.test_amounts,
            dataset_sha256=np.array(self.dataset_sha256),
            modeled_rows=np.array(self.modeled_rows),
        )

    @classmethod
    def load(cls, path: str | Path) -> PreparedDataset:
        """Carga un artefacto producido por la etapa ``prepare``."""
        with np.load(Path(path), allow_pickle=False) as data:
            required = {
                "train_features",
                "validation_features",
                "test_features",
                "train_labels",
                "validation_labels",
                "test_labels",
                "test_amounts",
                "dataset_sha256",
                "modeled_rows",
            }
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"Faltan arreglos en el dataset preparado: {sorted(missing)}")
            return cls(
                train_features=data["train_features"],
                validation_features=data["validation_features"],
                test_features=data["test_features"],
                train_labels=data["train_labels"],
                validation_labels=data["validation_labels"],
                test_labels=data["test_labels"],
                test_amounts=data["test_amounts"],
                dataset_sha256=str(data["dataset_sha256"].item()),
                modeled_rows=int(data["modeled_rows"].item()),
            )


def _set_reproducibility(seed: int) -> None:
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        logger.warning("TensorFlow no pudo activar operaciones deterministas")


@app.command("prepare")
def prepare(
    dataset_path: Path = RAW_DATA_DIR / "creditcard.csv",
    output_path: Path = PREPARED_DATA_PATH,
    preprocessor_path: Path = ARTIFACT_PATHS.preprocessor,
    params_path: Path = PARAMS_PATH,
) -> None:
    """Valida, divide y transforma el dataset sin fuga entre particiones."""
    parameters = PipelineParameters.load(params_path)
    loader = FraudDatasetLoader()
    dataframe = loader.load(dataset_path, drop_duplicates=True)
    labels = dataframe["Class"].to_numpy(dtype=np.int32)
    indices = create_stratified_indices(
        labels,
        validation_size=parameters.split.validation_size,
        test_size=parameters.split.test_size,
        seed=parameters.split.seed,
    )
    preprocessor = FraudPreprocessor()
    train, validation, test = preprocessor.process_splits(
        dataframe.iloc[indices.train],
        dataframe.iloc[indices.validation],
        dataframe.iloc[indices.test],
    )
    prepared = PreparedDataset(
        train_features=train,
        validation_features=validation,
        test_features=test,
        train_labels=labels[indices.train],
        validation_labels=labels[indices.validation],
        test_labels=labels[indices.test],
        test_amounts=dataframe.iloc[indices.test]["Amount"].to_numpy(dtype=float),
        dataset_sha256=loader.sha256(dataset_path),
        modeled_rows=len(dataframe),
    )
    prepared.save(output_path)
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)
    logger.success(
        f"Datos preparados: train={len(indices.train):,}, "
        f"validation={len(indices.validation):,}, test={len(indices.test):,}"
    )


@app.command("train-mlp")
def train_mlp(
    prepared_path: Path = PREPARED_DATA_PATH,
    model_path: Path = ARTIFACT_PATHS.mlp_model,
    params_path: Path = PARAMS_PATH,
) -> None:
    """Entrena exclusivamente la MLP supervisada."""
    parameters = PipelineParameters.load(params_path)
    _set_reproducibility(parameters.split.seed)
    data = PreparedDataset.load(prepared_path)
    trainer = FraudTrainingPipeline(
        split_config=parameters.split,
        mlp_config=parameters.mlp,
        autoencoder_config=parameters.autoencoder,
    )
    model = trainer.train_mlp(
        data.train_features,
        data.train_labels,
        data.validation_features,
        data.validation_labels,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    logger.success(f"MLP guardada en {model_path}")


@app.command("train-autoencoder")
def train_autoencoder(
    prepared_path: Path = PREPARED_DATA_PATH,
    model_path: Path = ARTIFACT_PATHS.autoencoder_model,
    params_path: Path = PARAMS_PATH,
) -> None:
    """Entrena exclusivamente el autoencoder con transacciones legítimas."""
    parameters = PipelineParameters.load(params_path)
    _set_reproducibility(parameters.split.seed)
    data = PreparedDataset.load(prepared_path)
    trainer = FraudTrainingPipeline(
        split_config=parameters.split,
        mlp_config=parameters.mlp,
        autoencoder_config=parameters.autoencoder,
    )
    model = trainer.train_autoencoder(
        data.train_features,
        data.train_labels,
        data.validation_features,
        data.validation_labels,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    logger.success(f"Autoencoder guardado en {model_path}")


def _evaluate_model(
    name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    test_amounts: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    selector = F2ThresholdSelector()
    evaluator = ModelEvaluator()
    threshold = selector.select(labels, scores).threshold
    metrics = evaluator.evaluate(
        name,
        test_labels,
        test_scores,
        threshold,
        test_amounts,
    )
    return threshold, metrics.as_dict()


@app.command("evaluate")
def evaluate(
    prepared_path: Path = PREPARED_DATA_PATH,
    mlp_path: Path = ARTIFACT_PATHS.mlp_model,
    autoencoder_path: Path = ARTIFACT_PATHS.autoencoder_model,
    thresholds_path: Path = ARTIFACT_PATHS.thresholds,
    metrics_path: Path = METRICS_JSON_PATH,
    provenance_path: Path = ARTIFACT_PATHS.provenance,
    params_path: Path = PARAMS_PATH,
) -> None:
    """Selecciona umbrales en validación y calcula métricas finales en test."""
    parameters = PipelineParameters.load(params_path)
    data = PreparedDataset.load(prepared_path)
    mlp = tf.keras.models.load_model(mlp_path)
    autoencoder = tf.keras.models.load_model(autoencoder_path)

    mlp_validation = mlp.predict(
        data.validation_features,
        batch_size=parameters.mlp.prediction_batch_size,
        verbose=0,
    ).ravel()
    mlp_test = mlp.predict(
        data.test_features,
        batch_size=parameters.mlp.prediction_batch_size,
        verbose=0,
    ).ravel()
    autoencoder_validation = reconstruction_error(
        data.validation_features,
        autoencoder.predict(
            data.validation_features,
            batch_size=parameters.autoencoder.prediction_batch_size,
            verbose=0,
        ),
    )
    autoencoder_test = reconstruction_error(
        data.test_features,
        autoencoder.predict(
            data.test_features,
            batch_size=parameters.autoencoder.prediction_batch_size,
            verbose=0,
        ),
    )
    mlp_threshold, mlp_metrics = _evaluate_model(
        "MLP supervisada sensible al costo",
        data.validation_labels,
        mlp_validation,
        data.test_labels,
        mlp_test,
        data.test_amounts,
    )
    autoencoder_threshold, autoencoder_metrics = _evaluate_model(
        "Autoencoder de anomalías",
        data.validation_labels,
        autoencoder_validation,
        data.test_labels,
        autoencoder_test,
        data.test_amounts,
    )
    thresholds = {
        "criterio": "Máximo F2 exclusivamente en validación",
        "mlp": mlp_threshold,
        "autoencoder": autoencoder_threshold,
    }
    metric_rows = [mlp_metrics, autoencoder_metrics]
    metrics_for_dvc = {
        "mlp": _numeric_metrics(mlp_metrics),
        "autoencoder": _numeric_metrics(autoencoder_metrics),
    }
    for path in (thresholds_path, metrics_path, provenance_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    thresholds_path.write_text(
        json.dumps(thresholds, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(metrics_for_dvc, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    ARTIFACT_PATHS.metrics.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(ARTIFACT_PATHS.metrics, index=False)
    provenance = {
        "dataset": {
            "archivo": "creditcard.csv",
            "sha256": data.dataset_sha256,
            "filas_modelado": data.modeled_rows,
        },
        "entorno": {
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "plataforma": platform.platform(),
        },
        "particion": asdict(parameters.split),
        "mlp": asdict(parameters.mlp),
        "autoencoder": asdict(parameters.autoencoder),
        "seleccion_umbral": "Máximo F2 exclusivamente en validación",
    }
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.success(f"Evaluación terminada; métricas DVC en {metrics_path}")


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, int | float]:
    """Extrae métricas numéricas para que ``dvc metrics`` pueda compararlas."""
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and value is not None and np.isfinite(value)
    }


if __name__ == "__main__":
    app()
