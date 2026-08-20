"""Pipeline de entrenamiento reproducible para MLP y autoencoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import sys

import joblib
from loguru import logger
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import typer

from laboratorio1.config import (
    ARTIFACT_PATHS,
    AUTOENCODER_CONFIG,
    MLP_CONFIG,
    RAW_DATA_DIR,
    SPLIT_CONFIG,
    ArtifactPaths,
    AutoencoderConfig,
    MLPConfig,
    SplitConfig,
)
from laboratorio1.dataset import FraudDatasetLoader, create_stratified_indices
from laboratorio1.evaluation import F2ThresholdSelector, ModelEvaluator
from laboratorio1.features import FraudPreprocessor
from laboratorio1.modeling.models import (
    build_autoencoder,
    build_mlp,
    calculate_class_weights,
    calculate_output_bias,
    reconstruction_error,
)

app = typer.Typer()


@dataclass(frozen=True)
class TrainingResult:
    """Resumen de los artefactos y métricas producidos."""

    thresholds: dict[str, float]
    metrics: pd.DataFrame
    train_rows: int
    validation_rows: int
    test_rows: int


class FraudTrainingPipeline:
    """Orquesta preparación, entrenamiento, evaluación y persistencia."""

    def __init__(
        self,
        *,
        split_config: SplitConfig = SPLIT_CONFIG,
        mlp_config: MLPConfig = MLP_CONFIG,
        autoencoder_config: AutoencoderConfig = AUTOENCODER_CONFIG,
        artifact_paths: ArtifactPaths = ARTIFACT_PATHS,
    ) -> None:
        self.split_config = split_config
        self.mlp_config = mlp_config
        self.autoencoder_config = autoencoder_config
        self.artifact_paths = artifact_paths
        self.loader = FraudDatasetLoader()
        self.selector = F2ThresholdSelector()
        self.evaluator = ModelEvaluator()

    def run(self, dataset_path: str | Path) -> TrainingResult:
        """Entrena los modelos y persiste artefactos compatibles con inferencia."""
        path = Path(dataset_path)
        self._set_reproducibility()
        dataframe = self.loader.load(path, drop_duplicates=True)
        labels = dataframe["Class"].to_numpy(dtype=np.int32)
        indices = create_stratified_indices(
            labels,
            validation_size=self.split_config.validation_size,
            test_size=self.split_config.test_size,
            seed=self.split_config.seed,
        )

        preprocessor = FraudPreprocessor()
        train_features, validation_features, test_features = preprocessor.process_splits(
            dataframe.iloc[indices.train],
            dataframe.iloc[indices.validation],
            dataframe.iloc[indices.test],
        )
        train_labels = labels[indices.train]
        validation_labels = labels[indices.validation]
        test_labels = labels[indices.test]
        test_amounts = dataframe.iloc[indices.test]["Amount"].to_numpy(dtype=float)

        mlp = self._train_mlp(
            train_features,
            train_labels,
            validation_features,
            validation_labels,
        )
        autoencoder = self._train_autoencoder(
            train_features,
            train_labels,
            validation_features,
            validation_labels,
        )

        mlp_validation_scores = mlp.predict(
            validation_features,
            batch_size=self.mlp_config.prediction_batch_size,
            verbose=0,
        ).ravel()
        validation_reconstruction = autoencoder.predict(
            validation_features,
            batch_size=self.autoencoder_config.prediction_batch_size,
            verbose=0,
        )
        autoencoder_validation_scores = reconstruction_error(
            validation_features,
            validation_reconstruction,
        )
        mlp_selection = self.selector.select(validation_labels, mlp_validation_scores)
        autoencoder_selection = self.selector.select(
            validation_labels,
            autoencoder_validation_scores,
        )

        mlp_test_scores = mlp.predict(
            test_features,
            batch_size=self.mlp_config.prediction_batch_size,
            verbose=0,
        ).ravel()
        test_reconstruction = autoencoder.predict(
            test_features,
            batch_size=self.autoencoder_config.prediction_batch_size,
            verbose=0,
        )
        autoencoder_test_scores = reconstruction_error(test_features, test_reconstruction)
        metrics = pd.DataFrame(
            [
                self.evaluator.evaluate(
                    "MLP supervisada sensible al costo",
                    test_labels,
                    mlp_test_scores,
                    mlp_selection.threshold,
                    test_amounts,
                ).as_dict(),
                self.evaluator.evaluate(
                    "Autoencoder de anomalías",
                    test_labels,
                    autoencoder_test_scores,
                    autoencoder_selection.threshold,
                    test_amounts,
                ).as_dict(),
            ]
        )
        thresholds = {
            "mlp": mlp_selection.threshold,
            "autoencoder": autoencoder_selection.threshold,
        }
        self._persist(
            path,
            dataframe,
            mlp,
            autoencoder,
            preprocessor,
            thresholds,
            metrics,
        )
        return TrainingResult(
            thresholds=thresholds,
            metrics=metrics,
            train_rows=len(indices.train),
            validation_rows=len(indices.validation),
            test_rows=len(indices.test),
        )

    def _set_reproducibility(self) -> None:
        np.random.seed(self.split_config.seed)
        tf.keras.utils.set_random_seed(self.split_config.seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except RuntimeError:
            logger.warning("TensorFlow no pudo activar operaciones deterministas")

    def _train_mlp(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        validation_features: np.ndarray,
        validation_labels: np.ndarray,
    ) -> tf.keras.Model:
        model = build_mlp(
            train_features.shape[1],
            calculate_output_bias(train_labels),
            self.mlp_config,
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=self.mlp_config.early_stopping_patience,
                min_delta=1e-4,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_pr_auc",
                mode="max",
                factor=0.5,
                patience=self.mlp_config.reduce_lr_patience,
                min_lr=self.mlp_config.min_learning_rate,
            ),
        ]
        model.fit(
            train_features,
            train_labels,
            validation_data=(validation_features, validation_labels),
            epochs=self.mlp_config.epochs,
            batch_size=self.mlp_config.batch_size,
            class_weight=calculate_class_weights(train_labels),
            callbacks=callbacks,
            shuffle=True,
            verbose=2,
        )
        return model

    def _train_autoencoder(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        validation_features: np.ndarray,
        validation_labels: np.ndarray,
    ) -> tf.keras.Model:
        model = build_autoencoder(train_features.shape[1], self.autoencoder_config)
        normal_train = train_features[train_labels == 0]
        normal_validation = validation_features[validation_labels == 0]
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=self.autoencoder_config.early_stopping_patience,
                min_delta=1e-5,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=0.5,
                patience=self.autoencoder_config.reduce_lr_patience,
                min_lr=self.autoencoder_config.min_learning_rate,
            ),
        ]
        model.fit(
            normal_train,
            normal_train,
            validation_data=(normal_validation, normal_validation),
            epochs=self.autoencoder_config.epochs,
            batch_size=self.autoencoder_config.batch_size,
            callbacks=callbacks,
            shuffle=True,
            verbose=2,
        )
        return model

    def _persist(
        self,
        dataset_path: Path,
        dataframe: pd.DataFrame,
        mlp: tf.keras.Model,
        autoencoder: tf.keras.Model,
        preprocessor: FraudPreprocessor,
        thresholds: dict[str, float],
        metrics: pd.DataFrame,
    ) -> None:
        paths = self.artifact_paths
        for path in (
            paths.mlp_model,
            paths.autoencoder_model,
            paths.preprocessor,
            paths.thresholds,
            paths.metrics,
            paths.provenance,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)

        mlp.save(paths.mlp_model)
        autoencoder.save(paths.autoencoder_model)
        joblib.dump(preprocessor, paths.preprocessor)
        paths.thresholds.write_text(
            json.dumps(
                {
                    "criterio": "Máximo F2 exclusivamente en validación",
                    **thresholds,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        metrics.to_csv(paths.metrics, index=False)
        provenance = {
            "dataset": {
                "archivo": dataset_path.name,
                "sha256": self.loader.sha256(dataset_path),
                "filas_modelado": len(dataframe),
            },
            "entorno": {
                "python": sys.version.split()[0],
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "plataforma": platform.platform(),
            },
            "particion": asdict(self.split_config),
            "mlp": asdict(self.mlp_config),
            "autoencoder": asdict(self.autoencoder_config),
            "seleccion_umbral": "Máximo F2 exclusivamente en validación",
        }
        paths.provenance.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


@app.command()
def main(dataset_path: Path = RAW_DATA_DIR / "creditcard.csv") -> None:
    """Entrena y persiste los detectores configurados."""
    result = FraudTrainingPipeline().run(dataset_path)
    logger.success(
        "Entrenamiento completo: "
        f"train={result.train_rows}, validation={result.validation_rows}, "
        f"test={result.test_rows}"
    )


if __name__ == "__main__":
    app()
