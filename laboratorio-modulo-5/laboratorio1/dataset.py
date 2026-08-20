"""Carga, validación, limpieza y partición del dataset de fraude."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetIndices:
    """Índices disjuntos de entrenamiento, validación y prueba."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


class FraudDatasetLoader:
    """Carga y valida el esquema de ``creditcard.csv``."""

    FEATURE_COLUMNS: ClassVar[tuple[str, ...]] = (
        "Time",
        "Amount",
        *(f"V{i}" for i in range(1, 29)),
    )
    EXPECTED_COLUMNS: ClassVar[set[str]] = {*FEATURE_COLUMNS, "Class"}

    def load(self, path: str | Path, *, drop_duplicates: bool = True) -> pd.DataFrame:
        """Lee, valida y opcionalmente elimina duplicados exactos."""
        dataset_path = Path(path)
        if not dataset_path.is_file():
            raise FileNotFoundError(f"No existe el dataset: {dataset_path}")

        dataframe = pd.read_csv(dataset_path)
        self.validate(dataframe)
        return self.clean(dataframe) if drop_duplicates else dataframe

    def validate(self, dataframe: pd.DataFrame) -> None:
        """Valida columnas, valores finitos y etiqueta binaria."""
        missing = sorted(self.EXPECTED_COLUMNS - set(dataframe.columns))
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")
        if dataframe.empty:
            raise ValueError("El dataset no puede estar vacío")

        required = dataframe.loc[:, sorted(self.EXPECTED_COLUMNS)]
        if required.isna().any().any():
            raise ValueError("El dataset contiene valores faltantes")
        try:
            finite = np.isfinite(required.to_numpy(dtype=float)).all()
        except (TypeError, ValueError) as error:
            raise ValueError("Las columnas requeridas deben ser numéricas") from error
        if not finite:
            raise ValueError("El dataset contiene valores no finitos")

        classes = set(dataframe["Class"].unique())
        if classes != {0, 1}:
            raise ValueError(f"Class debe contener 0 y 1; contiene {sorted(classes)}")

    @staticmethod
    def clean(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Elimina duplicados exactos sin modificar el DataFrame recibido."""
        return dataframe.drop_duplicates().reset_index(drop=True)

    @staticmethod
    def sha256(path: str | Path) -> str:
        """Calcula la huella SHA-256 de un archivo."""
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_stratified_indices(
    y: np.ndarray,
    *,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> DatasetIndices:
    """Crea particiones estratificadas sin consultar el conjunto de prueba."""
    labels = np.asarray(y)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("y debe ser un arreglo unidimensional no vacío")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("y debe contener ejemplos de las clases 0 y 1")
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("Las proporciones de validación y test no son válidas")

    indices = np.arange(len(labels))
    holdout_size = validation_size + test_size
    try:
        train, holdout = train_test_split(
            indices,
            test_size=holdout_size,
            stratify=labels,
            random_state=seed,
        )
        validation, test = train_test_split(
            holdout,
            test_size=test_size / holdout_size,
            stratify=labels[holdout],
            random_state=seed,
        )
    except ValueError as error:
        raise ValueError("No hay suficientes ejemplos para estratificar las clases") from error

    return DatasetIndices(train=train, validation=validation, test=test)
