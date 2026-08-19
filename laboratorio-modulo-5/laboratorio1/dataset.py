from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetIndices:
    """Indices de las particiones; permite conservar filas y montos originales."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


class FraudDatasetLoader:
    """Carga y valida el esquema del dataset creditcard.csv."""

    EXPECTED_COLUMNS: ClassVar[set[str]] = {
        "Time",
        "Amount",
        "Class",
        *[f"V{i}" for i in range(1, 29)],
    }

    def load(self, path: str | Path) -> pd.DataFrame:
        dataset_path = Path(path)
        if not dataset_path.is_file():
            raise FileNotFoundError(f"No existe el dataset: {dataset_path}")

        dataframe = pd.read_csv(dataset_path)
        self.validate(dataframe)
        return dataframe

    def validate(self, dataframe: pd.DataFrame) -> None:
        missing = sorted(self.EXPECTED_COLUMNS - set(dataframe.columns))
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")
        if dataframe[list(self.EXPECTED_COLUMNS)].isna().any().any():
            raise ValueError("El dataset contiene valores faltantes")

        classes = set(dataframe["Class"].unique())
        if classes != {0, 1}:
            raise ValueError(f"Class debe contener 0 y 1; contiene {sorted(classes)}")

    @staticmethod
    def sha256(path: str | Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_stratified_indices(
    y: np.ndarray,
    *,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> DatasetIndices:
    """Crea particiones estratificadas sin consultar el conjunto de prueba."""

    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("Las proporciones de validacion y test no son validas")

    indices = np.arange(len(y))
    holdout_size = validation_size + test_size
    train, holdout = train_test_split(
        indices,
        test_size=holdout_size,
        stratify=y,
        random_state=seed,
    )
    relative_test_size = test_size / holdout_size
    validation, test = train_test_split(
        holdout,
        test_size=relative_test_size,
        stratify=y[holdout],
        random_state=seed,
    )
    return DatasetIndices(train=train, validation=validation, test=test)
