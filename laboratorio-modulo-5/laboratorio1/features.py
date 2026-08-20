"""Ingeniería de variables reproducible y sin fuga de datos."""

from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import RobustScaler


class FraudPreprocessor:
    """Genera las 32 variables y escala usando solo datos de entrenamiento."""

    RAW_FEATURE_NAMES: ClassVar[tuple[str, ...]] = (
        "Time",
        "Amount",
        *(f"V{i}" for i in range(1, 29)),
    )
    FEATURE_NAMES: ClassVar[tuple[str, ...]] = (
        *(f"V{i}" for i in range(1, 29)),
        "Time",
        "LogAmount",
        "HourSin",
        "HourCos",
    )
    SCALE_COLUMNS: ClassVar[tuple[str, ...]] = ("Time", "LogAmount")

    def __init__(self) -> None:
        self.scaler = RobustScaler()
        self._is_fitted = False

    def _validate_input(self, dataframe: pd.DataFrame) -> None:
        missing = sorted(set(self.RAW_FEATURE_NAMES) - set(dataframe.columns))
        if missing:
            raise ValueError(f"Faltan variables de entrada: {missing}")
        if dataframe.empty:
            raise ValueError("Las transacciones no pueden estar vacías")
        values = dataframe.loc[:, self.RAW_FEATURE_NAMES]
        if values.isna().any().any():
            raise ValueError("Las variables de entrada contienen valores faltantes")
        try:
            finite = np.isfinite(values.to_numpy(dtype=float)).all()
        except (TypeError, ValueError) as error:
            raise ValueError("Las variables de entrada deben ser numéricas") from error
        if not finite:
            raise ValueError("Las variables de entrada contienen valores no finitos")

    def create_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Crea LogAmount y la representación cíclica de la hora."""
        self._validate_input(dataframe)
        data = dataframe.copy()
        data["LogAmount"] = np.log1p(data["Amount"].clip(lower=0))
        hour = (data["Time"] % 86_400) / 3_600.0
        data["HourSin"] = np.sin(2 * np.pi * hour / 24.0)
        data["HourCos"] = np.cos(2 * np.pi * hour / 24.0)
        return data

    def fit(self, dataframe: pd.DataFrame) -> "FraudPreprocessor":
        """Ajusta el scaler con la partición de entrenamiento."""
        data = self.create_features(dataframe)
        self.scaler.fit(data.loc[:, self.SCALE_COLUMNS])
        self._is_fitted = True
        return self

    def transform(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Transforma transacciones conservando orden y tipo de las variables."""
        if not self._is_fitted:
            raise NotFittedError("FraudPreprocessor debe ajustarse antes de transformar")
        data = self.create_features(dataframe)
        data.loc[:, self.SCALE_COLUMNS] = self.scaler.transform(data.loc[:, self.SCALE_COLUMNS])
        return data.loc[:, self.FEATURE_NAMES].to_numpy(dtype=np.float32)

    def fit_transform(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Ajusta con entrenamiento y devuelve sus variables transformadas."""
        return self.fit(dataframe).transform(dataframe)

    def process_splits(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ajusta solo con train y transforma las tres particiones."""
        train = self.fit_transform(train_df)
        validation = self.transform(validation_df)
        test = self.transform(test_df)
        return train, validation, test
