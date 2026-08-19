import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


class FraudPreprocessor:
    FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + [
        "Time",
        "LogAmount",
        "HourSin",
        "HourCos",
    ]

    SCALE_COLUMNS = ["Time", "LogAmount"]

    def __init__(self):
        self.scaler = RobustScaler()

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        data["LogAmount"] = np.log1p(
            data["Amount"].clip(lower=0)
        )

        data["Hour"] = (
            data["Time"] % 86400
        ) / 3600.0

        data["HourSin"] = np.sin(
            2 * np.pi * data["Hour"] / 24.0
        )

        data["HourCos"] = np.cos(
            2 * np.pi * data["Hour"] / 24.0
        )

        return data

    def fit(self, df: pd.DataFrame):
        data = self.create_features(df)

        self.scaler.fit(data[self.SCALE_COLUMNS])

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        data = self.create_features(df)

        data.loc[:, self.SCALE_COLUMNS] = self.scaler.transform(
            data[self.SCALE_COLUMNS]
        )

        return data[self.FEATURE_NAMES].to_numpy(dtype="float32")

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)

        return self.transform(df)