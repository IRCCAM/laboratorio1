import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def transactions_factory():
    """Fabrica DataFrames sinteticos con el esquema de creditcard.csv para tests."""

    def _make(n: int = 100, *, fraud_ratio: float = 0.2, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)

        n_fraud = max(2, round(n * fraud_ratio))
        n_fraud = min(n_fraud, n - 2)
        n_legit = n - n_fraud

        labels = np.array([0] * n_legit + [1] * n_fraud)
        rng.shuffle(labels)

        data = {f"V{i}": rng.normal(size=n) for i in range(1, 29)}
        data["Time"] = np.sort(rng.uniform(0, 172_792, size=n))
        data["Amount"] = np.round(rng.uniform(0, 5000, size=n), 2)
        data["Class"] = labels

        return pd.DataFrame(data)

    return _make
