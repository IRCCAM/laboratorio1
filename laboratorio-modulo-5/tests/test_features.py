import numpy as np
import pandas as pd

from laboratorio1.features import FraudPreprocessor


def create_test_dataframe(n_rows=10):
    return pd.DataFrame(
        {
            **{f"V{i}": np.random.randn(n_rows) for i in range(1, 29)},
            "Time": np.arange(n_rows) * 3600,
            "Amount": np.linspace(10, 100, n_rows),
            "Class": np.arange(n_rows) % 2,
        }
    )


def test_preprocessor_generates_32_features():
    df = create_test_dataframe()

    preprocessor = FraudPreprocessor()
    X = preprocessor.fit_transform(df)

    assert X.shape == (10, 32)

def test_feature_engineering_calculates_new_variables():
    df = create_test_dataframe()

    preprocessor = FraudPreprocessor()
    features = preprocessor.create_features(df)

    # Verificar LogAmount
    expected_log_amount = np.log1p(df["Amount"].clip(lower=0))

    assert np.allclose(
        features["LogAmount"],
        expected_log_amount
    )

    # Verificar HourSin y HourCos
    expected_hour = (df["Time"] % 86400) / 3600.0

    expected_hour_sin = np.sin(
        2 * np.pi * expected_hour / 24.0
    )

    expected_hour_cos = np.cos(
        2 * np.pi * expected_hour / 24.0
    )

    assert np.allclose(
        features["HourSin"],
        expected_hour_sin
    )

    assert np.allclose(
        features["HourCos"],
        expected_hour_cos
    )

def test_preprocessor_returns_float32():
    df = create_test_dataframe()

    preprocessor = FraudPreprocessor()
    X = preprocessor.fit_transform(df)

    assert X.dtype == np.float32

def test_process_splits_returns_correct_shapes():
    train_df = create_test_dataframe(10)
    val_df = create_test_dataframe(5)
    test_df = create_test_dataframe(5)

    preprocessor = FraudPreprocessor()

    X_train, X_val, X_test = preprocessor.process_splits(
        train_df,
        val_df,
        test_df,
    )

    assert X_train.shape == (10, 32)
    assert X_val.shape == (5, 32)
    assert X_test.shape == (5, 32)

    assert X_train.dtype == np.float32
    assert X_val.dtype == np.float32
    assert X_test.dtype == np.float32