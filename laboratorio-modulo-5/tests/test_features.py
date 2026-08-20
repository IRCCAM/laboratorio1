import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from laboratorio1.features import FraudPreprocessor


def test_preprocessor_generates_ordered_float32_features(transactions_factory):
    dataframe = transactions_factory(10)
    preprocessor = FraudPreprocessor()

    features = preprocessor.fit_transform(dataframe)

    assert features.shape == (10, 32)
    assert features.dtype == np.float32
    assert FraudPreprocessor.FEATURE_NAMES == (
        *(f"V{i}" for i in range(1, 29)),
        "Time",
        "LogAmount",
        "HourSin",
        "HourCos",
    )


def test_feature_engineering_matches_notebook_formulas(transactions_factory):
    dataframe = transactions_factory(10)
    engineered = FraudPreprocessor().create_features(dataframe)
    hour = (dataframe["Time"] % 86_400) / 3_600.0

    assert np.allclose(engineered["LogAmount"], np.log1p(dataframe["Amount"]))
    assert np.allclose(engineered["HourSin"], np.sin(2 * np.pi * hour / 24))
    assert np.allclose(engineered["HourCos"], np.cos(2 * np.pi * hour / 24))


def test_validation_and_test_do_not_refit_scaler(transactions_factory):
    train = transactions_factory(100, seed=1)
    validation = transactions_factory(30, seed=2).assign(Amount=1_000_000.0)
    test = transactions_factory(30, seed=3).assign(Time=9_999_999.0)
    preprocessor = FraudPreprocessor()

    preprocessor.fit(train)
    center_before = preprocessor.scaler.center_.copy()
    scale_before = preprocessor.scaler.scale_.copy()
    preprocessor.transform(validation)
    preprocessor.transform(test)

    assert np.array_equal(preprocessor.scaler.center_, center_before)
    assert np.array_equal(preprocessor.scaler.scale_, scale_before)


def test_transform_rejects_unfitted_preprocessor(transactions_factory):
    with pytest.raises(NotFittedError, match="ajustarse"):
        FraudPreprocessor().transform(transactions_factory())


def test_preprocessor_rejects_missing_or_empty_input(transactions_factory):
    preprocessor = FraudPreprocessor()

    with pytest.raises(ValueError, match="Faltan variables"):
        preprocessor.fit(transactions_factory().drop(columns="Amount"))
    with pytest.raises(ValueError, match="vacías"):
        preprocessor.fit(transactions_factory().iloc[0:0])
