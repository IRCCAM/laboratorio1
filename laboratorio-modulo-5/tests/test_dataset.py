import numpy as np
import pytest

from laboratorio1.dataset import FraudDatasetLoader, create_stratified_indices


def test_loader_validates_expected_schema(transactions_factory):
    dataframe = transactions_factory()
    FraudDatasetLoader().validate(dataframe)

    with pytest.raises(ValueError, match="Faltan columnas"):
        FraudDatasetLoader().validate(dataframe.drop(columns="V28"))


def test_stratified_split_is_disjoint_and_complete(transactions_factory):
    dataframe = transactions_factory(200)
    split = create_stratified_indices(dataframe["Class"].to_numpy())
    all_indices = np.concatenate([split.train, split.validation, split.test])

    assert len(np.unique(all_indices)) == len(dataframe)
    assert len(split.train) == 140
    assert len(split.validation) == 30
    assert len(split.test) == 30
