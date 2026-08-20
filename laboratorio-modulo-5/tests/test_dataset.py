import numpy as np
import pandas as pd
import pytest

from laboratorio1.dataset import FraudDatasetLoader, create_stratified_indices


def test_loader_validates_and_removes_duplicates(tmp_path, transactions_factory):
    dataframe = transactions_factory(100)
    dataframe = pd.concat([dataframe, dataframe.iloc[[0]]], ignore_index=True)
    path = tmp_path / "creditcard.csv"
    dataframe.to_csv(path, index=False)

    loaded = FraudDatasetLoader().load(path)

    assert len(loaded) == 100
    assert not loaded.duplicated().any()


def test_loader_rejects_nonexistent_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="No existe"):
        FraudDatasetLoader().load(tmp_path / "missing.csv")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns="V28"), "Faltan columnas"),
        (lambda frame: frame.assign(V1=np.nan), "valores faltantes"),
        (lambda frame: frame.assign(Class=0), "Class debe contener"),
    ],
)
def test_loader_rejects_invalid_data(transactions_factory, mutation, message):
    dataframe = mutation(transactions_factory())

    with pytest.raises(ValueError, match=message):
        FraudDatasetLoader().validate(dataframe)


def test_stratified_split_is_disjoint_complete_and_balanced(transactions_factory):
    dataframe = transactions_factory(200)
    labels = dataframe["Class"].to_numpy()

    split = create_stratified_indices(labels)
    all_indices = np.concatenate([split.train, split.validation, split.test])

    assert len(np.unique(all_indices)) == len(dataframe)
    assert (len(split.train), len(split.validation), len(split.test)) == (140, 30, 30)
    assert labels[split.train].mean() == pytest.approx(labels.mean())
    assert labels[split.validation].mean() == pytest.approx(labels.mean())
    assert labels[split.test].mean() == pytest.approx(labels.mean())


def test_split_rejects_invalid_sizes(transactions_factory):
    labels = transactions_factory()["Class"].to_numpy()

    with pytest.raises(ValueError, match="proporciones"):
        create_stratified_indices(labels, validation_size=0.6, test_size=0.4)
