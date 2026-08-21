from dataclasses import asdict

import numpy as np
import pytest
import yaml

from laboratorio1.pipeline import PipelineParameters, PreparedDataset


def test_pipeline_parameters_load_all_sections(tmp_path):
    path = tmp_path / "params.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "split": {"validation_size": 0.2, "test_size": 0.1, "seed": 7},
                "mlp": {"epochs": 3},
                "autoencoder": {"epochs": 4},
            }
        ),
        encoding="utf-8",
    )

    parameters = PipelineParameters.load(path)

    assert asdict(parameters.split) == {
        "validation_size": 0.2,
        "test_size": 0.1,
        "seed": 7,
    }
    assert parameters.mlp.epochs == 3
    assert parameters.autoencoder.epochs == 4


def test_pipeline_parameters_reject_invalid_split(tmp_path):
    path = tmp_path / "params.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "split": {"validation_size": 0.6, "test_size": 0.4},
                "mlp": {},
                "autoencoder": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="proporciones"):
        PipelineParameters.load(path)


def test_prepared_dataset_round_trip(tmp_path):
    path = tmp_path / "prepared.npz"
    prepared = PreparedDataset(
        train_features=np.ones((4, 2), dtype=np.float32),
        validation_features=np.ones((2, 2), dtype=np.float32) * 2,
        test_features=np.ones((2, 2), dtype=np.float32) * 3,
        train_labels=np.array([0, 0, 1, 0]),
        validation_labels=np.array([0, 1]),
        test_labels=np.array([0, 1]),
        test_amounts=np.array([10.0, 20.0]),
        dataset_sha256="abc123",
        modeled_rows=8,
    )

    prepared.save(path)
    loaded = PreparedDataset.load(path)

    assert np.array_equal(loaded.train_features, prepared.train_features)
    assert np.array_equal(loaded.test_labels, prepared.test_labels)
    assert loaded.dataset_sha256 == "abc123"
    assert loaded.modeled_rows == 8
