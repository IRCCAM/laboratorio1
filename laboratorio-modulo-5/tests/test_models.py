import numpy as np
import pytest

from laboratorio1.modeling.models import (
    build_autoencoder,
    build_mlp,
    calculate_class_weights,
    calculate_output_bias,
    reconstruction_error,
)


def test_class_weights_and_output_bias():
    labels = np.array([0, 0, 0, 1])

    assert calculate_class_weights(labels) == {0: 4 / 6, 1: 2.0}
    assert calculate_output_bias(labels) == pytest.approx(np.log(1 / 3))


def test_class_statistics_require_both_classes():
    with pytest.raises(ValueError, match="clases 0 y 1"):
        calculate_class_weights(np.zeros(4))


def test_reconstruction_error_is_row_wise_mae():
    features = np.array([[1.0, 2.0], [3.0, 5.0]])
    reconstructed = np.array([[0.0, 2.0], [5.0, 4.0]])

    assert np.allclose(reconstruction_error(features, reconstructed), [0.5, 1.5])


def test_reconstruction_error_rejects_different_shapes():
    with pytest.raises(ValueError, match="misma forma"):
        reconstruction_error(np.ones((2, 3)), np.ones((2, 2)))


def test_model_architectures_match_notebook():
    pytest.importorskip("tensorflow")

    mlp = build_mlp(32, output_bias=-1.0)
    autoencoder = build_autoencoder(32)

    assert mlp.input_shape == (None, 32)
    assert mlp.output_shape == (None, 1)
    assert autoencoder.input_shape == (None, 32)
    assert autoencoder.output_shape == (None, 32)
    assert autoencoder.get_layer("bottleneck_6").units == 6


def test_mlp_keeps_notebook_architecture():
    """Verifica que la MLP conserve la arquitectura del notebook original."""
    pytest.importorskip("tensorflow")

    model = build_mlp(input_dim=32, output_bias=-1.0)

    assert model.get_layer("dense_64").units == 64
    assert model.get_layer("dense_32").units == 32
    assert model.get_layer("dense_16").units == 16
    assert model.get_layer("dropout_025").rate == pytest.approx(0.25)
    assert model.get_layer("dropout_015").rate == pytest.approx(0.15)
    assert model.get_layer("fraud_probability").activation.__name__ == "sigmoid"
