"""Constructores y utilidades de los modelos de detección de fraude."""

from __future__ import annotations

import numpy as np

try:
    import tensorflow as tf
except ModuleNotFoundError:  # Permite probar utilidades sin instalar TensorFlow.
    tf = None

from laboratorio1.config import AutoencoderConfig, MLPConfig


def _binary_counts(labels: np.ndarray) -> tuple[int, int]:
    values = np.asarray(labels)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels debe ser un arreglo unidimensional no vacío")
    if set(np.unique(values)) != {0, 1}:
        raise ValueError("Deben existir ejemplos de las clases 0 y 1")
    negative, positive = np.bincount(values.astype(np.int32), minlength=2)
    return int(negative), int(positive)


def calculate_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Calcula pesos balanceados para las clases legítima y fraude."""
    negative, positive = _binary_counts(labels)
    total = negative + positive
    return {0: total / (2 * negative), 1: total / (2 * positive)}


def calculate_output_bias(labels: np.ndarray) -> float:
    """Calcula el sesgo logarítmico inicial de la salida de la MLP."""
    negative, positive = _binary_counts(labels)
    return float(np.log(positive / negative))


def build_mlp(
    input_dim: int,
    output_bias: float,
    config: MLPConfig | None = None,
) -> tf.keras.Model:
    """Construye y compila la MLP supervisada sensible al costo."""
    if tf is None:
        raise ModuleNotFoundError("TensorFlow es necesario para construir la MLP")
    if input_dim <= 0:
        raise ValueError("input_dim debe ser mayor que cero")
    settings = config or MLPConfig()
    regularizer = tf.keras.regularizers.l2(settings.l2_regularization)

    inputs = tf.keras.Input(shape=(input_dim,), name="transaction_features")
    layer = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizer,
        name="dense_64",
    )(inputs)
    layer = tf.keras.layers.BatchNormalization(name="batch_norm_1")(layer)
    layer = tf.keras.layers.Dropout(0.25, name="dropout_025")(layer)
    layer = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizer,
        name="dense_32",
    )(layer)
    layer = tf.keras.layers.BatchNormalization(name="batch_norm_2")(layer)
    layer = tf.keras.layers.Dropout(0.15, name="dropout_015")(layer)
    layer = tf.keras.layers.Dense(16, activation="relu", name="dense_16")(layer)
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(output_bias),
        name="fraud_probability",
    )(layer)

    model = tf.keras.Model(inputs, outputs, name="cost_sensitive_fraud_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=settings.learning_rate,
            clipnorm=1.0,
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def build_autoencoder(
    input_dim: int,
    config: AutoencoderConfig | None = None,
) -> tf.keras.Model:
    """Construye y compila el autoencoder denoising de anomalías."""
    if tf is None:
        raise ModuleNotFoundError("TensorFlow es necesario para construir el autoencoder")
    if input_dim <= 0:
        raise ValueError("input_dim debe ser mayor que cero")
    settings = config or AutoencoderConfig()

    inputs = tf.keras.Input(shape=(input_dim,), name="transaction_features")
    layer = tf.keras.layers.GaussianNoise(
        settings.noise_stddev,
        name="gaussian_noise",
    )(inputs)
    layer = tf.keras.layers.Dense(24, activation="relu", name="encoder_24")(layer)
    layer = tf.keras.layers.Dense(12, activation="relu", name="encoder_12")(layer)
    layer = tf.keras.layers.Dense(6, activation="relu", name="bottleneck_6")(layer)
    layer = tf.keras.layers.Dense(12, activation="relu", name="decoder_12")(layer)
    layer = tf.keras.layers.Dense(24, activation="relu", name="decoder_24")(layer)
    outputs = tf.keras.layers.Dense(
        input_dim,
        activation="linear",
        name="reconstruction",
    )(layer)

    model = tf.keras.Model(inputs, outputs, name="denoising_fraud_autoencoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=settings.learning_rate,
            clipnorm=1.0,
        ),
        loss="mae",
    )
    return model


def reconstruction_error(
    features: np.ndarray,
    reconstructed: np.ndarray,
) -> np.ndarray:
    """Calcula el MAE por transacción entre entrada y reconstrucción."""
    original = np.asarray(features, dtype=np.float32)
    predictions = np.asarray(reconstructed, dtype=np.float32)
    if original.ndim != 2 or predictions.ndim != 2:
        raise ValueError("features y reconstructed deben ser matrices bidimensionales")
    if original.shape != predictions.shape:
        raise ValueError("features y reconstructed deben tener la misma forma")
    if original.shape[0] == 0:
        raise ValueError("No se puede calcular el error de una matriz vacía")
    return np.mean(np.abs(original - predictions), axis=1)
