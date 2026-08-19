"""Construcción y utilidades de los modelos de detección de fraude."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def calculate_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Calcula pesos para compensar el desbalance entre clases."""
    labels = np.asarray(labels, dtype=np.int32)

    if labels.ndim != 1:
        raise ValueError("labels debe ser un arreglo de una dimensión.")

    counts = np.bincount(labels, minlength=2)

    if np.any(counts == 0):
        raise ValueError("Deben existir ejemplos de las clases 0 y 1.")

    total = int(counts.sum())

    return {
        0: total / (2 * int(counts[0])),
        1: total / (2 * int(counts[1])),
    }


def calculate_output_bias(labels: np.ndarray) -> float:
    """Calcula el sesgo inicial de salida para la MLP."""
    counts = np.bincount(np.asarray(labels, dtype=np.int32), minlength=2)
    negative, positive = counts[0], counts[1]

    if negative == 0 or positive == 0:
        raise ValueError("Deben existir ejemplos de las clases 0 y 1.")

    return float(np.log(positive / negative))
    
def build_mlp(input_dim: int, output_bias: float) -> tf.keras.Model:
    """Construye y compila una MLP para clasificación de fraude."""
    if input_dim <= 0:
        raise ValueError("input_dim debe ser mayor que cero.")

    regularizer = tf.keras.regularizers.l2(1e-4)

    inputs = tf.keras.Input(
        shape=(input_dim,),
        name="transaction_features",
    )

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizer,
        name="dense_64",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="batch_norm_1")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout_025")(x)

    x = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizer,
        name="dense_32",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="batch_norm_2")(x)
    x = tf.keras.layers.Dropout(0.15, name="dropout_015")(x)

    x = tf.keras.layers.Dense(
        16,
        activation="relu",
        name="dense_16",
    )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(output_bias),
        name="fraud_probability",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cost_sensitive_fraud_mlp",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3,
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