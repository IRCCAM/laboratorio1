import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from sklearn.metrics import average_precision_score, fbeta_score, precision_recall_curve
from sklearn.model_selection import train_test_split
import typer

from laboratorio1.config import MODELS_DIR, PROCESSED_DATA_DIR
from laboratorio1.features import FraudPreprocessor

app = typer.Typer()

SEED = 42


def best_threshold_f2(y_true, scores):
    """Pick the score threshold that maximizes F2 (weights recall higher than precision)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]
    denominator = 4 * precision + recall
    f2 = np.divide(
        5 * precision * recall, denominator,
        out=np.zeros_like(precision), where=denominator > 0,
    )
    idx = int(np.nanargmax(f2))
    return float(thresholds[idx]), float(f2[idx])


def build_mlp(input_dim: int, output_bias: float) -> tf.keras.Model:
    """Cost-sensitive MLP: Dense(64) -> Dense(32) -> Dense(16) -> sigmoid."""
    reg = tf.keras.regularizers.l2(1e-4)
    inputs = tf.keras.Input(shape=(input_dim,), name="transaction_features")
    x = tf.keras.layers.Dense(
        64, activation="relu", kernel_initializer="he_normal", kernel_regularizer=reg
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(
        32, activation="relu", kernel_initializer="he_normal", kernel_regularizer=reg
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(output_bias),
    )(x)
    model = tf.keras.Model(inputs, outputs, name="fraud_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )
    return model


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    model_path: Path = MODELS_DIR / "modelo_mlp_fraude.keras",
    preprocessor_path: Path = MODELS_DIR / "preprocesador.joblib",
    threshold_path: Path = MODELS_DIR / "umbrales_decision.json",
    threshold_key: str = "mlp",
    epochs: int = 30,
    batch_size: int = 2048,
    val_size: float = 0.15,
    test_size: float = 0.15,
    # -----------------------------------------
):
    """Train the cost-sensitive MLP fraud detector and save its artifacts.

    Expects `input_path` to hold the cleaned dataset (raw Time/Amount/V1..V28
    columns plus a `Class` label, duplicates already dropped). The train/val/test
    split happens here so the preprocessor and class weights are fit on the
    training split only, matching the notebook's methodology.
    """
    tf.keras.utils.set_random_seed(SEED)

    logger.info(f"Reading dataset from {input_path}...")
    df = pd.read_csv(input_path).reset_index(drop=True)
    assert "Class" in df.columns, "Input dataset must include a 'Class' label column."

    y = df["Class"].to_numpy(dtype=np.int32)
    indices = np.arange(len(df))

    logger.info("Splitting into train/val/test (stratified)...")
    train_idx, temp_idx = train_test_split(
        indices, test_size=val_size + test_size, stratify=y, random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_size / (val_size + test_size),
        stratify=y[temp_idx],
        random_state=SEED,
    )

    logger.info("Fitting preprocessor on training data...")
    preprocessor = FraudPreprocessor()
    X_train = preprocessor.fit_transform(df.loc[train_idx])
    X_val = preprocessor.transform(df.loc[val_idx])
    X_test = preprocessor.transform(df.loc[test_idx])
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    neg, pos = np.bincount(y_train)
    class_weight = {0: (neg + pos) / (2 * neg), 1: (neg + pos) / (2 * pos)}
    output_bias = float(np.log(pos / neg))
    logger.info(f"Class weights: {class_weight}")

    logger.info("Building and training the MLP...")
    model = build_mlp(X_train.shape[1], output_bias)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max", patience=5,
            min_delta=1e-4, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc", mode="max", factor=0.5,
            patience=2, min_lr=1e-6, verbose=1,
        ),
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        class_weight=class_weight, callbacks=callbacks,
        shuffle=True, verbose=2,
    )

    logger.info("Selecting decision threshold on validation (max F2)...")
    val_scores = model.predict(X_val, batch_size=4096, verbose=0).ravel()
    threshold, f2_val = best_threshold_f2(y_val, val_scores)
    logger.info(f"Threshold={threshold:.6f}  F2(val)={f2_val:.4f}")

    test_scores = model.predict(X_test, batch_size=4096, verbose=0).ravel()
    pred_test = (test_scores >= threshold).astype(int)
    logger.info(
        f"Test PR-AUC={average_precision_score(y_test, test_scores):.4f}  "
        f"F2(test)={fbeta_score(y_test, pred_test, beta=2):.4f}"
    )

    logger.info(f"Saving model to {model_path}...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    logger.info(f"Saving preprocessor to {preprocessor_path}...")
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)

    logger.info(f"Saving threshold to {threshold_path}...")
    thresholds = {}
    if threshold_path.exists():
        thresholds = json.loads(threshold_path.read_text())
    thresholds[threshold_key] = threshold
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_path.write_text(json.dumps(thresholds, indent=2))

    logger.success("Training complete.")


if __name__ == "__main__":
    app()
