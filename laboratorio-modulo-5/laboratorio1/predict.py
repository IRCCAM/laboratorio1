import json
from pathlib import Path

import joblib
import pandas as pd
import tensorflow as tf
from loguru import logger
import typer

from laboratorio1.config import MODELS_DIR, PROCESSED_DATA_DIR
from laboratorio1.features import FraudPreprocessor  # noqa: F401 (needed to unpickle)

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "predictions.csv",
    model_path: Path = MODELS_DIR / "modelo_mlp_fraude.keras",
    preprocessor_path: Path = MODELS_DIR / "preprocesador.joblib",
    threshold_path: Path = MODELS_DIR / "umbrales_decision.json",
    threshold_key: str = "mlp",
    # ----------------------------------------------
):
    """Score transactions with the trained MLP and flag fraud alerts.

    Expects `input_path` to contain the raw columns the model was trained on
    (Time, Amount, V1..V28) — feature engineering is handled here via the
    saved FraudPreprocessor, so pass raw transactions, not pre-scaled ones.
    """
    logger.info(f"Reading transactions from {input_path}...")
    transactions = pd.read_csv(input_path)

    logger.info(f"Loading preprocessor from {preprocessor_path}...")
    preprocessor: FraudPreprocessor = joblib.load(preprocessor_path)

    logger.info(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    logger.info(f"Loading decision threshold from {threshold_path}...")
    threshold = json.loads(threshold_path.read_text())[threshold_key]
    logger.info(f"Using threshold={threshold:.6f} (key='{threshold_key}')")

    logger.info("Transforming features...")
    X = preprocessor.transform(transactions)

    logger.info("Scoring transactions...")
    scores = model.predict(X, verbose=0).ravel()
    alerts = (scores >= threshold).astype(int)

    results = transactions.copy()
    results["fraud_probability"] = scores
    results["fraud_alert"] = alerts

    results.to_csv(output_path, index=False)
    logger.success(
        f"Wrote {len(results):,} predictions to {output_path} "
        f"({alerts.sum():,} alerts)."
    )


if __name__ == "__main__":
    app()
