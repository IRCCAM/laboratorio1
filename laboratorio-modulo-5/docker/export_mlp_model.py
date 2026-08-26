"""Convierte el artefacto Keras de DVC en un modelo servible por MLflow."""

from __future__ import annotations

import argparse
from pathlib import Path

FEATURE_COUNT = 32


def export_model(source: Path, destination: Path) -> None:
    """Guarda el MLP con firma tensorial estable para ``/invocations``."""
    from mlflow.models import infer_signature
    import mlflow.tensorflow
    import numpy as np
    import tensorflow as tf

    if not source.is_file():
        raise FileNotFoundError(f"No se encontró el modelo MLP: {source}")
    if destination.exists():
        raise FileExistsError(f"El destino ya existe: {destination}")

    model = tf.keras.models.load_model(source)
    input_example = np.zeros((1, FEATURE_COUNT), dtype=np.float32)
    prediction_example = model.predict(input_example, verbose=0)
    signature = infer_signature(input_example, prediction_example)

    destination.parent.mkdir(parents=True, exist_ok=True)
    mlflow.tensorflow.save_model(
        model=model,
        path=str(destination),
        signature=signature,
        input_example=input_example,
        pip_requirements=[
            "mlflow==3.15.1",
            "tensorflow==2.20.0",
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_model(args.source, args.destination)


if __name__ == "__main__":
    main()
