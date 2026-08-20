"""API pública de entrenamiento, modelos e inferencia."""

from typing import TYPE_CHECKING

from laboratorio1.modeling.models import (
    build_autoencoder,
    build_mlp,
    calculate_class_weights,
    calculate_output_bias,
    reconstruction_error,
)

if TYPE_CHECKING:
    from laboratorio1.modeling.predict import FraudDetectionService

__all__ = [
    "FraudDetectionService",
    "build_autoencoder",
    "build_mlp",
    "calculate_class_weights",
    "calculate_output_bias",
    "reconstruction_error",
]


def __getattr__(name: str):
    """Carga el servicio de inferencia sin interferir con su ejecución como CLI."""
    if name == "FraudDetectionService":
        from laboratorio1.modeling.predict import FraudDetectionService

        return FraudDetectionService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
