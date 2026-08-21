"""Configuración central del proyecto de detección de fraude."""

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"


@dataclass(frozen=True)
class SplitConfig:
    """Parámetros reproducibles de la partición estratificada."""

    validation_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42


@dataclass(frozen=True)
class MLPConfig:
    """Hiperparámetros de la red supervisada."""

    learning_rate: float = 1e-3
    l2_regularization: float = 1e-4
    epochs: int = 30
    batch_size: int = 2048
    prediction_batch_size: int = 4096
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 2
    min_learning_rate: float = 1e-6


@dataclass(frozen=True)
class AutoencoderConfig:
    """Hiperparámetros del autoencoder de anomalías."""

    learning_rate: float = 1e-3
    noise_stddev: float = 0.02
    epochs: int = 40
    batch_size: int = 2048
    prediction_batch_size: int = 4096
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 2
    min_learning_rate: float = 1e-6


@dataclass(frozen=True)
class MLflowConfig:
    """Configuración del seguimiento local de experimentos."""

    enabled: bool = True
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "deteccion_fraude"
    log_models: bool = True


@dataclass(frozen=True)
class ArtifactPaths:
    """Rutas de los artefactos producidos por el entrenamiento."""

    mlp_model: Path = MODELS_DIR / "modelo_mlp_fraude.keras"
    autoencoder_model: Path = MODELS_DIR / "modelo_autoencoder_fraude.keras"
    preprocessor: Path = MODELS_DIR / "preprocesador_fraude.joblib"
    thresholds: Path = MODELS_DIR / "umbrales_decision.json"
    metrics: Path = RESULTS_DIR / "metricas_modelos.csv"
    provenance: Path = RESULTS_DIR / "proveniencia_reproducibilidad.json"


SPLIT_CONFIG = SplitConfig()
MLP_CONFIG = MLPConfig()
AUTOENCODER_CONFIG = AutoencoderConfig()
ARTIFACT_PATHS = ArtifactPaths()
