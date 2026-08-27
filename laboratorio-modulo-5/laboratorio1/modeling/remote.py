"""Cliente HTTP para consumir el modelo MLP servido por MLflow."""

from __future__ import annotations

import json
import os
from typing import Protocol

import httpx2
import joblib
import numpy as np
import pandas as pd

from laboratorio1.config import ARTIFACT_PATHS, ArtifactPaths
from laboratorio1.features import FraudPreprocessor


class MLPServiceUnavailable(RuntimeError):
    """El servicio HTTP del modelo no pudo ser alcanzado."""


class MLPServiceResponseError(RuntimeError):
    """El servicio HTTP respondió con un error o un formato inválido."""


class HTTPClient(Protocol):
    """Interfaz mínima para el cliente HTTP real y los dobles de prueba."""

    def get(self, url: str) -> httpx2.Response: ...

    def post(self, url: str, **kwargs: object) -> httpx2.Response: ...

    def close(self) -> None: ...


class RemoteMLPFraudDetectionService:
    """Preprocesa transacciones y delega la inferencia al MLP de Docker."""

    DEFAULT_SERVICE_URL = "http://127.0.0.1:8080"
    DEFAULT_TIMEOUT_SECONDS = 10.0

    def __init__(
        self,
        preprocessor: FraudPreprocessor,
        threshold: float,
        service_url: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: HTTPClient | None = None,
    ) -> None:
        if not np.isfinite(threshold):
            raise ValueError("El umbral debe ser finito")
        if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("El tiempo de espera debe ser un número positivo")

        normalized_url = service_url.strip().rstrip("/")
        if not normalized_url:
            raise ValueError("La URL del servicio MLP no puede estar vacía")

        self.preprocessor = preprocessor
        self.threshold = float(threshold)
        self.service_url = normalized_url
        self._owns_client = client is None
        self._client = client or httpx2.Client(timeout=float(timeout_seconds))

    @classmethod
    def from_artifacts(
        cls,
        artifact_paths: ArtifactPaths = ARTIFACT_PATHS,
        service_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> RemoteMLPFraudDetectionService:
        """Carga solo el preprocesador y el umbral; el MLP vive en Docker."""
        preprocessor = joblib.load(artifact_paths.preprocessor)
        if not isinstance(preprocessor, FraudPreprocessor):
            raise TypeError("El artefacto no contiene un FraudPreprocessor")

        thresholds = json.loads(artifact_paths.thresholds.read_text(encoding="utf-8"))
        if "mlp" not in thresholds:
            raise ValueError("Falta el umbral requerido para la MLP")

        resolved_url = service_url or os.getenv(
            "MLP_SERVICE_URL",
            cls.DEFAULT_SERVICE_URL,
        )
        if timeout_seconds is None:
            raw_timeout = os.getenv(
                "MLP_SERVICE_TIMEOUT_SECONDS",
                str(cls.DEFAULT_TIMEOUT_SECONDS),
            )
            try:
                timeout_seconds = float(raw_timeout)
            except ValueError as error:
                raise ValueError("MLP_SERVICE_TIMEOUT_SECONDS debe ser un número") from error

        return cls(
            preprocessor=preprocessor,
            threshold=thresholds["mlp"],
            service_url=resolved_url,
            timeout_seconds=timeout_seconds,
        )

    def is_ready(self) -> bool:
        """Comprueba que el contenedor MLP responde correctamente."""
        try:
            response = self._client.get(f"{self.service_url}/health")
            response.raise_for_status()
        except (httpx2.RequestError, httpx2.HTTPStatusError):
            return False
        return True

    def predict(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Envía variables transformadas al endpoint MLflow ``/invocations``."""
        features = self.preprocessor.transform(transactions)
        try:
            response = self._client.post(
                f"{self.service_url}/invocations",
                json={"inputs": features.tolist()},
            )
        except httpx2.RequestError as error:
            raise MLPServiceUnavailable(
                f"No fue posible conectar con el modelo MLP en {self.service_url}"
            ) from error

        try:
            response.raise_for_status()
        except httpx2.HTTPStatusError as error:
            raise MLPServiceResponseError(
                f"El modelo MLP respondió con HTTP {response.status_code}"
            ) from error

        try:
            payload = response.json()
            raw_scores = np.asarray(payload["predictions"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise MLPServiceResponseError(
                "El modelo MLP devolvió una respuesta inválida"
            ) from error

        if raw_scores.shape == (len(transactions),):
            scores = raw_scores
        elif raw_scores.shape == (len(transactions), 1):
            scores = raw_scores[:, 0]
        else:
            raise MLPServiceResponseError(
                "El modelo MLP devolvió una forma inesperada de predicciones"
            )
        if not np.isfinite(scores).all():
            raise MLPServiceResponseError("El modelo MLP devolvió valores no finitos")
        if ((scores < 0) | (scores > 1)).any():
            raise MLPServiceResponseError(
                "El modelo MLP devolvió probabilidades fuera del intervalo [0, 1]"
            )

        return pd.DataFrame(
            {
                "probabilidad_fraude": scores,
                "alerta_fraude": (scores >= self.threshold).astype(np.int32),
            },
            index=transactions.index,
        )

    def close(self) -> None:
        """Libera las conexiones HTTP creadas por el servicio."""
        if self._owns_client:
            self._client.close()
