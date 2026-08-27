"""API HTTP local que consume el modelo MLP servido en Docker."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
import logging
from typing import Annotated, Protocol

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from laboratorio1.config import ARTIFACT_PATHS
from laboratorio1.modeling.remote import (
    MLPServiceResponseError,
    MLPServiceUnavailable,
    RemoteMLPFraudDetectionService,
)

logger = logging.getLogger(__name__)


class TransactionFeatures(BaseModel):
    """Variables originales de una transacción del dataset Credit Card Fraud."""

    model_config = ConfigDict(extra="forbid")

    Time: Annotated[float, Field(ge=0)]
    Amount: Annotated[float, Field(ge=0)]
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


class PredictionRequest(BaseModel):
    """Lote no vacío de transacciones a evaluar."""

    model_config = ConfigDict(extra="forbid")

    data: Annotated[list[TransactionFeatures], Field(min_length=1, max_length=10_000)]


class PredictionResult(BaseModel):
    index: int
    probabilidad_fraude: float
    alerta_fraude: int


class PredictionResponse(BaseModel):
    total_predicciones: int
    umbral: float
    resultados: list[PredictionResult]


class PredictionService(Protocol):
    """Contrato requerido por los servicios real y de prueba."""

    threshold: float
    service_url: str

    def is_ready(self) -> bool: ...

    def predict(self, transactions: object) -> object: ...

    def close(self) -> None: ...


ServiceLoader = Callable[[], PredictionService]


def create_app(
    service_loader: ServiceLoader = RemoteMLPFraudDetectionService.from_artifacts,
) -> FastAPI:
    """Crea la API local y configura el cliente del MLP en Docker."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.service = None
        application.state.load_error = None
        try:
            application.state.service = service_loader()
        except Exception as error:
            application.state.load_error = str(error)
            logger.exception("No fue posible configurar el cliente del modelo MLP")
        try:
            yield
        finally:
            service = application.state.service
            if service is not None:
                try:
                    service.close()
                except Exception:
                    logger.exception("No fue posible cerrar el cliente del modelo MLP")

    application = FastAPI(
        title="API de Detección de Fraude",
        description="FastAPI local con inferencia MLP proporcionada por Docker.",
        version="1.2.0",
        lifespan=lifespan,
    )

    @application.get("/", tags=["estado"])
    def root(request: Request) -> dict[str, object]:
        service = request.app.state.service
        return {
            "status": "api_online",
            "cliente_mlp": "configurado" if service is not None else "no_configurado",
            "modelo": "mlp_deteccion_fraude",
            "documentacion": "/docs",
        }

    @application.get("/health", tags=["estado"])
    def health(request: Request) -> dict[str, object]:
        service: PredictionService | None = request.app.state.service
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "estado": "no_disponible",
                    "error": request.app.state.load_error,
                },
            )
        try:
            model_is_ready = service.is_ready()
        except Exception as error:
            logger.exception("Falló la comprobación del servicio MLP")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"estado": "no_disponible", "error": str(error)},
            ) from error
        if not model_is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "estado": "no_disponible",
                    "error": "El contenedor del modelo MLP no responde.",
                },
            )
        return {
            "estado": "disponible",
            "servicio_mlp": service.service_url,
            "artefactos": {
                "preprocesador": str(ARTIFACT_PATHS.preprocessor),
                "umbral": str(ARTIFACT_PATHS.thresholds),
            },
        }

    @application.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["inferencia"],
    )
    def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        service: PredictionService | None = request.app.state.service
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="El cliente del modelo no está configurado. Consulte GET /health.",
            )

        try:
            import pandas as pd

            transactions = pd.DataFrame([transaction.model_dump() for transaction in payload.data])
            prediction_frame = service.predict(transactions).reset_index(drop=True)
            results = [
                PredictionResult(
                    index=index,
                    probabilidad_fraude=float(row.probabilidad_fraude),
                    alerta_fraude=int(row.alerta_fraude),
                )
                for index, row in prediction_frame.iterrows()
            ]
        except MLPServiceUnavailable as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except MLPServiceResponseError as error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(error),
            ) from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            logger.exception("Error inesperado durante la inferencia")
            raise HTTPException(
                status_code=500,
                detail="No fue posible completar la inferencia.",
            ) from error

        return PredictionResponse(
            total_predicciones=len(results),
            umbral=service.threshold,
            resultados=results,
        )

    return application


app = create_app()
