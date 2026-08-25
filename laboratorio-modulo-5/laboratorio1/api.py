"""API HTTP local para servir el modelo MLP de detección de fraude."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
import logging
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from laboratorio1.config import ARTIFACT_PATHS
from laboratorio1.modeling.predict import MLPFraudDetectionService

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


ServiceLoader = Callable[[], MLPFraudDetectionService]


def create_app(
    service_loader: ServiceLoader = MLPFraudDetectionService.from_artifacts,
) -> FastAPI:
    """Crea la aplicación y carga los artefactos locales durante el arranque."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.service = None
        application.state.load_error = None
        try:
            application.state.service = service_loader()
        except Exception as error:
            application.state.load_error = str(error)
            logger.exception("No fue posible cargar los artefactos locales del modelo")
        yield

    application = FastAPI(
        title="API de Detección de Fraude",
        description="Inferencia local con un modelo MLP supervisado.",
        version="1.1.0",
        lifespan=lifespan,
    )

    @application.get("/", tags=["estado"])
    def root(request: Request) -> dict[str, object]:
        service = request.app.state.service
        return {
            "status": "online" if service is not None else "modelo_no_disponible",
            "modelo": "mlp_deteccion_fraude",
            "documentacion": "/docs",
        }

    @application.get("/health", tags=["estado"])
    def health(request: Request) -> dict[str, object]:
        service = request.app.state.service
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "estado": "no_disponible",
                    "error": request.app.state.load_error,
                },
            )
        return {
            "estado": "disponible",
            "artefactos": {
                "mlp": str(ARTIFACT_PATHS.mlp_model),
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
        service: MLPFraudDetectionService | None = request.app.state.service
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="El modelo local no está cargado. Consulte GET /health.",
            )

        try:
            import pandas as pd

            transactions = pd.DataFrame(
                [transaction.model_dump() for transaction in payload.data]
            )
            prediction_frame = service.predict(transactions).reset_index(drop=True)
            results = [
                PredictionResult(
                    index=index,
                    probabilidad_fraude=float(row.probabilidad_fraude),
                    alerta_fraude=int(row.alerta_fraude),
                )
                for index, row in prediction_frame.iterrows()
            ]
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
