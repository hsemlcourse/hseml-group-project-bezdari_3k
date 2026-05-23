"""FastAPI application for Home Credit default-risk predictions."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException

from src.api.model_service import CreditRiskModelService, ModelArtifactMissingError
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)


def create_app(service: Any | None = None) -> FastAPI:
    """Create the FastAPI app with injectable model service for tests."""

    model_service = service or CreditRiskModelService()
    api = FastAPI(
        title="Home Credit Default Risk API",
        description="Prediction API for the HSE ML Home Credit Default Risk project.",
        version="1.0.0",
    )

    @api.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=model_service.is_loaded(),
            model_path=str(model_service.model_path),
            detail=model_service.status_detail(),
        )

    @api.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        try:
            return ModelInfoResponse(**model_service.get_model_info())
        except ModelArtifactMissingError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read model info: {exc}") from exc

    @api.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        try:
            result = model_service.predict_one(request.features)
            return PredictionResponse(**asdict(result))
        except ModelArtifactMissingError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    @api.post("/predict-batch", response_model=BatchPredictionResponse)
    def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
        try:
            predictions = [
                PredictionResponse(**asdict(prediction))
                for prediction in model_service.predict_many([item.features for item in request.items])
            ]
            return BatchPredictionResponse(predictions=predictions)
        except ModelArtifactMissingError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc

    return api


app = create_app()
