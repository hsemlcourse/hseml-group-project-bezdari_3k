"""Pydantic schemas for the credit-risk deploy API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


FeatureValue = int | float | str | bool | None


class PredictionRequest(BaseModel):
    """One applicant feature payload."""

    features: dict[str, FeatureValue] = Field(..., min_length=1)


class BatchPredictionRequest(BaseModel):
    """Batch prediction payload."""

    items: list[PredictionRequest] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    """Prediction result returned by `/predict`."""

    default_probability: float
    risk_level: str
    model_name: str
    missing_features: list[str]
    extra_features: list[str]


class BatchPredictionResponse(BaseModel):
    """Prediction result returned by `/predict-batch`."""

    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Service health response."""

    status: str
    model_loaded: bool
    model_path: str
    detail: str


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    model_loaded: bool
    model_path: str
    metadata: dict[str, Any]
    expected_features: list[str]
    expected_feature_count: int
