"""Model loading and prediction service for the FastAPI deploy layer."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from src.preprocessing import add_domain_features


DEFAULT_MODEL_PATH = Path("models/best_model.joblib")
DEFAULT_METADATA_PATH = Path("models/run_metadata.json")
TRAINING_COMMAND = "python3 -m src.modeling --sample-size 50000 --top-n-features 120"


class ModelArtifactMissingError(RuntimeError):
    """Raised when prediction is requested before a model artifact exists."""


@dataclass(frozen=True)
class PredictionResult:
    """Serializable prediction result returned by the service."""

    default_probability: float
    risk_level: str
    model_name: str
    missing_features: list[str]
    extra_features: list[str]


def get_risk_level(probability: float) -> str:
    """Map default probability to a simple product-facing risk label."""

    if probability < 0.2:
        return "low"
    if probability < 0.5:
        return "medium"
    return "high"


class CreditRiskModelService:
    """Load a sklearn pipeline and produce default-risk predictions."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        metadata_path: Path | str | None = None,
        *,
        model: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model_path = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
        self.metadata_path = Path(metadata_path or os.getenv("MODEL_METADATA_PATH", DEFAULT_METADATA_PATH))
        self._model = model
        self._metadata = metadata

    @classmethod
    def from_model(
        cls,
        *,
        model: Any,
        metadata: dict[str, Any] | None = None,
        model_path: Path | str = DEFAULT_MODEL_PATH,
    ) -> "CreditRiskModelService":
        """Build a service around an already-loaded model, mainly for tests."""

        return cls(model_path=model_path, model=model, metadata=metadata or {})

    def is_loaded(self) -> bool:
        """Return whether the service has or can see a model artifact."""

        return self._model is not None or self.model_path.exists()

    def status_detail(self) -> str:
        """Return a short human-readable model status."""

        if self._model is not None:
            return "Model artifact is loaded."
        if self.model_path.exists():
            return "Model artifact is available and will be loaded on first prediction."
        return f"Model artifact is missing. Train it locally with: {TRAINING_COMMAND}"

    def load_model(self) -> Any:
        """Load the sklearn model lazily."""

        if self._model is not None:
            return self._model
        if not self.model_path.exists():
            raise ModelArtifactMissingError(
                f"Model artifact not found at {self.model_path}. Train the model with: {TRAINING_COMMAND}"
            )
        self._model = joblib.load(self.model_path)
        return self._model

    def load_metadata(self) -> dict[str, Any]:
        """Load run metadata if it exists."""

        if self._metadata is not None:
            return self._metadata
        if not self.metadata_path.exists():
            self._metadata = {}
            return self._metadata
        self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return self._metadata

    def expected_features(self) -> list[str]:
        """Return the feature order expected by the fitted sklearn pipeline."""

        model = self.load_model()
        direct_feature_names = self._normalise_feature_names(getattr(model, "feature_names_in_", []))
        if direct_feature_names:
            return direct_feature_names

        named_steps = getattr(model, "named_steps", {})
        for step in named_steps.values():
            step_feature_names = self._normalise_feature_names(getattr(step, "feature_names_in_", []))
            if step_feature_names:
                return step_feature_names

        return []

    @staticmethod
    def _normalise_feature_names(feature_names: Any) -> list[str]:
        if feature_names is None:
            return []
        return [str(feature) for feature in list(feature_names)]

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata for HTTP clients."""

        expected_features = self.expected_features() if self.is_loaded() else []
        return {
            "model_loaded": self.is_loaded(),
            "model_path": str(self.model_path),
            "metadata": self.load_metadata(),
            "expected_features": expected_features,
            "expected_feature_count": len(expected_features),
        }

    def predict_one(self, features: Mapping[str, Any]) -> PredictionResult:
        """Predict default probability for one applicant payload."""

        model = self.load_model()
        expected_features = self.expected_features()
        frame, missing_features, extra_features = self._build_input_frame(features, expected_features)
        probabilities = model.predict_proba(frame)
        default_probability = float(probabilities[0][1])
        metadata = self.load_metadata()

        return PredictionResult(
            default_probability=default_probability,
            risk_level=get_risk_level(default_probability),
            model_name=str(metadata.get("best_model") or metadata.get("model_name") or "unknown"),
            missing_features=missing_features,
            extra_features=extra_features,
        )

    @staticmethod
    def _build_input_frame(
        features: Mapping[str, Any],
        expected_features: list[str],
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        feature_dict = dict(features)
        expanded_features = add_domain_features(pd.DataFrame([feature_dict])).iloc[0].to_dict()
        if not expected_features:
            return pd.DataFrame([expanded_features]), [], []

        missing_features = [feature for feature in expected_features if feature not in expanded_features]
        extra_features = sorted(feature for feature in feature_dict if feature not in expected_features)
        aligned = {feature: expanded_features.get(feature) for feature in expected_features}
        return pd.DataFrame([aligned], columns=expected_features), missing_features, extra_features
