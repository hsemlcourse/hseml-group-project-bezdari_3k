from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.model_service import ModelArtifactMissingError, PredictionResult


class FakeService:
    model_path = Path("models/best_model.joblib")

    def is_loaded(self):
        return True

    def status_detail(self):
        return "Model artifact is loaded."

    def get_model_info(self):
        return {
            "model_loaded": True,
            "model_path": str(self.model_path),
            "metadata": {"best_model": "fake_model"},
            "expected_features": ["AMT_CREDIT", "AMT_INCOME_TOTAL"],
            "expected_feature_count": 2,
        }

    def predict_one(self, features):
        return PredictionResult(
            default_probability=0.61,
            risk_level="high",
            model_name="fake_model",
            missing_features=[],
            extra_features=sorted(set(features) - {"AMT_CREDIT", "AMT_INCOME_TOTAL"}),
        )


class MissingModelService(FakeService):
    def is_loaded(self):
        return False

    def status_detail(self):
        return "Model artifact is missing."

    def predict_one(self, features):
        raise ModelArtifactMissingError("Train the model with python3 -m src.modeling")


def test_health_reports_loaded_model():
    client = TestClient(create_app(FakeService()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_model_info_returns_expected_features():
    client = TestClient(create_app(FakeService()))

    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["expected_feature_count"] == 2


def test_predict_returns_probability_and_risk_level():
    client = TestClient(create_app(FakeService()))

    response = client.post(
        "/predict",
        json={"features": {"AMT_CREDIT": 100000.0, "AMT_INCOME_TOTAL": 50000.0, "UNUSED": 1}},
    )

    assert response.status_code == 200
    assert response.json()["default_probability"] == 0.61
    assert response.json()["risk_level"] == "high"
    assert response.json()["extra_features"] == ["UNUSED"]


def test_predict_batch_returns_predictions():
    client = TestClient(create_app(FakeService()))

    response = client.post(
        "/predict-batch",
        json={
            "items": [
                {"features": {"AMT_CREDIT": 100000.0}},
                {"features": {"AMT_CREDIT": 200000.0}},
            ]
        },
    )

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_predict_returns_503_when_model_is_missing():
    client = TestClient(create_app(MissingModelService()))

    response = client.post("/predict", json={"features": {"AMT_CREDIT": 100000.0}})

    assert response.status_code == 503
    assert "python3 -m src.modeling" in response.json()["detail"]
