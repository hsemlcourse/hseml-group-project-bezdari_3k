from pathlib import Path

import numpy as np

from src.api.model_service import CreditRiskModelService, ModelArtifactMissingError, get_risk_level


class DummyProbabilityModel:
    feature_names_in_ = np.array(["AMT_CREDIT", "AMT_INCOME_TOTAL", "EXT_SOURCE_2"])

    def predict_proba(self, frame):
        assert list(frame.columns) == ["AMT_CREDIT", "AMT_INCOME_TOTAL", "EXT_SOURCE_2"]
        assert frame.loc[0, "EXT_SOURCE_2"] is None
        return np.array([[0.2, 0.8]])


class DummyPreprocessor:
    feature_names_in_ = np.array(["APARTMENTS_MODE", "AMT_CREDIT", "EXT_SOURCE_2"])


class DummyPipelineModel:
    named_steps = {"preprocessor": DummyPreprocessor()}

    def predict_proba(self, frame):
        assert list(frame.columns) == ["APARTMENTS_MODE", "AMT_CREDIT", "EXT_SOURCE_2"]
        assert frame.loc[0, "APARTMENTS_MODE"] is None
        return np.array([[0.7, 0.3]])


class DummyFeatureEngineeringModel:
    feature_names_in_ = np.array(["AMT_CREDIT", "AMT_INCOME_TOTAL", "NEW_CREDIT_TO_INCOME_RATIO"])

    def predict_proba(self, frame):
        assert frame.loc[0, "NEW_CREDIT_TO_INCOME_RATIO"] == 2.0
        return np.array([[0.6, 0.4]])


class DummyBatchModel:
    feature_names_in_ = np.array(["AMT_CREDIT", "AMT_INCOME_TOTAL", "NEW_CREDIT_TO_INCOME_RATIO"])

    def predict_proba(self, frame):
        assert list(frame.columns) == ["AMT_CREDIT", "AMT_INCOME_TOTAL", "NEW_CREDIT_TO_INCOME_RATIO"]
        assert len(frame) == 2
        assert frame["NEW_CREDIT_TO_INCOME_RATIO"].tolist() == [2.0, 3.0]
        return np.array([[0.9, 0.1], [0.3, 0.7]])


def test_get_risk_level_maps_probability_bands():
    assert get_risk_level(0.19) == "low"
    assert get_risk_level(0.49) == "medium"
    assert get_risk_level(0.5) == "high"


def test_predict_one_aligns_features_and_reports_missing_values():
    service = CreditRiskModelService.from_model(
        model=DummyProbabilityModel(),
        metadata={"best_model": "dummy_model"},
        model_path=Path("models/best_model.joblib"),
    )

    result = service.predict_one({"AMT_CREDIT": 100000.0, "AMT_INCOME_TOTAL": 50000.0, "UNUSED": 1})

    assert result.default_probability == 0.8
    assert result.risk_level == "high"
    assert result.model_name == "dummy_model"
    assert result.missing_features == ["EXT_SOURCE_2"]
    assert result.extra_features == ["UNUSED"]


def test_predict_one_adds_domain_features_before_alignment():
    service = CreditRiskModelService.from_model(
        model=DummyFeatureEngineeringModel(),
        metadata={"best_model": "dummy_fe"},
        model_path=Path("models/best_model.joblib"),
    )

    result = service.predict_one({"AMT_CREDIT": 100000.0, "AMT_INCOME_TOTAL": 50000.0})

    assert result.default_probability == 0.4
    assert result.risk_level == "medium"
    assert result.missing_features == []


def test_predict_one_reads_expected_features_from_pipeline_step():
    service = CreditRiskModelService.from_model(
        model=DummyPipelineModel(),
        metadata={"best_model": "dummy_pipeline"},
        model_path=Path("models/best_model.joblib"),
    )

    result = service.predict_one({"AMT_CREDIT": 100000.0, "UNUSED": 1})

    assert result.default_probability == 0.3
    assert result.risk_level == "medium"
    assert result.model_name == "dummy_pipeline"
    assert result.missing_features == ["APARTMENTS_MODE", "EXT_SOURCE_2"]
    assert result.extra_features == ["UNUSED"]


def test_predict_many_vectorizes_feature_alignment_and_prediction():
    service = CreditRiskModelService.from_model(
        model=DummyBatchModel(),
        metadata={"best_model": "dummy_batch"},
        model_path=Path("models/best_model.joblib"),
    )

    results = service.predict_many(
        [
            {"AMT_CREDIT": 100000.0, "AMT_INCOME_TOTAL": 50000.0, "UNUSED": 1},
            {"AMT_CREDIT": 300000.0, "AMT_INCOME_TOTAL": 100000.0, "UNUSED": 2},
        ]
    )

    assert [result.default_probability for result in results] == [0.1, 0.7]
    assert [result.risk_level for result in results] == ["low", "high"]
    assert [result.extra_features for result in results] == [["UNUSED"], ["UNUSED"]]


def test_missing_model_raises_actionable_error(tmp_path):
    service = CreditRiskModelService(model_path=tmp_path / "missing.joblib")

    try:
        service.predict_one({"AMT_CREDIT": 100000.0})
    except ModelArtifactMissingError as exc:
        assert "python3 -m src.modeling" in str(exc)
    else:
        raise AssertionError("Expected ModelArtifactMissingError")
