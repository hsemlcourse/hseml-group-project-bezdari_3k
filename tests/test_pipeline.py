import numpy as np
import pandas as pd

from src.preprocessing import (
    TARGET_COLUMN,
    DataConfig,
    add_domain_features,
    clean_target,
    load_top_features,
    load_training_frame,
    split_dataset,
)


def test_clean_target_removes_kaggle_test_rows():
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4, 5],
            TARGET_COLUMN: [0, 1, -999, np.nan, 0],
            "AMT_CREDIT": [100, 200, 300, 400, 500],
        }
    )

    cleaned = clean_target(frame)

    assert cleaned[TARGET_COLUMN].tolist() == [0, 1, 0]
    assert cleaned[TARGET_COLUMN].dtype == int


def test_add_domain_features_creates_credit_ratios():
    frame = pd.DataFrame(
        {
            "AMT_CREDIT": [100_000.0],
            "AMT_ANNUITY": [10_000.0],
            "AMT_INCOME_TOTAL": [50_000.0],
            "AMT_GOODS_PRICE": [90_000.0],
            "DAYS_EMPLOYED": [-1_000.0],
            "DAYS_BIRTH": [-10_000.0],
            "EXT_SOURCE_1": [0.2],
            "EXT_SOURCE_2": [0.4],
            "EXT_SOURCE_3": [0.6],
        }
    )

    engineered = add_domain_features(frame)

    assert engineered.loc[0, "NEW_CREDIT_TO_INCOME_RATIO"] == 2.0
    assert engineered.loc[0, "NEW_ANNUITY_TO_INCOME_RATIO"] == 0.2
    assert np.isclose(engineered.loc[0, "NEW_EXT_SOURCE_MEAN"], 0.4)


def test_split_dataset_is_stratified_and_disjoint():
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": range(100),
            TARGET_COLUMN: [0] * 80 + [1] * 20,
            "AMT_CREDIT": np.linspace(100_000, 500_000, 100),
            "AMT_INCOME_TOTAL": np.linspace(50_000, 150_000, 100),
        }
    )

    bundle = split_dataset(frame, add_features=True, random_state=42)

    assert len(bundle.x_train) + len(bundle.x_val) + len(bundle.x_test) == len(frame)
    assert abs(len(bundle.x_test) / len(frame) - 0.15) <= 0.01
    assert abs(len(bundle.x_val) / len(frame) - 0.15) <= 0.02
    assert set(bundle.x_train.index).isdisjoint(bundle.x_val.index)
    assert set(bundle.x_train.index).isdisjoint(bundle.x_test.index)
    assert "NEW_CREDIT_TO_INCOME_RATIO" in bundle.x_train.columns


def test_load_top_features_maps_one_hot_importance_to_raw_categorical_column(tmp_path):
    data_path = tmp_path / "feature_matrix.csv"
    importance_path = tmp_path / "feature_importances.csv"

    pd.DataFrame(
        columns=[
            "SK_ID_CURR",
            TARGET_COLUMN,
            "NAME_EDUCATION_TYPE",
            "AMT_CREDIT",
            "EXT_SOURCE_2",
        ]
    ).to_csv(data_path, index=False)
    pd.DataFrame(
        {
            "feature": [
                "NAME_EDUCATION_TYPE_Higher education",
                "AMT_CREDIT",
                "EXT_SOURCE_2",
            ],
            "importance": [100.0, 50.0, 25.0],
        }
    ).to_csv(importance_path, index=False)

    selected = load_top_features(data_path=data_path, importance_path=importance_path, top_n=2)

    assert selected == ["NAME_EDUCATION_TYPE", "AMT_CREDIT"]


def test_load_training_frame_keeps_interactive_ui_fields_even_when_not_top_ranked(tmp_path):
    data_path = tmp_path / "feature_matrix.csv"
    importance_path = tmp_path / "feature_importances.csv"

    pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            TARGET_COLUMN: [0, 1],
            "AMT_CREDIT": [100000.0, 200000.0],
            "NAME_INCOME_TYPE": ["Working", "Pensioner"],
            "OCCUPATION_TYPE": ["Core staff", "Laborers"],
            "CNT_CHILDREN": [0, 1],
            "CNT_FAM_MEMBERS": [2, 3],
        }
    ).to_csv(data_path, index=False)
    pd.DataFrame({"feature": ["AMT_CREDIT"], "importance": [100.0]}).to_csv(importance_path, index=False)

    frame = load_training_frame(
        DataConfig(
            data_path=data_path,
            importance_path=importance_path,
            top_n_features=1,
            sample_size=None,
        )
    )

    assert "NAME_INCOME_TYPE" in frame.columns
    assert "OCCUPATION_TYPE" in frame.columns
    assert "CNT_CHILDREN" in frame.columns
    assert "CNT_FAM_MEMBERS" in frame.columns
