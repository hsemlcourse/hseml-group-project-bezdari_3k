import numpy as np
import pandas as pd

from src.preprocessing import TARGET_COLUMN, add_domain_features, clean_target, split_dataset


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
