"""Data loading and preprocessing utilities for the Home Credit CP1 project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"
TEST_TARGET_VALUES = {-999, -999.0}
RANDOM_STATE = 42


FEATURE_ENGINEERING_INPUTS = [
    "AMT_ANNUITY",
    "AMT_CREDIT",
    "AMT_GOODS_PRICE",
    "AMT_INCOME_TOTAL",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]


@dataclass(frozen=True)
class DataConfig:
    """Configuration for a reproducible supervised-learning dataset."""

    data_path: Path = Path("data/feature_matrix_spec.csv")
    importance_path: Path = Path("data/spec_feature_importances_ohe.csv")
    top_n_features: int = 120
    sample_size: int | None = 50_000
    random_state: int = RANDOM_STATE


@dataclass(frozen=True)
class DatasetBundle:
    """Train/validation/test split with feature matrices and labels."""

    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip numeric features by quantiles fitted on the training split only."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "OutlierClipper":
        x_df = pd.DataFrame(x).copy()
        self.numeric_columns_ = list(x_df.select_dtypes(include=np.number).columns)
        if self.numeric_columns_:
            numeric = x_df[self.numeric_columns_].replace([np.inf, -np.inf], np.nan)
            self.lower_bounds_ = numeric.quantile(self.lower_quantile)
            self.upper_bounds_ = numeric.quantile(self.upper_quantile)
        else:
            self.lower_bounds_ = pd.Series(dtype=float)
            self.upper_bounds_ = pd.Series(dtype=float)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x_df = pd.DataFrame(x).copy()
        for column in self.numeric_columns_:
            x_df[column] = x_df[column].clip(self.lower_bounds_[column], self.upper_bounds_[column])
        return x_df


def read_columns(path: str | Path) -> list[str]:
    """Read only the CSV header."""

    return list(pd.read_csv(path, nrows=0).columns)


def load_top_features(
    data_path: str | Path,
    importance_path: str | Path,
    top_n: int,
    extra_features: list[str] | None = None,
) -> list[str]:
    """Choose top features present in the feature matrix.

    The Kaggle archive contains separate feature-importance files. Reading only
    the most important columns keeps experiments reproducible on a laptop.
    """

    available_columns = set(read_columns(data_path))
    excluded = {TARGET_COLUMN, ID_COLUMN, "index", "Unnamed: 0"}
    extra_features = extra_features or []

    importances = pd.read_csv(importance_path)
    importances = importances.sort_values("importance", ascending=False)
    ranked = [
        feature
        for feature in importances["feature"].astype(str)
        if feature in available_columns and feature not in excluded
    ]

    selected = ranked[:top_n]
    for feature in extra_features:
        if feature in available_columns and feature not in selected and feature not in excluded:
            selected.append(feature)
    return selected


def load_training_frame(config: DataConfig) -> pd.DataFrame:
    """Load labelled rows from the selected feature matrix."""

    selected_features = load_top_features(
        data_path=config.data_path,
        importance_path=config.importance_path,
        top_n=config.top_n_features,
        extra_features=FEATURE_ENGINEERING_INPUTS,
    )
    usecols = [ID_COLUMN, TARGET_COLUMN, *selected_features]
    frame = pd.read_csv(config.data_path, usecols=lambda column: column in usecols)
    frame = frame.drop(columns=[column for column in frame.columns if column.startswith("Unnamed:")], errors="ignore")
    frame = clean_target(frame)
    frame = frame.drop_duplicates(subset=ID_COLUMN).reset_index(drop=True)

    if config.sample_size is not None and config.sample_size < len(frame):
        _, frame = train_test_split(
            frame,
            test_size=config.sample_size,
            stratify=frame[TARGET_COLUMN],
            random_state=config.random_state,
        )
        frame = frame.reset_index(drop=True)

    return frame


def clean_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove Kaggle test rows and cast target to binary integers."""

    frame = frame.copy()
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    labelled_mask = frame[TARGET_COLUMN].isin([0, 1]) & ~frame[TARGET_COLUMN].isin(TEST_TARGET_VALUES)
    frame = frame.loc[labelled_mask].copy()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
    return frame


def add_domain_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create transparent credit-risk ratios from existing columns."""

    frame = frame.copy()
    safe_income = frame.get("AMT_INCOME_TOTAL", pd.Series(np.nan, index=frame.index)).replace(0, np.nan)
    safe_credit = frame.get("AMT_CREDIT", pd.Series(np.nan, index=frame.index)).replace(0, np.nan)

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(frame.columns):
        frame["NEW_CREDIT_TO_INCOME_RATIO"] = frame["AMT_CREDIT"] / safe_income
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(frame.columns):
        frame["NEW_ANNUITY_TO_INCOME_RATIO"] = frame["AMT_ANNUITY"] / safe_income
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(frame.columns):
        frame["NEW_CREDIT_TERM"] = frame["AMT_ANNUITY"] / safe_credit
    if {"AMT_GOODS_PRICE", "AMT_CREDIT"}.issubset(frame.columns):
        frame["NEW_GOODS_TO_CREDIT_RATIO"] = frame["AMT_GOODS_PRICE"] / safe_credit
    if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(frame.columns):
        frame["NEW_EMPLOYED_TO_BIRTH_RATIO"] = frame["DAYS_EMPLOYED"] / frame["DAYS_BIRTH"].replace(0, np.nan)

    ext_sources = [column for column in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if column in frame.columns]
    if ext_sources:
        frame["NEW_EXT_SOURCE_MEAN"] = frame[ext_sources].mean(axis=1)
        frame["NEW_EXT_SOURCE_STD"] = frame[ext_sources].std(axis=1)

    return frame.replace([np.inf, -np.inf], np.nan)


def split_dataset(
    frame: pd.DataFrame,
    *,
    add_features: bool,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> DatasetBundle:
    """Make a stratified train/validation/test split."""

    working = add_domain_features(frame) if add_features else frame.copy()
    y = working[TARGET_COLUMN].astype(int)
    x = working.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    val_fraction = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_fraction,
        stratify=y_train_val,
        random_state=random_state,
    )
    return DatasetBundle(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def infer_feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical columns for sklearn transformers."""

    numeric_columns = list(frame.select_dtypes(include=np.number).columns)
    categorical_columns = [column for column in frame.columns if column not in numeric_columns]
    return numeric_columns, categorical_columns
