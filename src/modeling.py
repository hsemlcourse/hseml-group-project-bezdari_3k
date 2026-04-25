"""Model experiments for CP1."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.preprocessing import DataConfig, OutlierClipper, infer_feature_types, load_training_frame, split_dataset


@dataclass(frozen=True)
class ModelSpec:
    """Experiment configuration."""

    name: str
    estimator: object
    hypothesis: str
    encoding: str
    add_features: bool = True
    use_svd: bool = False


def build_preprocessor(x_train: pd.DataFrame, encoding: str) -> ColumnTransformer:
    """Build preprocessing fitted only inside each sklearn pipeline."""

    numeric_columns, categorical_columns = infer_feature_types(x_train)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if encoding == "onehot":
        numeric_steps.append(("scaler", StandardScaler()))

    if encoding == "onehot":
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)),
            ]
        )
    elif encoding == "ordinal":
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        )
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ],
        remainder="drop",
    )


def make_pipeline(spec: ModelSpec, x_train: pd.DataFrame) -> Pipeline:
    """Create a full model pipeline."""

    steps: list[tuple[str, object]] = [
        ("clipper", OutlierClipper()),
        ("preprocessor", build_preprocessor(x_train, spec.encoding)),
    ]
    if spec.use_svd:
        steps.append(("svd", TruncatedSVD(n_components=30, random_state=42)))
    steps.append(("model", spec.estimator))
    return Pipeline(steps=steps)


def get_model_specs(random_state: int, quick: bool) -> list[ModelSpec]:
    """Return baseline, classical models, dimensionality reduction and an ensemble."""

    rf_estimators = 40 if quick else 140
    hgb_iterations = 60 if quick else 180

    tree_rf = RandomForestClassifier(
        n_estimators=rf_estimators,
        max_depth=12,
        min_samples_leaf=25,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=1,
    )
    tree_extra = ExtraTreesClassifier(
        n_estimators=rf_estimators,
        max_depth=14,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )
    tree_hgb = HistGradientBoostingClassifier(
        max_iter=hgb_iterations,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=random_state,
    )

    return [
        ModelSpec(
            name="baseline_logistic_regression",
            estimator=LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                solver="saga",
                n_jobs=1,
                random_state=random_state,
            ),
            hypothesis="Simple baseline without manual feature engineering.",
            encoding="onehot",
            add_features=False,
        ),
        ModelSpec(
            name="logistic_regression_fe",
            estimator=LogisticRegression(
                max_iter=600,
                class_weight="balanced",
                solver="saga",
                n_jobs=1,
                random_state=random_state,
            ),
            hypothesis="Credit ratios and external-score aggregates improve a linear model.",
            encoding="onehot",
        ),
        ModelSpec(
            name="random_forest",
            estimator=tree_rf,
            hypothesis="Bagging over nonlinear trees captures interactions and is robust to outliers.",
            encoding="ordinal",
        ),
        ModelSpec(
            name="extra_trees",
            estimator=ExtraTreesClassifier(
                n_estimators=rf_estimators,
                max_depth=14,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
            hypothesis="More randomized trees can reduce variance versus RandomForest.",
            encoding="ordinal",
        ),
        ModelSpec(
            name="hist_gradient_boosting",
            estimator=tree_hgb,
            hypothesis="Boosting should perform best on tabular nonlinear risk data.",
            encoding="ordinal",
        ),
        ModelSpec(
            name="svd_logistic_regression",
            estimator=LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                solver="lbfgs",
                random_state=random_state,
            ),
            hypothesis="Dimensionality reduction may keep signal while reducing noisy sparse features.",
            encoding="onehot",
            use_svd=True,
        ),
        ModelSpec(
            name="soft_voting_ensemble",
            estimator=VotingClassifier(
                estimators=[
                    ("rf", tree_rf),
                    ("extra", tree_extra),
                    ("hgb", tree_hgb),
                ],
                voting="soft",
                n_jobs=1,
            ),
            hypothesis="Averaging linear, bagging and boosting probabilities can improve stability.",
            encoding="ordinal",
        ),
    ]


def evaluate_predictions(y_true: pd.Series, probabilities: pd.Series) -> dict[str, float]:
    """Calculate metrics for an imbalanced binary classification problem."""

    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "accuracy": accuracy_score(y_true, predictions),
    }


def run_experiments(
    config: DataConfig,
    output_dir: Path,
    quick: bool = False,
    selected_models: list[str] | None = None,
) -> pd.DataFrame:
    """Fit all experiments and save the result table plus the best model."""

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_training_frame(config)
    raw_bundle = split_dataset(frame, add_features=False, random_state=config.random_state)
    fe_bundle = split_dataset(frame, add_features=True, random_state=config.random_state)

    rows: list[dict[str, object]] = []
    best_auc = -1.0
    best_pipeline: Pipeline | None = None
    best_name = ""

    for spec in get_model_specs(config.random_state, quick=quick):
        if selected_models and spec.name not in selected_models:
            continue

        bundle = fe_bundle if spec.add_features else raw_bundle
        pipeline = make_pipeline(spec, bundle.x_train)
        pipeline.fit(bundle.x_train, bundle.y_train)

        val_probabilities = pipeline.predict_proba(bundle.x_val)[:, 1]
        test_probabilities = pipeline.predict_proba(bundle.x_test)[:, 1]
        val_metrics = evaluate_predictions(bundle.y_val, pd.Series(val_probabilities, index=bundle.y_val.index))
        test_metrics = evaluate_predictions(bundle.y_test, pd.Series(test_probabilities, index=bundle.y_test.index))

        row = {
            "model": spec.name,
            "hypothesis": spec.hypothesis,
            "encoding": spec.encoding,
            "manual_feature_engineering": spec.add_features,
            "dimensionality_reduction": spec.use_svd,
            "val_roc_auc": val_metrics["roc_auc"],
            "val_average_precision": val_metrics["average_precision"],
            "val_f1": val_metrics["f1"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_average_precision": test_metrics["average_precision"],
            "test_f1": test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_accuracy": test_metrics["accuracy"],
        }
        rows.append(row)

        if test_metrics["roc_auc"] > best_auc:
            best_auc = test_metrics["roc_auc"]
            best_pipeline = pipeline
            best_name = spec.name

    results = pd.DataFrame(rows).sort_values("test_roc_auc", ascending=False)
    results_path = output_dir / "experiment_results.csv"
    results.to_csv(results_path, index=False)

    metadata = {
        "data_path": str(config.data_path),
        "importance_path": str(config.importance_path),
        "top_n_features": config.top_n_features,
        "sample_size": config.sample_size,
        "random_state": config.random_state,
        "best_model": best_name,
        "best_test_roc_auc": best_auc,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if best_pipeline is not None:
        joblib.dump(best_pipeline, output_dir / "best_model.joblib")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CP1 Home Credit model experiments.")
    parser.add_argument("--data-path", type=Path, default=Path("data/feature_matrix_spec.csv"))
    parser.add_argument("--importance-path", type=Path, default=Path("data/spec_feature_importances_ohe.csv"))
    parser.add_argument("--top-n-features", type=int, default=120)
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--quick", action="store_true", help="Use fewer tree iterations for a faster smoke run.")
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of model names.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DataConfig(
        data_path=args.data_path,
        importance_path=args.importance_path,
        top_n_features=args.top_n_features,
        sample_size=args.sample_size,
    )
    results = run_experiments(config=config, output_dir=args.output_dir, quick=args.quick, selected_models=args.models)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
