"""Exploratory plots for the CP1 report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import DataConfig, TARGET_COLUMN, add_domain_features, load_training_frame


def save_target_distribution(frame: pd.DataFrame, output_dir: Path) -> None:
    counts = frame[TARGET_COLUMN].value_counts().sort_index()
    labels = ["no default", "default"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, counts.values, color=["#4c78a8", "#f58518"])
    ax.set_title("Target distribution")
    ax.set_ylabel("Rows")
    for index, value in enumerate(counts.values):
        ax.text(index, value, f"{value:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "target_distribution.png", dpi=160)
    plt.close(fig)


def save_missingness(frame: pd.DataFrame, output_dir: Path) -> None:
    missing = frame.drop(columns=[TARGET_COLUMN], errors="ignore").isna().mean().sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(missing.index[::-1], missing.values[::-1], color="#72b7b2")
    ax.set_title("Top missing-value rates")
    ax.set_xlabel("Missing fraction")
    fig.tight_layout()
    fig.savefig(output_dir / "missingness_top20.png", dpi=160)
    plt.close(fig)


def save_feature_importances(importance_path: Path, output_dir: Path, top_n: int = 20) -> None:
    importances = pd.read_csv(importance_path).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importances["feature"][::-1], importances["importance"][::-1], color="#54a24b")
    ax.set_title("Top feature importances from Kaggle Featuretools matrix")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importances_top20.png", dpi=160)
    plt.close(fig)


def save_target_correlations(frame: pd.DataFrame, output_dir: Path, top_n: int = 20) -> None:
    numeric = frame.select_dtypes(include="number")
    correlations = numeric.corr(numeric_only=True)[TARGET_COLUMN].drop(TARGET_COLUMN).dropna()
    top = correlations.reindex(correlations.abs().sort_values(ascending=False).head(top_n).index)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#e45756" if value < 0 else "#4c78a8" for value in top[::-1]]
    ax.barh(top.index[::-1], top.values[::-1], color=colors)
    ax.set_title("Strongest numeric correlations with TARGET")
    ax.set_xlabel("Pearson correlation")
    fig.tight_layout()
    fig.savefig(output_dir / "target_correlations_top20.png", dpi=160)
    plt.close(fig)


def save_pca_projection(frame: pd.DataFrame, output_dir: Path, sample_size: int = 5_000) -> None:
    sample = frame.sample(n=min(sample_size, len(frame)), random_state=42)
    y = sample[TARGET_COLUMN]
    x_numeric = sample.drop(columns=[TARGET_COLUMN], errors="ignore").select_dtypes(include="number")

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=42)),
        ]
    )
    components = pipeline.fit_transform(x_numeric)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(components[:, 0], components[:, 1], c=y, cmap="coolwarm", s=8, alpha=0.55)
    ax.set_title("PCA projection of selected numeric features")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="TARGET")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_projection.png", dpi=160)
    plt.close(fig)


def create_eda_artifacts(config: DataConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_training_frame(config)
    engineered = add_domain_features(frame)

    save_target_distribution(frame, output_dir)
    save_missingness(engineered, output_dir)
    save_feature_importances(config.importance_path, output_dir)
    save_target_correlations(engineered, output_dir)
    save_pca_projection(engineered, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CP1 EDA plots.")
    parser.add_argument("--data-path", type=Path, default=Path("data/feature_matrix_spec.csv"))
    parser.add_argument("--importance-path", type=Path, default=Path("data/spec_feature_importances_ohe.csv"))
    parser.add_argument("--top-n-features", type=int, default=120)
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--output-dir", type=Path, default=Path("report/images"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DataConfig(
        data_path=args.data_path,
        importance_path=args.importance_path,
        top_n_features=args.top_n_features,
        sample_size=args.sample_size,
    )
    create_eda_artifacts(config=config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
