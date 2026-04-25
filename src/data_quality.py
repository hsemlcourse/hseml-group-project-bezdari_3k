"""Generate a data quality report for the CP1 dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import ID_COLUMN, TARGET_COLUMN, clean_target, load_top_features


@dataclass(frozen=True)
class DataQualityConfig:
    """Configuration for a reproducible data quality report."""

    data_path: Path = Path("data/feature_matrix_spec.csv")
    importance_path: Path = Path("data/spec_feature_importances_ohe.csv")
    output_dir: Path = Path("report/data_quality")
    report_path: Path = Path("report/data_quality_report.md")
    top_n_features: int = 120
    sample_rows_for_dtypes: int = 20_000
    chunksize: int = 10_000


def count_csv_rows(path: Path) -> int:
    """Count data rows without loading the full CSV into memory."""

    with path.open("rb") as file:
        return max(sum(1 for _ in file) - 1, 0)


def get_target_summary(path: Path) -> pd.DataFrame:
    target = pd.read_csv(path, usecols=[TARGET_COLUMN])[TARGET_COLUMN]
    summary = target.value_counts(dropna=False).rename_axis(TARGET_COLUMN).reset_index(name="rows")
    summary["share"] = summary["rows"] / len(target)
    summary[TARGET_COLUMN] = summary[TARGET_COLUMN].astype(str)
    return summary


def get_duplicate_summary(path: Path) -> dict[str, int]:
    ids = pd.read_csv(path, usecols=[ID_COLUMN])[ID_COLUMN]
    return {
        "rows": int(len(ids)),
        "unique_sk_id_curr": int(ids.nunique(dropna=True)),
        "duplicate_sk_id_curr_rows": int(ids.duplicated().sum()),
        "missing_sk_id_curr": int(ids.isna().sum()),
    }


def get_dtype_summary(path: Path, nrows: int) -> tuple[pd.DataFrame, pd.Series]:
    sample = pd.read_csv(path, nrows=nrows)
    dtype_by_column = sample.dtypes.astype(str).rename("dtype").reset_index().rename(columns={"index": "feature"})
    dtype_counts = sample.dtypes.astype(str).value_counts().rename_axis("dtype").reset_index(name="columns")
    return dtype_by_column, dtype_counts


def get_missing_profile(path: Path, chunksize: int) -> pd.DataFrame:
    missing_counts: pd.Series | None = None
    total_rows = 0

    for chunk in pd.read_csv(path, chunksize=chunksize):
        total_rows += len(chunk)
        chunk_missing = chunk.isna().sum()
        if missing_counts is None:
            missing_counts = chunk_missing
        else:
            missing_counts = missing_counts.add(chunk_missing, fill_value=0)

    if missing_counts is None:
        return pd.DataFrame(columns=["feature", "missing_count", "missing_share"])

    profile = missing_counts.astype(int).rename("missing_count").reset_index().rename(columns={"index": "feature"})
    profile["missing_share"] = profile["missing_count"] / total_rows
    return profile.sort_values(["missing_share", "missing_count"], ascending=False).reset_index(drop=True)


def get_selected_features(config: DataQualityConfig) -> pd.DataFrame:
    features = load_top_features(
        data_path=config.data_path,
        importance_path=config.importance_path,
        top_n=config.top_n_features,
    )
    importances = pd.read_csv(config.importance_path)
    selected = pd.DataFrame({"feature": features})
    selected = selected.merge(importances[["feature", "importance"]], on="feature", how="left")
    selected.insert(0, "rank", range(1, len(selected) + 1))
    return selected


def get_outlier_profile(config: DataQualityConfig, selected_features: list[str]) -> pd.DataFrame:
    usecols = [ID_COLUMN, TARGET_COLUMN, *selected_features]
    frame = pd.read_csv(config.data_path, usecols=lambda column: column in usecols)
    frame = clean_target(frame)
    numeric = frame.drop(columns=[ID_COLUMN, TARGET_COLUMN], errors="ignore").select_dtypes(include=np.number)

    rows: list[dict[str, float | int | str]] = []
    for column in numeric.columns:
        series = numeric[column].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr
        outlier_count = int(((series < lower_iqr) | (series > upper_iqr)).sum())
        rows.append(
            {
                "feature": column,
                "non_missing_count": int(series.shape[0]),
                "missing_count": int(numeric[column].isna().sum()),
                "min": float(series.min()),
                "q01": float(series.quantile(0.01)),
                "q25": float(q1),
                "median": float(series.median()),
                "q75": float(q3),
                "q99": float(series.quantile(0.99)),
                "max": float(series.max()),
                "iqr_outlier_count": outlier_count,
                "iqr_outlier_share": outlier_count / len(series),
            }
        )
    return pd.DataFrame(rows).sort_values("iqr_outlier_share", ascending=False)


def make_markdown_table(frame: pd.DataFrame, max_rows: int = 10) -> str:
    display = frame.head(max_rows).copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}")

    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    table = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)


def write_report(
    config: DataQualityConfig,
    metadata: dict[str, int | float | str],
    target_summary: pd.DataFrame,
    duplicate_summary: dict[str, int],
    dtype_counts: pd.DataFrame,
    missing_profile: pd.DataFrame,
    selected_features: pd.DataFrame,
    outlier_profile: pd.DataFrame,
) -> None:
    labelled_rows = int(target_summary.loc[target_summary[TARGET_COLUMN].isin(["0", "1"]), "rows"].sum())
    kaggle_test_rows = int(target_summary.loc[target_summary[TARGET_COLUMN].isin(["-999", "-999.0"]), "rows"].sum())
    high_missing_90 = int((missing_profile["missing_share"] >= 0.90).sum())
    high_missing_70 = int((missing_profile["missing_share"] >= 0.70).sum())
    columns_with_missing = int((missing_profile["missing_count"] > 0).sum())

    lines = [
        "# Data Quality Report",
        "",
        f"Основной файл анализа: `{config.data_path}`.",
        "",
        "## Краткий Вывод",
        "",
        f"- Всего строк: `{metadata['rows']}`.",
        f"- Всего колонок: `{metadata['columns']}`.",
        f"- Размеченных строк для обучения (`TARGET` 0/1): `{labelled_rows}`.",
        f"- Kaggle test rows без разметки (`TARGET = -999`): `{kaggle_test_rows}`.",
        f"- Дубликатов по `SK_ID_CURR`: `{duplicate_summary['duplicate_sk_id_curr_rows']}`.",
        f"- Колонок с пропусками: `{columns_with_missing}`.",
        f"- Колонок с >= 70% пропусков: `{high_missing_70}`.",
        f"- Колонок с >= 90% пропусков: `{high_missing_90}`.",
        f"- В модельном пайплайне выбрано top-N признаков: `{len(selected_features)}`.",
        "",
        "## TARGET",
        "",
        make_markdown_table(target_summary, max_rows=10),
        "",
        "## Типы Данных",
        "",
        make_markdown_table(dtype_counts, max_rows=10),
        "",
        "## Дубликаты И Идентификаторы",
        "",
        make_markdown_table(pd.DataFrame([duplicate_summary]), max_rows=1),
        "",
        "## Пропуски",
        "",
        "Таблица ниже показывает признаки с максимальной долей пропусков. Полная таблица сохранена в "
        f"`{config.output_dir / 'missing_values.csv'}`.",
        "",
        make_markdown_table(missing_profile, max_rows=20),
        "",
        "## Выбросы",
        "",
        "Выбросы оценены IQR-правилом на выбранных числовых признаках после удаления Kaggle test rows. "
        "В самом пайплайне значения клипуются по 1% и 99% квантилям, причём границы считаются только на train split.",
        "",
        make_markdown_table(outlier_profile, max_rows=20),
        "",
        "## Отбор Признаков",
        "",
        "Для воспроизводимости и экономии памяти модель читает top-N признаков из "
        f"`{config.importance_path}`. Полная таблица выбранных признаков сохранена в "
        f"`{config.output_dir / 'selected_features.csv'}`.",
        "",
        make_markdown_table(selected_features, max_rows=20),
        "",
        "## Что Делает Очистка В Пайплайне",
        "",
        "- Удаляет строки `TARGET = -999` и `TARGET = NaN`, потому что это test rows без ответа.",
        "- Приводит `TARGET` к бинарному `int`.",
        "- Удаляет дубликаты по `SK_ID_CURR`.",
        "- Исключает служебные колонки вроде `Unnamed: 0`.",
        "- Заполняет числовые пропуски медианой, категориальные - значением `missing`.",
        "- Кодирует категориальные признаки через OneHotEncoder или OrdinalEncoder в зависимости от модели.",
        "- Клипует числовые выбросы по квантилям внутри sklearn Pipeline, чтобы не было data leakage.",
        "",
    ]

    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text("\n".join(lines), encoding="utf-8")


def generate_data_quality_report(config: DataQualityConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    columns = list(pd.read_csv(config.data_path, nrows=0).columns)
    metadata = {
        "data_path": str(config.data_path),
        "importance_path": str(config.importance_path),
        "rows": count_csv_rows(config.data_path),
        "columns": len(columns),
        "top_n_features": config.top_n_features,
    }

    target_summary = get_target_summary(config.data_path)
    duplicate_summary = get_duplicate_summary(config.data_path)
    dtype_by_column, dtype_counts = get_dtype_summary(config.data_path, config.sample_rows_for_dtypes)
    missing_profile = get_missing_profile(config.data_path, config.chunksize)
    selected_features = get_selected_features(config)
    outlier_profile = get_outlier_profile(config, selected_features["feature"].tolist())

    target_summary.to_csv(config.output_dir / "target_distribution.csv", index=False)
    dtype_by_column.to_csv(config.output_dir / "dtype_by_column.csv", index=False)
    dtype_counts.to_csv(config.output_dir / "dtype_counts.csv", index=False)
    missing_profile.to_csv(config.output_dir / "missing_values.csv", index=False)
    selected_features.to_csv(config.output_dir / "selected_features.csv", index=False)
    outlier_profile.to_csv(config.output_dir / "outlier_summary.csv", index=False)
    (config.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    write_report(
        config=config,
        metadata=metadata,
        target_summary=target_summary,
        duplicate_summary=duplicate_summary,
        dtype_counts=dtype_counts,
        missing_profile=missing_profile,
        selected_features=selected_features,
        outlier_profile=outlier_profile,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a CP1 data quality report.")
    parser.add_argument("--data-path", type=Path, default=Path("data/feature_matrix_spec.csv"))
    parser.add_argument("--importance-path", type=Path, default=Path("data/spec_feature_importances_ohe.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("report/data_quality"))
    parser.add_argument("--report-path", type=Path, default=Path("report/data_quality_report.md"))
    parser.add_argument("--top-n-features", type=int, default=120)
    parser.add_argument("--sample-rows-for-dtypes", type=int, default=20_000)
    parser.add_argument("--chunksize", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DataQualityConfig(
        data_path=args.data_path,
        importance_path=args.importance_path,
        output_dir=args.output_dir,
        report_path=args.report_path,
        top_n_features=args.top_n_features,
        sample_rows_for_dtypes=args.sample_rows_for_dtypes,
        chunksize=args.chunksize,
    )
    generate_data_quality_report(config)
    print(f"Data quality report saved to {config.report_path}")


if __name__ == "__main__":
    main()
