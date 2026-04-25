# Отчёт CP1: Home Credit Default Risk

**Студент:** [ФИО / Student ID]
**Группа:** [Группа]

## 1. Введение и постановка задачи

Цель проекта - построить модель, которая по анкетным, кредитным и историческим признакам клиента оценивает риск дефолта по кредиту. Целевая переменная `TARGET`: `1` означает дефолт, `0` - отсутствие дефолта.

Это задача бинарной классификации. Практическая ценность: такая модель помогает кредитной организации ранжировать заявки по риску и принимать более аккуратные решения.

Основная метрика - ROC-AUC. Причины выбора:

- классы сильно несбалансированы: среди размеченных строк `282686` клиентов без дефолта и `24825` клиентов с дефолтом;
- ROC-AUC оценивает качество ранжирования вероятностей, а не только фиксированный порог `0.5`;
- эта метрика использовалась в оригинальном Kaggle-соревновании Home Credit.

Дополнительно считаются Average Precision, F1, precision, recall и accuracy. При выборе финальной модели приоритет отдаётся ROC-AUC, затем Average Precision.

## 2. Поиск и описание данных

Источник данных - Kaggle: [Home Credit Default Risk Feature Tools](https://www.kaggle.com/datasets/willkoehrsen/home-credit-default-risk-feature-tools?select=correlations.csv). Датасет выбран, потому что он относится к реальной tabular ML-задаче кредитного скоринга, содержит много признаков, дисбаланс классов, пропуски и уже рассчитанные Featuretools-признаки.

В папке `data/` находится 9 CSV:

| Файл | Назначение |
|---|---|
| `feature_matrix.csv` | основная Featuretools-матрица, 356255 строк и 1698 столбцов |
| `feature_matrix_advanced.csv` | расширенная матрица, 356255 строк и 3596 столбцов |
| `feature_matrix_article.csv` | матрица из статьи/примера, 1821 столбец |
| `feature_matrix_spec.csv` | компактная матрица для основного пайплайна, 356255 строк и 885 столбцов |
| `feature_importances.csv`, `fi_fma.csv`, `spec_feature_importances_ohe.csv` | важности признаков |
| `correlations.csv`, `correlations_spec.csv` | корреляционные матрицы для анализа |

Для обучения используется `feature_matrix_spec.csv`, потому что он меньше остальных feature matrix и подходит для воспроизводимого запуска на обычной машине. В нём есть `48744` строк с `TARGET = -999`: это Kaggle test set без разметки, поэтому он не используется в supervised-обучении.

## 3. Обработка и подготовка данных

Очистка реализована в `src/preprocessing.py`.

Дополнительно реализован отдельный data quality report: [`report/data_quality_report.md`](data_quality_report.md). Он фиксирует фактическое качество основного датасета до моделирования:

- всего строк: `356255`;
- всего колонок: `885`;
- размеченных строк для обучения: `307511`;
- Kaggle test rows без разметки (`TARGET = -999`): `48744`;
- дубликатов по `SK_ID_CURR`: `0`;
- колонок с пропусками: `823`;
- колонок с >= 70% пропусков: `302`;
- колонок с >= 90% пропусков: `6`;
- выбранных top-N признаков для пайплайна: `120`.

Что сделано:

- строки с `TARGET = -999` и `NaN` удаляются из обучающего набора;
- `TARGET` приводится к бинарному `int`;
- дубликаты удаляются по `SK_ID_CURR`;
- служебные колонки вроде `Unnamed: 0` исключаются;
- чтение данных ограничено top-N признаками по `spec_feature_importances_ohe.csv`, чтобы эксперимент был воспроизводимым;
- пропуски обрабатываются внутри sklearn pipeline: медиана для числовых признаков, `missing` для категориальных;
- категориальные признаки кодируются через OneHotEncoder для линейных моделей и OrdinalEncoder для деревьев;
- выбросы в числовых признаках клипуются по 1% и 99% квантилям, причём границы считаются только на train split.

Артефакты data quality анализа сохранены в `report/data_quality/`:

- `missing_values.csv` - доля и количество пропусков по каждой колонке;
- `dtype_by_column.csv` и `dtype_counts.csv` - типы данных;
- `target_distribution.csv` - распределение целевой переменной;
- `outlier_summary.csv` - IQR-анализ выбросов по выбранным числовым признакам;
- `selected_features.csv` - признаки, выбранные по importance-файлу.

Manual feature engineering:

- `NEW_CREDIT_TO_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL`;
- `NEW_ANNUITY_TO_INCOME_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL`;
- `NEW_CREDIT_TERM = AMT_ANNUITY / AMT_CREDIT`;
- `NEW_GOODS_TO_CREDIT_RATIO = AMT_GOODS_PRICE / AMT_CREDIT`;
- `NEW_EMPLOYED_TO_BIRTH_RATIO = DAYS_EMPLOYED / DAYS_BIRTH`;
- `NEW_EXT_SOURCE_MEAN` и `NEW_EXT_SOURCE_STD` по `EXT_SOURCE_1/2/3`.

Split: используется стратифицированное разбиение `70% / 15% / 15%` на train/validation/test. Data leakage предотвращается тем, что imputer, encoder, scaler и outlier clipping обучаются только внутри sklearn pipeline на train-части.

Визуализации создаются командой `python -m src.eda` и сохраняются в `report/images/`:

- `target_distribution.png` - дисбаланс классов;
- `missingness_top20.png` - признаки с максимальной долей пропусков;
- `feature_importances_top20.png` - top feature importances из Kaggle-файла;
- `target_correlations_top20.png` - сильнейшие числовые корреляции с `TARGET`;
- `pca_projection.png` - PCA-проекция выбранных числовых признаков.

## 4. Baseline-модель

Baseline - `LogisticRegression` без manual feature engineering. Для неё используется стандартная предобработка: median imputation, StandardScaler для числовых признаков и OneHotEncoder для категориальных.

Smoke-run на `3000` строках и `30` признаках:

| Модель | Val ROC-AUC | Test ROC-AUC | Test AP | Test F1 |
|---|---:|---:|---:|---:|
| `baseline_logistic_regression` | 0.740 | 0.735 | 0.186 | 0.265 |

Эта модель задаёт точку отсчёта для более сложных экспериментов.

## 5. Эксперименты

Эксперименты реализованы в `src/modeling.py`. В коде есть быстрый режим `--quick` и основной запуск с большим sample size.

Таблица smoke-run (`sample_size=3000`, `top_n_features=30`, `--quick`):

| Модель | Идея | Test ROC-AUC | Test AP | Test F1 |
|---|---|---:|---:|---:|
| `baseline_logistic_regression` | простая линейная модель без новых фич | 0.735 | 0.186 | 0.265 |
| `extra_trees` | randomized tree ensemble | 0.728 | 0.183 | 0.272 |
| `soft_voting_ensemble` | усреднение RF, ExtraTrees и HGB | 0.721 | 0.163 | 0.000 |
| `random_forest` | bagging по деревьям | 0.715 | 0.165 | 0.227 |
| `svd_logistic_regression` | уменьшение размерности через TruncatedSVD | 0.695 | 0.159 | 0.225 |
| `logistic_regression_fe` | линейная модель с manual feature engineering | 0.685 | 0.153 | 0.244 |
| `hist_gradient_boosting` | boosting для tabular data | 0.678 | 0.131 | 0.000 |

Уменьшение размерности проверено двумя способами:

- `svd_logistic_regression` использует `TruncatedSVD(n_components=30)` после one-hot encoding;
- `pca_projection.png` визуализирует PCA на числовых признаках для EDA.

## 6. Финальная модель и интерпретируемость

По smoke-run лучший результат по test ROC-AUC показал `baseline_logistic_regression`: `0.735`. На таком маленьком сэмпле это не доказывает, что baseline будет лучшим на полном запуске, но показывает, что пайплайн работает и метрики сохраняются корректно.

Для защиты финальный запуск стоит делать так:

```bash
python -m src.eda --sample-size 50000 --top-n-features 120
python -m src.modeling --sample-size 50000 --top-n-features 120
```

Интерпретируемость обеспечивается через:

- исходный файл `spec_feature_importances_ohe.csv`;
- график `feature_importances_top20.png`;
- коэффициенты линейной модели можно извлечь из `models/best_model.joblib` после финального запуска.

## 7. Воспроизводимость

Воспроизводимость обеспечивается фиксированным `random_state = 42`, sklearn Pipeline, сохранением `models/experiment_results.csv`, `models/run_metadata.json` и `models/best_model.joblib`.

Проверки:

```bash
python -m pytest -q
python -m src.eda --sample-size 5000 --top-n-features 40
python -m src.modeling --sample-size 3000 --top-n-features 30 --quick
```

## 8. Заключение и выводы

В CP1 реализованы подготовка данных, feature engineering, визуализации, корректный split, baseline, несколько моделей, эксперимент с уменьшением размерности и ансамбль. Основной риск проекта - большой размер CSV, поэтому код читает только top-N признаков по importance-файлу и поддерживает sample size.

Самостоятельный парсинг данных в этой версии не заявляется: используются готовые CSV из Kaggle.
