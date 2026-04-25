[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project - Home Credit Default Risk

**Студент:** [Карагюлян Армен Андраникович и Лепехов Александр Александрович]

**Группа:** [БИВ 232]

## Описание задачи

**Задача:** бинарная классификация кредитного риска.

**Что предсказываем:** вероятность того, что клиент Home Credit не вернёт кредит (`TARGET = 1`).

**Датасет:** Kaggle, [Home Credit Default Risk Feature Tools](https://www.kaggle.com/datasets/willkoehrsen/home-credit-default-risk-feature-tools?select=correlations.csv).

**Основная метрика:** ROC-AUC. Она подходит для несбалансированной классификации и совпадает с логикой Kaggle-соревнования. Дополнительно считаются Average Precision, F1, precision, recall и accuracy.

## Что такое CP1

CP1 в этом проекте закрывает две большие части:

- обработка и подготовка данных: описание источника, очистка, пропуски, дубли, выбросы, feature engineering, визуализации, корректный split и метрики;
- моделирование и эксперименты: baseline, несколько моделей, подбор гиперпараметров на уровне разумных конфигураций, уменьшение размерности и ансамбль.

## Данные

В папке `data/` лежат 9 CSV из Kaggle. Это не сырые таблицы Home Credit, а уже подготовленные Featuretools-файлы:

- `feature_matrix.csv`, `feature_matrix_advanced.csv`, `feature_matrix_article.csv`, `feature_matrix_spec.csv` - признаковые матрицы;
- `feature_importances.csv`, `fi_fma.csv`, `spec_feature_importances_ohe.csv` - важности признаков;
- `correlations.csv`, `correlations_spec.csv` - корреляционные матрицы для анализа, не обучающие данные.

Для основного CP1-пайплайна используется `data/feature_matrix_spec.csv`: это более компактная матрица с `356255` строками и `885` столбцами. Строки с `TARGET = -999` являются Kaggle test set без разметки, поэтому они исключаются из supervised-обучения.

## Структура репозитория

```text
.
├── data                         # CSV из Kaggle
├── models                       # Результаты экспериментов и best_model.joblib
├── presentation                 # Материалы для защиты
├── report
│   ├── data_quality             # Таблицы качества данных
│   ├── images                   # EDA-графики
│   ├── data_quality_report.md   # Data quality report
│   └── report.md                # Отчёт CP1
├── src
│   ├── data_quality.py          # Анализ пропусков, типов, дублей, выбросов
│   ├── eda.py                   # Генерация графиков
│   ├── modeling.py              # Baseline, модели, ансамбль, метрики
│   └── preprocessing.py         # Очистка, split, feature engineering
├── tests
│   └── test_pipeline.py         # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Быстрый smoke-run на небольшом сэмпле:

```bash
python -m src.data_quality
python -m src.eda --sample-size 5000 --top-n-features 40
python -m src.modeling --sample-size 5000 --top-n-features 40 --quick
```

Основной запуск CP1:

```bash
python -m src.data_quality --top-n-features 120
python -m src.eda --sample-size 50000 --top-n-features 120
python -m src.modeling --sample-size 50000 --top-n-features 120
```

## Результаты

После запуска `src.modeling` таблица экспериментов сохраняется в `models/experiment_results.csv`, а лучшая модель - в `models/best_model.joblib`.

Data quality report сохраняется в [`report/data_quality_report.md`](report/data_quality_report.md), а подробные таблицы - в `report/data_quality/`.

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
