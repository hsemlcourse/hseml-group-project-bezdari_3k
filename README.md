[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project - Home Credit Default Risk

**Студенты:** Карагюлян Армен Андраникович и Лепехов Александр Александрович

**Группа:** БИВ 232

## Описание задачи

**Задача:** бинарная классификация кредитного риска.

**Что предсказываем:** вероятность того, что клиент Home Credit не вернёт кредит (`TARGET = 1`).

**Датасет:** Kaggle, [Home Credit Default Risk Feature Tools](https://www.kaggle.com/datasets/willkoehrsen/home-credit-default-risk-feature-tools?select=correlations.csv).

**Основная метрика:** ROC-AUC. Она подходит для несбалансированной классификации и совпадает с логикой Kaggle-соревнования. Дополнительно считаются Average Precision, F1, precision, recall и accuracy.

## Что закрывает проект

Проект закрывает полный пайплайн CP1-CP3:

- обработка и подготовка данных: описание источника, очистка, пропуски, дубли, выбросы, feature engineering, визуализации, корректный split и метрики;
- моделирование и эксперименты: baseline, несколько моделей, подбор гиперпараметров на уровне разумных конфигураций, уменьшение размерности и ансамбль;
- deploy: FastAPI для HTTP-запросов к модели и Streamlit scoring dashboard для аналитика банка.

## Данные

В папке `data/` лежат 9 CSV из Kaggle. Это не сырые таблицы Home Credit, а уже подготовленные Featuretools-файлы:

- `feature_matrix.csv`, `feature_matrix_advanced.csv`, `feature_matrix_article.csv`, `feature_matrix_spec.csv` - признаковые матрицы;
- `feature_importances.csv`, `fi_fma.csv`, `spec_feature_importances_ohe.csv` - важности признаков;
- `correlations.csv`, `correlations_spec.csv` - корреляционные матрицы для анализа, не обучающие данные.

Для основного пайплайна используется `data/feature_matrix_spec.csv`: это более компактная матрица с `356255` строками и `885` столбцами. Строки с `TARGET = -999` являются Kaggle test set без разметки, поэтому они исключаются из supervised-обучения.

### Загрузка данных Kaggle / FeatureTools

Большие CSV не коммитятся в репозиторий. Для воспроизведения нужно скачать Kaggle dataset
`willkoehrsen/home-credit-default-risk-feature-tools` и положить файлы в `data/`.

Вариант через `kagglehub`:

```bash
python -m pip install -r requirements.txt
python -c "import kagglehub, pathlib, shutil; src = pathlib.Path(kagglehub.dataset_download('willkoehrsen/home-credit-default-risk-feature-tools')); dst = pathlib.Path('data'); dst.mkdir(exist_ok=True); [shutil.copy2(p, dst / p.name) for p in src.glob('*.csv')]"
```

Минимальный набор файлов для обучения и отчёта:

```text
data/feature_matrix_spec.csv
data/spec_feature_importances_ohe.csv
data/correlations.csv
```

Если скачиваете вручную через Kaggle UI, используйте страницу:
https://www.kaggle.com/datasets/willkoehrsen/home-credit-default-risk-feature-tools?select=correlations.csv

## Структура репозитория

```text
.
├── data                         # CSV из Kaggle
├── Dockerfile                   # Общий образ для FastAPI и Streamlit
├── docker-compose.yml           # Локальный deploy API + UI
├── models                       # Результаты экспериментов и best_model.joblib
├── presentation                 # Материалы для защиты
├── report
│   ├── data_quality             # Таблицы качества данных
│   ├── images                   # EDA-графики
│   ├── data_quality_report.md   # Data quality report
│   └── report.md                # Итоговый отчёт CP3
├── src
│   ├── api                  # FastAPI deploy layer
│   ├── data_quality.py          # Анализ пропусков, типов, дублей, выбросов
│   ├── eda.py                   # Генерация графиков
│   ├── modeling.py              # Baseline, модели, ансамбль, метрики
│   ├── preprocessing.py         # Очистка, split, feature engineering
│   └── ui                       # Streamlit interface
├── tests
│   ├── test_api.py              # Тесты FastAPI endpoints
│   ├── test_model_service.py    # Тесты deploy model service
│   └── test_pipeline.py         # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск локально

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Быстрый smoke-run на небольшом сэмпле:

```bash
python -m src.data_quality
python -m src.eda --sample-size 5000 --top-n-features 40
python -m src.modeling --sample-size 5000 --top-n-features 40 --quick
```

Основной запуск обучения:

```bash
.venv/bin/python -m src.data_quality --top-n-features 120
.venv/bin/python -m src.eda --sample-size 50000 --top-n-features 120
.venv/bin/python -m src.modeling --sample-size 50000 --top-n-features 120
```

После обучения должен появиться файл `models/best_model.joblib`. Он специально разрешён в `.gitignore`, потому что
для CP3 проверяющему нужен рабочий deploy без переобучения модели.

## FastAPI deploy

API реализован в `src/api/`. Он запускается даже без `models/best_model.joblib`: в этом случае `/health`
покажет, что модель не найдена, а `/predict` вернёт HTTP `503` с командой для обучения модели.

Сначала нужно установить зависимости и, если есть исходные CSV из Kaggle, обучить модель:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src.modeling --sample-size 50000 --top-n-features 120
```

Запуск API:

```bash
.venv/bin/python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Полезные URL:

- health check: `http://127.0.0.1:8000/health`;
- Swagger UI: `http://127.0.0.1:8000/docs`;
- модель и ожидаемые признаки: `http://127.0.0.1:8000/model-info`.

Пример запроса:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"AMT_CREDIT":406597.5,"AMT_INCOME_TOTAL":202500.0,"EXT_SOURCE_2":0.2629}}'
```

## Streamlit interface

Интерфейс реализован в `src/ui/streamlit_app.py` и работает поверх FastAPI. Это не одиночная форма, а небольшой
скоринговый кабинет для аналитика:

- `Overview` - KPI проекта и позиционирование как decision-support;
- `Single Client Scoring` - скоринг одной заявки и факторы риска;
- `Batch Scoring` - загрузка CSV и выгрузка скоринга;
- `Data Explorer` - EDA-графики и таблица пропусков;
- `Model Explainability` - feature importance и понятные risk drivers;
- `Model Performance` - метрики моделей и threshold review.

Запуск в двух терминалах:

```bash
.venv/bin/python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

```bash
API_BASE_URL=http://127.0.0.1:8000 .venv/bin/python -m streamlit run src/ui/streamlit_app.py
```

После запуска Streamlit откроется на `http://localhost:8501`.

## Docker Compose deploy

Удалённый сервер для CP3 не обязателен: по критериям достаточно локального deploy, FastAPI endpoint и видео работы.
Для проверки всего проекта одной командой:

```bash
docker compose up --build
```

Сервисы:

- FastAPI: `http://127.0.0.1:8000`;
- Swagger UI: `http://127.0.0.1:8000/docs`;
- Streamlit dashboard: `http://127.0.0.1:8501`.

Compose использует один образ `home-credit-risk-scoring:cp3` и два процесса:

- `api` запускает `uvicorn src.api.app:app`;
- `streamlit` запускает `streamlit run src/ui/streamlit_app.py` и обращается к API по `http://api:8000`.

Проверки:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests --line-length 120
```

## Результаты

После запуска `src.modeling` таблица экспериментов сохраняется в `models/experiment_results.csv`, а лучшая модель - в `models/best_model.joblib`.

Финальный запуск на `50000` строках и `120` top признаках:

- best model: `hist_gradient_boosting`;
- test ROC-AUC: `0.7783`;
- test Average Precision: `0.2685`;
- test F1: `0.0411`.

Data quality report сохраняется в [`report/data_quality_report.md`](report/data_quality_report.md), а подробные таблицы - в `report/data_quality/`.

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md).

Перед сдачей в отчёте должна быть фактическая ссылка на видео работы FastAPI + Streamlit dashboard.
