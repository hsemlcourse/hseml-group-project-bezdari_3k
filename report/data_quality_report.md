# Data Quality Report

Основной файл анализа: `data\feature_matrix_spec.csv`.

## Краткий Вывод

- Всего строк: `356255`.
- Всего колонок: `885`.
- Размеченных строк для обучения (`TARGET` 0/1): `307511`.
- Kaggle test rows без разметки (`TARGET = -999`): `48744`.
- Дубликатов по `SK_ID_CURR`: `0`.
- Колонок с пропусками: `823`.
- Колонок с >= 70% пропусков: `302`.
- Колонок с >= 90% пропусков: `6`.
- В модельном пайплайне выбрано top-N признаков: `120`.

## TARGET

| TARGET | rows | share |
| --- | --- | --- |
| 0 | 282686 | 0.7935 |
| -999 | 48744 | 0.1368 |
| 1 | 24825 | 0.0697 |

## Типы Данных

| dtype | columns |
| --- | --- |
| float64 | 802 |
| object | 42 |
| int64 | 41 |

## Дубликаты И Идентификаторы

| rows | unique_sk_id_curr | duplicate_sk_id_curr_rows | missing_sk_id_curr |
| --- | --- | --- | --- |
| 356255 | 356255 | 0 | 0 |

## Пропуски

Таблица ниже показывает признаки с максимальной долей пропусков. Полная таблица сохранена в `report\data_quality\missing_values.csv`.

| feature | missing_count | missing_share |
| --- | --- | --- |
| MAX(previous_app.RATE_INTEREST_PRIMARY) | 350534 | 0.9839 |
| MAX(previous_app.RATE_INTEREST_PRIVILEGED) | 350534 | 0.9839 |
| MEAN(previous_app.RATE_INTEREST_PRIVILEGED) | 350534 | 0.9839 |
| MIN(previous_app.RATE_INTEREST_PRIMARY) | 350534 | 0.9839 |
| MIN(previous_app.RATE_INTEREST_PRIVILEGED) | 350534 | 0.9839 |
| MEAN(previous_app.RATE_INTEREST_PRIMARY) | 350534 | 0.9839 |
| MEAN(credit.AMT_PAYMENT_CURRENT) | 294314 | 0.8261 |
| MIN(credit.AMT_PAYMENT_CURRENT) | 294314 | 0.8261 |
| MAX(credit.AMT_PAYMENT_CURRENT) | 294314 | 0.8261 |
| MAX(previous_app.MIN(credit.AMT_PAYMENT_CURRENT)) | 294314 | 0.8261 |
| MIN(previous_app.MEAN(credit.AMT_PAYMENT_CURRENT)) | 294314 | 0.8261 |
| MIN(previous_app.MAX(credit.AMT_PAYMENT_CURRENT)) | 294314 | 0.8261 |
| MEAN(previous_app.MAX(credit.AMT_PAYMENT_CURRENT)) | 294314 | 0.8261 |
| MEAN(previous_app.MEAN(credit.AMT_PAYMENT_CURRENT)) | 294314 | 0.8261 |
| MEAN(previous_app.MIN(credit.AMT_PAYMENT_CURRENT)) | 294314 | 0.8261 |
| MAX(previous_app.MEAN(credit.AMT_PAYMENT_CURRENT)) | 294314 | 0.8261 |
| MEAN(credit.AMT_DRAWINGS_OTHER_CURRENT) | 294231 | 0.8259 |
| MAX(credit.AMT_DRAWINGS_ATM_CURRENT) | 294231 | 0.8259 |
| MEAN(credit.AMT_DRAWINGS_ATM_CURRENT) | 294231 | 0.8259 |
| MAX(credit.AMT_DRAWINGS_OTHER_CURRENT) | 294231 | 0.8259 |

## Выбросы

Выбросы оценены IQR-правилом на выбранных числовых признаках после удаления Kaggle test rows. В самом пайплайне значения клипуются по 1% и 99% квантилям, причём границы считаются только на train split.

| feature | non_missing_count | missing_count | min | q01 | q25 | median | q75 | q99 | max | iqr_outlier_count | iqr_outlier_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MEAN(installments.NUM_INSTALMENT_VERSION) | 289406 | 18105 | 0.0000 | 0.0463 | 1.0000 | 1.0244 | 1.1111 | 3.2726 | 39.0000 | 82064 | 0.2836 |
| REGION_RATING_CLIENT_W_CITY | 307511 | 0 | 1.0000 | 1.0000 | 2.0000 | 2.0000 | 2.0000 | 3.0000 | 3.0000 | 78027 | 0.2537 |
| SUM(bureau.AMT_CREDIT_MAX_OVERDUE) | 263491 | 44020 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 382.5000 | 79023.9015 | 115987185.0000 | 64330 | 0.2441 |
| DAYS_EMPLOYED | 307511 | 0 | -17912.0000 | -10894.9000 | -2760.0000 | -1213.0000 | -289.0000 | 365243.0000 | 365243.0000 | 72217 | 0.2348 |
| MEAN(bureau.AMT_CREDIT_SUM_LIMIT) | 242442 | 65069 | -97891.6613 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 112811.0687 | 4500000.0000 | 48852 | 0.2015 |
| MAX(bureau.DAYS_CREDIT_ENDDATE) | 261242 | 46269 | -41875.0000 | -2162.0000 | 125.0000 | 909.0000 | 1683.0000 | 31171.0000 | 31199.0000 | 51845 | 0.1985 |
| MAX(bureau.DAYS_CREDIT_UPDATE) | 263491 | 44020 | -41890.0000 | -1883.1000 | -50.0000 | -19.0000 | -9.0000 | -1.0000 | 372.0000 | 50659 | 0.1923 |
| SUM(bureau.DAYS_CREDIT_ENDDATE) | 263491 | 44020 | -155271.0000 | -14741.3000 | -2876.0000 | -382.0000 | 1754.0000 | 56711.8000 | 214193.0000 | 48077 | 0.1825 |
| MEAN(previous_app.MEAN(installments.NUM_INSTALMENT_VERSION)) | 289406 | 18105 | 0.0000 | 0.5000 | 1.0000 | 1.0370 | 1.1667 | 2.8750 | 39.0000 | 52648 | 0.1819 |
| MAX(previous_app.MEAN(installments.NUM_INSTALMENT_VERSION)) | 289406 | 18105 | 0.0000 | 1.0000 | 1.0000 | 1.1250 | 1.3333 | 5.2381 | 89.5000 | 50236 | 0.1736 |
| MIN(previous_app.SELLERPLACE_AREA) | 291057 | 16454 | -1.0000 | -1.0000 | -1.0000 | -1.0000 | 30.0000 | 2586.0000 | 4000000.0000 | 42220 | 0.1451 |
| MEAN(bureau.AMT_CREDIT_MAX_OVERDUE) | 183886 | 123625 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2124.0000 | 41269.5000 | 115987185.0000 | 25423 | 0.1383 |
| MIN(bureau.AMT_CREDIT_SUM_DEBT) | 256131 | 51380 | -4705600.3200 | -384.9975 | 0.0000 | 0.0000 | 0.0000 | 744498.0000 | 43650000.0000 | 33746 | 0.1318 |
| MEAN(previous_app.SELLERPLACE_AREA) | 291057 | 16454 | -1.0000 | -1.0000 | 23.0000 | 79.0000 | 350.0000 | 3333.0000 | 4000000.0000 | 34832 | 0.1197 |
| SUM(previous_app.MIN(cash.CNT_INSTALMENT_FUTURE)) | 291057 | 16454 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0000 | 55.0000 | 132.0000 | 34430 | 0.1183 |
| MEAN(bureau.DAYS_CREDIT_ENDDATE) | 261242 | 46269 | -41875.0000 | -2233.2992 | -703.5000 | -135.5778 | 602.5000 | 15265.5000 | 31198.0000 | 29570 | 0.1132 |
| MEAN(previous_app.MIN(cash.CNT_INSTALMENT_FUTURE)) | 286935 | 20576 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.7500 | 21.0000 | 60.0000 | 31853 | 0.1110 |
| MAX(bureau.AMT_CREDIT_MAX_OVERDUE) | 183886 | 123625 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5928.7612 | 75627.0000 | 115987185.0000 | 20126 | 0.1094 |
| MAX(cash.MONTHS_BALANCE) | 286967 | 20544 | -93.0000 | -78.0000 | -14.0000 | -4.0000 | -2.0000 | -1.0000 | -1.0000 | 31185 | 0.1087 |
| MAX(installments.DAYS_INSTALMENT) | 289406 | 18105 | -2883.0000 | -2344.0000 | -376.0000 | -54.0000 | -19.0000 | -3.0000 | -1.0000 | 30651 | 0.1059 |

## Отбор Признаков

Для воспроизводимости и экономии памяти модель читает top-N признаков из `data\spec_feature_importances_ohe.csv`. Полная таблица выбранных признаков сохранена в `report\data_quality\selected_features.csv`.

| rank | feature | importance |
| --- | --- | --- |
| 1 | EXT_SOURCE_1 | 357.6000 |
| 2 | EXT_SOURCE_3 | 311.6000 |
| 3 | EXT_SOURCE_2 | 307.0000 |
| 4 | DAYS_BIRTH | 228.6000 |
| 5 | AMT_CREDIT | 204.4000 |
| 6 | AMT_ANNUITY | 195.2000 |
| 7 | DAYS_EMPLOYED | 150.2000 |
| 8 | AMT_GOODS_PRICE | 144.2000 |
| 9 | MAX(bureau.DAYS_CREDIT) | 138.8000 |
| 10 | DAYS_ID_PUBLISH | 122.8000 |
| 11 | MAX(bureau.DAYS_CREDIT_ENDDATE) | 121.8000 |
| 12 | OWN_CAR_AGE | 104.0000 |
| 13 | SUM(previous_app.MIN(installments.AMT_PAYMENT)) | 101.8000 |
| 14 | MAX(bureau.DAYS_ENDDATE_FACT) | 99.6000 |
| 15 | MEAN(previous_app.MIN(installments.AMT_PAYMENT)) | 99.0000 |
| 16 | MIN(installments.AMT_PAYMENT) | 98.0000 |
| 17 | MEAN(bureau.AMT_CREDIT_SUM_DEBT) | 93.0000 |
| 18 | DAYS_REGISTRATION | 92.6000 |
| 19 | DAYS_LAST_PHONE_CHANGE | 86.4000 |
| 20 | MEAN(previous_app.COUNT(cash)) | 82.8000 |

## Что Делает Очистка В Пайплайне

- Удаляет строки `TARGET = -999` и `TARGET = NaN`, потому что это test rows без ответа.
- Приводит `TARGET` к бинарному `int`.
- Удаляет дубликаты по `SK_ID_CURR`.
- Исключает служебные колонки вроде `Unnamed: 0`.
- Заполняет числовые пропуски медианой, категориальные - значением `missing`.
- Кодирует категориальные признаки через OneHotEncoder или OrdinalEncoder в зависимости от модели.
- Клипует числовые выбросы по квантилям внутри sklearn Pipeline, чтобы не было data leakage.
