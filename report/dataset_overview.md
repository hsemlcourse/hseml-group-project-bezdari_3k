# Обзор CSV-датасетов

Превью ниже показывает первые 10 строк каждого CSV. Для широких feature-matrix файлов выведены первые 10 колонок плюс TARGET, потому что полный вывод 800-3500 колонок нечитаем.

## correlations.csv

Что это: Корреляционная матрица для полной Featuretools-матрицы.

За что отвечает: Показывает Pearson-корреляции между признаками. Используется для EDA: поиск сильно связанных признаков, признаков, связанных с TARGET, и потенциальной мультиколлинеарности. Это не обучающий датасет: строки/столбцы здесь признаки, а значения - корреляции.

Всего колонок: 1985

Показанные колонки: SK_ID_CURR, REG_REGION_NOT_LIVE_REGION, EXT_SOURCE_2, FLAG_PHONE, AMT_INCOME_TOTAL, AMT_CREDIT, REGION_RATING_CLIENT, REG_REGION_NOT_WORK_REGION, CNT_FAM_MEMBERS, DAYS_BIRTH, TARGET

```text
 SK_ID_CURR  REG_REGION_NOT_LIVE_REGION  EXT_SOURCE_2  FLAG_PHONE  AMT_INCOME_TOTAL  AMT_CREDIT  REGION_RATING_CLIENT  REG_REGION_NOT_WORK_REGION  CNT_FAM_MEMBERS  DAYS_BIRTH    TARGET
   1.000000                   -0.000283      0.002342    0.002753         -0.001820   -0.000343             -0.001075                    0.001097        -0.002895   -0.001500 -0.002108
  -0.000283                    1.000000      0.015570    0.002078          0.031191    0.024010             -0.044166                    0.450804        -0.017133    0.065486  0.005576
   0.002342                    0.015570      1.000000    0.061178          0.060925    0.131228             -0.292895                    0.029517        -0.001823   -0.091996 -0.160472
   0.002753                    0.002078      0.061178    1.000000          0.000159    0.026213             -0.083827                    0.004284        -0.015418   -0.042402 -0.023806
  -0.001820                    0.031191      0.060925    0.000159          1.000000    0.156870             -0.085465                    0.062340         0.016342    0.027261 -0.003982
  -0.000343                    0.024010      0.131228    0.026213          0.156870    1.000000             -0.101776                    0.051929         0.063160   -0.055436 -0.030369
  -0.001075                   -0.044166     -0.292895   -0.083827         -0.085465   -0.101776              1.000000                   -0.139890         0.029688    0.009361  0.058899
   0.001097                    0.450804      0.029517    0.004284          0.062340    0.051929             -0.139890                    1.000000         0.003135    0.095819  0.006942
  -0.002895                   -0.017133     -0.001823   -0.015418          0.016342    0.063160              0.029688                    0.003135         1.000000    0.278894  0.009308
  -0.001500                    0.065486     -0.091996   -0.042402          0.027261   -0.055436              0.009361                    0.095819         0.278894    1.000000  0.078239
```

## correlations_spec.csv

Что это: Корреляционная матрица для compact/spec набора признаков.

За что отвечает: То же самое, что correlations.csv, но для более компактной версии признаковой матрицы. Колонка Unnamed: 0 хранит имя признака-строки.

Всего колонок: 1173

Показанные колонки: Unnamed: 0, SK_ID_CURR, FLAG_EMP_PHONE, DEF_60_CNT_SOCIAL_CIRCLE, LIVE_REGION_NOT_WORK_REGION, YEARS_BEGINEXPLUATATION_AVG, APARTMENTS_MODE, FLAG_MOBIL, FLOORSMIN_MEDI, BASEMENTAREA_AVG, TARGET

```text
                 Unnamed: 0  SK_ID_CURR  FLAG_EMP_PHONE  DEF_60_CNT_SOCIAL_CIRCLE  LIVE_REGION_NOT_WORK_REGION  YEARS_BEGINEXPLUATATION_AVG  APARTMENTS_MODE  FLAG_MOBIL  FLOORSMIN_MEDI  BASEMENTAREA_AVG    TARGET
                 SK_ID_CURR    1.000000       -0.001337                  0.001187                     0.002903                     0.001551         0.001961    0.002804        0.002837         -0.002070 -0.002108
             FLAG_EMP_PHONE   -0.001337        1.000000                 -0.014034                     0.096447                    -0.008672         0.014720   -0.000845        0.016486          0.001292  0.045982
   DEF_60_CNT_SOCIAL_CIRCLE    0.001187       -0.014034                  1.000000                    -0.016693                    -0.004751        -0.015337    0.000499       -0.020663         -0.012925  0.031276
LIVE_REGION_NOT_WORK_REGION    0.002903        0.096447                 -0.016693                     1.000000                     0.012031         0.017831    0.000371        0.053927          0.000965  0.002819
YEARS_BEGINEXPLUATATION_AVG    0.001551       -0.008672                 -0.004751                     0.012031                     1.000000         0.100665   -0.000649        0.166756          0.085950 -0.009728
            APARTMENTS_MODE    0.001961        0.014720                 -0.015337                     0.017831                     0.100665         1.000000   -0.000283        0.418234          0.666023 -0.027284
                 FLAG_MOBIL    0.002804       -0.000845                  0.000499                     0.000371                    -0.000649        -0.000283    1.000000             NaN          0.000963  0.000534
             FLOORSMIN_MEDI    0.002837        0.016486                 -0.020663                     0.053927                     0.166756         0.418234         NaN        1.000000          0.219523 -0.033394
           BASEMENTAREA_AVG   -0.002070        0.001292                 -0.012925                     0.000965                     0.085950         0.666023    0.000963        0.219523          1.000000 -0.022746
              LANDAREA_MODE    0.001548        0.008657                 -0.002882                    -0.006717                     0.070008         0.508160   -0.000618        0.139227          0.461969 -0.010174
```

## feature_importances.csv

Что это: Важности признаков для feature_matrix.csv.

За что отвечает: Список признаков и их importance, рассчитанный автором Kaggle-датасета. Нужен для отбора top-N признаков и интерпретации.

Всего колонок: 3

Показанные колонки: index, feature, importance

```text
 index                 feature  importance
    24       ORGANIZATION_TYPE        5504
    74            EXT_SOURCE_1        2304
     2            EXT_SOURCE_2        2298
    42            EXT_SOURCE_3        2298
   105             AMT_ANNUITY        1121
    10              DAYS_BIRTH        1108
     6              AMT_CREDIT        1015
   524 MAX(bureau.DAYS_CREDIT)         906
    27         DAYS_ID_PUBLISH         727
    95           DAYS_EMPLOYED         723
```

## feature_matrix.csv

Что это: Основная признаковая матрица Featuretools.

За что отвечает: Каждая строка - клиент/заявка SK_ID_CURR. Колонки - исходные и автоматически сгенерированные признаки. TARGET: 0 - нет дефолта, 1 - дефолт, -999 - Kaggle test без разметки.

Всего колонок: 1698

Показанные колонки: SK_ID_CURR, NAME_TYPE_SUITE, REG_REGION_NOT_LIVE_REGION, EXT_SOURCE_2, OCCUPATION_TYPE, FLAG_PHONE, AMT_INCOME_TOTAL, AMT_CREDIT, REGION_RATING_CLIENT, REG_REGION_NOT_WORK_REGION, TARGET

```text
 SK_ID_CURR NAME_TYPE_SUITE  REG_REGION_NOT_LIVE_REGION  EXT_SOURCE_2    OCCUPATION_TYPE  FLAG_PHONE  AMT_INCOME_TOTAL  AMT_CREDIT  REGION_RATING_CLIENT  REG_REGION_NOT_WORK_REGION  TARGET
     100001   Unaccompanied                           0      0.789654                NaN           0          135000.0    568800.0                     2                           0    -999
     100002   Unaccompanied                           0      0.262949           Laborers           1          202500.0    406597.5                     2                           0       1
     100003          Family                           0      0.622246         Core staff           1          270000.0   1293502.5                     1                           0       0
     100004   Unaccompanied                           0      0.555912           Laborers           1           67500.0    135000.0                     2                           0       0
     100005   Unaccompanied                           0      0.291656 Low-skill Laborers           0           99000.0    222768.0                     2                           0    -999
     100006   Unaccompanied                           0      0.650442           Laborers           0          135000.0    312682.5                     2                           0       0
     100007   Unaccompanied                           0      0.322738         Core staff           0          121500.0    513000.0                     2                           0       0
     100008 Spouse, partner                           0      0.354225           Laborers           1           99000.0    490495.5                     2                           0       0
     100009   Unaccompanied                           0      0.724000        Accountants           1          171000.0   1560726.0                     2                           0       0
     100010   Unaccompanied                           0      0.714279           Managers           0          360000.0   1530000.0                     3                           0       0
```

## feature_matrix_advanced.csv

Что это: Расширенная advanced-признаковая матрица.

За что отвечает: Похожа на feature_matrix.csv, но содержит больше автоматически созданных и ручных признаков, например NEW_ANNUITY_TO_INCOME_RATIO. TARGET может быть NaN для Kaggle test rows.

Всего колонок: 3596

Показанные колонки: SK_ID_CURR, COMMONAREA_MODE, NEW_ANNUITY_TO_INCOME_RATIO, FLAG_DOCUMENT_9, LIVINGAPARTMENTS_MODE, EXT_SOURCE_3, NAME_INCOME_TYPE, FLAG_EMP_PHONE, index, FLAG_DOCUMENT_6, TARGET

```text
 SK_ID_CURR  COMMONAREA_MODE  NEW_ANNUITY_TO_INCOME_RATIO  FLAG_DOCUMENT_9  LIVINGAPARTMENTS_MODE  EXT_SOURCE_3     NAME_INCOME_TYPE  FLAG_EMP_PHONE  index  FLAG_DOCUMENT_6  TARGET
     100001              NaN                     0.152299                0                    NaN      0.159520              Working               1      0                0     NaN
     100002           0.0144                     0.121977                0                  0.022      0.139376              Working               1      0                0     1.0
     100003           0.0497                     0.132216                0                  0.079           NaN        State servant               1      1                0     0.0
     100004              NaN                     0.099999                0                    NaN      0.729567              Working               1      2                0     0.0
     100005              NaN                     0.175453                0                    NaN      0.432962              Working               1      1                0     NaN
     100006              NaN                     0.219898                0                    NaN           NaN              Working               1      3                0     0.0
     100007              NaN                     0.179961                0                    NaN           NaN              Working               1      4                0     0.0
     100008              NaN                     0.277952                0                    NaN      0.621226        State servant               1      5                0     0.0
     100009              NaN                     0.241525                0                    NaN      0.492060 Commercial associate               1      6                0     0.0
     100010              NaN                     0.116875                0                    NaN      0.540654        State servant               1      7                0     0.0
```

## feature_matrix_article.csv

Что это: Feature matrix из статьи/примера автора.

За что отвечает: Признаковая матрица, подготовленная для демонстрационного article/notebook варианта. Строки - клиенты, колонки - признаки, TARGET - целевая переменная.

Всего колонок: 1821

Показанные колонки: SK_ID_CURR, AMT_ANNUITY, AMT_CREDIT, AMT_GOODS_PRICE, AMT_INCOME_TOTAL, AMT_REQ_CREDIT_BUREAU_DAY, AMT_REQ_CREDIT_BUREAU_HOUR, AMT_REQ_CREDIT_BUREAU_MON, AMT_REQ_CREDIT_BUREAU_QRT, AMT_REQ_CREDIT_BUREAU_WEEK, TARGET

```text
 SK_ID_CURR  AMT_ANNUITY  AMT_CREDIT  AMT_GOODS_PRICE  AMT_INCOME_TOTAL  AMT_REQ_CREDIT_BUREAU_DAY  AMT_REQ_CREDIT_BUREAU_HOUR  AMT_REQ_CREDIT_BUREAU_MON  AMT_REQ_CREDIT_BUREAU_QRT  AMT_REQ_CREDIT_BUREAU_WEEK  TARGET
     124904      15808.5    325908.0         247500.0           72000.0                        0.0                         0.0                        0.0                        0.0                         0.0     0.0
     124907      11250.0    225000.0         225000.0          112500.0                        1.0                         0.0                        0.0                        0.0                         0.0     0.0
     124908       9000.0    180000.0         180000.0          112500.0                        0.0                         0.0                        0.0                        0.0                         0.0     0.0
     124909      25560.0    528633.0         472500.0           54000.0                        0.0                         0.0                        0.0                        0.0                         0.0     0.0
     124910      36054.0    497520.0         450000.0          247500.0                        0.0                         0.0                        0.0                        0.0                         0.0     0.0
     124911      17527.5    284400.0         225000.0          135000.0                        0.0                         0.0                        0.0                        0.0                         0.0     0.0
     124912      24408.0    333621.0         288000.0          270000.0                        0.0                         0.0                        0.0                        0.0                         0.0     0.0
     124913       9828.0     95940.0          90000.0           67500.0                        0.0                         0.0                        0.0                        0.0                         0.0     0.0
     124914       9058.5    284400.0         225000.0           72000.0                        NaN                         NaN                        NaN                        NaN                         NaN     0.0
     124915      26640.0    545040.0         450000.0          180000.0                        0.0                         0.0                        0.0                        0.0                         0.0     0.0
```

## feature_matrix_spec.csv

Что это: Компактная spec-признаковая матрица.

За что отвечает: Главный датасет для нашего CP1-пайплайна: он меньше остальных feature matrix, но содержит достаточно признаков для обучения. TARGET = -999 исключаем из supervised learning.

Всего колонок: 885

Показанные колонки: SK_ID_CURR, FLAG_EMP_PHONE, DEF_60_CNT_SOCIAL_CIRCLE, LIVE_REGION_NOT_WORK_REGION, ORGANIZATION_TYPE, YEARS_BEGINEXPLUATATION_AVG, APARTMENTS_MODE, FLAG_MOBIL, FLOORSMIN_MEDI, BASEMENTAREA_AVG, TARGET

```text
 SK_ID_CURR  FLAG_EMP_PHONE  DEF_60_CNT_SOCIAL_CIRCLE  LIVE_REGION_NOT_WORK_REGION      ORGANIZATION_TYPE  YEARS_BEGINEXPLUATATION_AVG  APARTMENTS_MODE  FLAG_MOBIL  FLOORSMIN_MEDI  BASEMENTAREA_AVG  TARGET
     100001               1                       0.0                            0           Kindergarten                       0.9732           0.0672           1             NaN            0.0590    -999
     100002               1                       2.0                            0 Business Entity Type 3                       0.9722           0.0252           1          0.1250            0.0369       1
     100003               1                       0.0                            0                 School                       0.9851           0.0924           1          0.3333            0.0529       0
     100004               1                       0.0                            0             Government                          NaN              NaN           1             NaN               NaN       0
     100005               1                       0.0                            0          Self-employed                          NaN              NaN           1             NaN               NaN    -999
     100006               1                       0.0                            0 Business Entity Type 3                          NaN              NaN           1             NaN               NaN       0
     100007               1                       0.0                            0               Religion                          NaN              NaN           1             NaN               NaN       0
     100008               1                       0.0                            0                  Other                          NaN              NaN           1             NaN               NaN       0
     100009               1                       0.0                            0 Business Entity Type 3                          NaN              NaN           1             NaN               NaN       0
     100010               1                       0.0                            0                  Other                          NaN              NaN           1             NaN               NaN       0
```

## fi_fma.csv

Что это: Feature importances для feature_matrix_advanced.csv.

За что отвечает: Важности признаков advanced-матрицы. Помогает выбрать признаки из самого широкого advanced-набора.

Всего колонок: 3

Показанные колонки: Unnamed: 0, feature, importance

```text
 Unnamed: 0                     feature  importance
          0             COMMONAREA_MODE        37.6
          1 NEW_ANNUITY_TO_INCOME_RATIO       192.4
          2             FLAG_DOCUMENT_9         0.0
          3       LIVINGAPARTMENTS_MODE        27.6
          4                EXT_SOURCE_3       429.6
          5              FLAG_EMP_PHONE         0.4
          6                       index        23.2
          7             FLAG_DOCUMENT_6         0.6
          8            BASEMENTAREA_AVG        32.2
          9           BASEMENTAREA_MEDI        27.4
```

## spec_feature_importances_ohe.csv

Что это: Feature importances для feature_matrix_spec.csv после OHE.

За что отвечает: Используется в нашем коде для чтения top-N признаков из feature_matrix_spec.csv, чтобы не загружать все 885 колонок.

Всего колонок: 3

Показанные колонки: Unnamed: 0, feature, importance

```text
 Unnamed: 0                     feature  importance
          0              FLAG_EMP_PHONE         1.4
          1    DEF_60_CNT_SOCIAL_CIRCLE        22.8
          2 LIVE_REGION_NOT_WORK_REGION         0.6
          3 YEARS_BEGINEXPLUATATION_AVG        19.8
          4             APARTMENTS_MODE        30.0
          5                  FLAG_MOBIL         0.0
          6              FLOORSMIN_MEDI         5.2
          7            BASEMENTAREA_AVG        21.8
          8               LANDAREA_MODE        27.2
          9            FLAG_DOCUMENT_21         0.0
```
