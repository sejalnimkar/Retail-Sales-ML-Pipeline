# Results

Sales are in cases. Part 1 is the first stretch of data; part 2 (if present) is after the break.

Train 2017-06 to 2019-05 (179,188 item-months), validation 2019-06 to 2019-10 (60,356), test 2019-11 to 2020-09 (57,314).

## Part 1: model selection (validation months)

| Method | R² (log scale) | MAE (cases) | WAPE |
|---|---|---|---|
| Random Forest | 0.9397 | 2.16 | 29.6% |
| XGBoost | 0.9388 | 2.53 | 34.7% |
| Gradient Boosting (sklearn) | 0.9333 | 2.68 | 36.6% |
| Baseline: average of last 3 months | 0.9202 | 2.68 | 36.7% |
| Ridge Regression | 0.9124 | 2.98 | 40.8% |
| Baseline: last observed month | 0.8958 | 3.37 | 46.2% |

## Part 1: final score (test months; selected model: Random Forest)

| Method | R² (log scale) | MAE (cases) | WAPE |
|---|---|---|---|
| Random Forest | 0.8959 | 3.09 | 38.9% |
| Baseline: average of last 3 months | 0.8807 | 3.00 | 37.8% |
| Baseline: last observed month | 0.8721 | 3.48 | 43.9% |

On test rows with non-zero sales only: R² (log) 0.8428, MAE 5.21, WAPE 38.5%.

## Part 1: selected model vs. last-month baseline, by test month

| Month | Item-months | Baseline R² (log) | Model R² (log) | Baseline MAE | Model MAE |
|---|---|---|---|---|---|
| 2019-11 | 12,407 | 0.8673 | 0.9192 | 4.07 | 2.62 |
| 2020-01 | 11,550 | 0.8504 | 0.8898 | 3.35 | 2.56 |
| 2020-03 | 11,172 | 0.9006 | 0.9086 | 3.73 | 3.98 |
| 2020-07 | 10,933 | 0.8507 | 0.8607 | 3.70 | 3.56 |
| 2020-09 | 11,252 | 0.8866 | 0.8966 | 2.51 | 2.81 |

## Part 1: selected model by item type (test months)

| Item type | Item-months | R² (log scale) | MAE (cases) | WAPE |
|---|---|---|---|---|
| BEER | 8,629 | 0.8499 | 5.31 | 36.4% |
| LIQUOR | 12,333 | 0.8681 | 5.21 | 36.6% |
| NON-ALCOHOL | 436 | 0.8564 | 5.36 | 35.0% |
| WINE | 35,916 | 0.9112 | 1.80 | 43.9% |

## Part 2: after the break in the data, month by month

Each month is forecast by Random Forest trained on every earlier month.

| Method | R² (log scale) | MAE (cases) | WAPE |
|---|---|---|---|
| Random Forest | 0.9435 | 1.48 | 24.4% |
| Baseline: last observed month | 0.9329 | 1.56 | 25.8% |
| Baseline: average of last 3 months | 0.9328 | 1.56 | 25.8% |

| Month | Item-months | Model MAE | Last-month baseline MAE | 3-month-average baseline MAE |
|---|---|---|---|---|
| 2026-07 | 10,711 | 1.72 | 1.63 | 1.63 |
| 2026-08 | 10,627 | 1.40 | 1.69 | 1.56 |
| 2026-09 | 11,064 | 1.33 | 1.37 | 1.50 |
