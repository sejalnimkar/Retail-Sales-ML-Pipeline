# Results

Train 2017-06 to 2019-05 (179,188 item-months), validation 2019-06 to 2019-10 (60,356), test 2019-11 to 2020-09 (57,314).

## Model selection (validation months)

| Method | R² (log scale) | MAE (units) | WAPE |
|---|---|---|---|
| XGBoost | 0.9326 | 2.58 | 35.3% |
| Random Forest | 0.9325 | 2.35 | 32.2% |
| Gradient Boosting (sklearn) | 0.9283 | 2.68 | 36.7% |
| Ridge Regression | 0.9076 | 3.09 | 42.2% |
| Baseline: last observed month | 0.8958 | 3.37 | 46.2% |
| Baseline: item's historical average | 0.8476 | 3.33 | 45.6% |

## Final score (test months; selected model: XGBoost)

| Method | R² (log scale) | MAE (units) | WAPE |
|---|---|---|---|
| XGBoost | 0.8964 | 3.39 | 42.7% |
| Baseline: last observed month | 0.8721 | 3.48 | 43.9% |
| Baseline: item's historical average | 0.8413 | 3.50 | 44.1% |

On test rows with non-zero sales only: R² (log) 0.8416, MAE 5.73, WAPE 42.3%.

## Selected model vs. last-month baseline, by test month

| Month | Item-months | Baseline R² (log) | Model R² (log) | Baseline MAE | Model MAE |
|---|---|---|---|---|---|
| 2019-11 | 12,407 | 0.8673 | 0.9262 | 4.07 | 2.81 |
| 2020-01 | 11,550 | 0.8504 | 0.8935 | 3.35 | 2.83 |
| 2020-03 | 11,172 | 0.9006 | 0.9097 | 3.73 | 4.09 |
| 2020-07 | 10,933 | 0.8507 | 0.8661 | 3.70 | 3.88 |
| 2020-09 | 11,252 | 0.8866 | 0.8810 | 2.51 | 3.43 |

## Selected model by item type (test months)

| Item type | Item-months | R² (log scale) | MAE (units) | WAPE |
|---|---|---|---|---|
| BEER | 8,629 | 0.8497 | 6.36 | 43.6% |
| LIQUOR | 12,333 | 0.8772 | 5.50 | 38.7% |
| NON-ALCOHOL | 436 | 0.8874 | 4.94 | 32.3% |
| WINE | 35,916 | 0.9069 | 1.93 | 47.1% |
