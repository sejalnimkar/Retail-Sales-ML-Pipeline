# Montgomery County Warehouse & Retail Sales Forecaster

[![CI](https://github.com/sejalnimkar/retail-sales-ml-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/sejalnimkar/retail-sales-ml-pipeline/actions/workflows/ci.yml)

A machine learning pipeline that forecasts next-month retail sales for individual liquor, wine, beer and non-alcohol products in Montgomery County, MD, using only what is known before the month starts. Built on public county data (307K+ records, 2017–2020), evaluated on months the model never saw, with a Streamlit app for interactive forecasts.

---

## Results

<!-- RESULTS:START -->
Scored on the five latest months in the data (2019-11 to 2020-09, 57,314 item-months), which the model never saw during training or model selection.

| Method | R² (log scale) | MAE (units) | WAPE |
|---|---|---|---|
| **XGBoost** | 0.896 | 3.39 | 42.7% |
| Baseline: same as the item's last recorded month | 0.872 | 3.48 | 43.9% |
| Baseline: the item's historical average | 0.841 | 3.50 | 44.1% |

**How to read this.** The model beats both baselines, but the margin on the test months is small: MAE 3.39 vs. 3.48 units. On the validation months (2019-06 to 2019-10) the margin was much wider: MAE 2.58 vs. 3.37. The month-by-month view shows why:

| Test month | Baseline R² | Model R² | Baseline MAE | Model MAE |
|---|---|---|---|---|
| 2019-11 | 0.867 | 0.926 | 4.07 | 2.81 |
| 2020-01 | 0.850 | 0.894 | 3.35 | 2.83 |
| 2020-03 | 0.901 | 0.910 | 3.73 | 4.09 |
| 2020-07 | 0.851 | 0.866 | 3.70 | 3.88 |
| 2020-09 | 0.887 | 0.881 | 2.51 | 3.43 |

The model is clearly ahead in the two months that resemble its training data. From March 2020 its lead disappears, and on MAE the "same as last month" guess is better. Two things change at that point: the source only has data for scattered months of 2020, and those months fall in the pandemic (average sales per item jump in March 2020). A model trained on 2017–2019 does not transfer cleanly to them.

**Compared with the first version.** The earlier R² of 0.8875 came from a random split and is not comparable. On the same kind of rows it used (non-zero sales only), this model scores 0.842 on the test months.

**One disclosure.** The first full run exposed a flaw in a feature: the gap since an item's last appearance was counted in calendar months, so months missing from the source looked like the item being absent. It is now counted in months of data. The flaw was found by looking at test-month results, so the test score is not perfectly untouched. Validation scores did not move with the fix.

Full tables, including scores by item type: [`reports/results.md`](reports/results.md).
<!-- RESULTS:END -->

---

## What changed from the first version, and why

The first version of this project reported R² = 0.8875. That number did not measure what it appeared to, for four reasons, all now fixed:

| Problem | Why it inflated the score | Fix |
|---|---|---|
| Random train/test split | Rows from the same item and the same months landed on both sides, so the model was tested on months it had effectively seen | Split by time: train on the earliest months, validate on the next, test on the latest |
| Same-month warehouse sales as a feature | That figure is not known until the month is over, so it cannot be used to forecast the month | Only prior-month history is used as input |
| Model chosen on the test set | Picking the best of seven models by test score makes the test score optimistic | Models are compared on validation months; the test months are scored once |
| Zero-sales months dropped | Whether an item sells is part of what is being predicted; dropping those rows filters on the answer | Zero-sales months are kept |

Results are also reported against two model-free baselines, because a score only means something next to the obvious guess.

---

## Demo

```bash
streamlit run app.py
```

Pick a supplier and an item, choose a month after the item's last recorded month, and get a forecast along with the history it is based on. Train the model first (see Setup).

---

## Project Structure

```
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Loading, cleaning, time-based split
│   │   ├── data_transformation.py  # Prior-month history features, preprocessing
│   │   └── model_trainer.py        # Baselines, model selection on validation, final test score
│   ├── pipeline/
│   │   ├── train_pipeline.py       # Runs everything end to end
│   │   └── predict_pipeline.py     # Forecast for one item and month
│   ├── logger.py
│   └── utils.py                    # save/load, scoring (R², MAE, WAPE)
├── tests/                          # pytest: features, leakage guard, full run, the app itself
├── reports/                        # results.md and metrics.json from the latest full run
├── data/                           # raw CSV goes here (not tracked in git)
├── artifacts/                      # model.pkl, item_state.csv (generated, not tracked)
├── .github/workflows/ci.yml
├── app.py                          # Streamlit app
├── requirements.txt
└── README.md
```

---

## Dataset

**Source:** [Montgomery County Open Data — Warehouse and Retail Sales](https://data.montgomerycountymd.gov/w/v76h-r7br)
> **Note:** Copy and paste the dataset URL directly into your browser —
> clicking the link may return a 403 error due to portal restrictions.

- 307,645 records of monthly item-level sales across suppliers and product categories
- Covers LIQUOR, WINE, BEER, NON-ALCOHOL, KEGS, and store supplies
- 24 months of data between June 2017 and September 2020; many calendar months are missing from the source

**Cleaning:**
- Dropped KEGS, DUNNAGE, STR_SUPPLIES, REF (not retail products)
- Removed null and "Default" supplier rows
- Combined duplicate rows for the same item and month
- Kept months with zero retail sales

---

## Approach

### Features
Every history feature describes months **before** the month being forecast.

| Feature | Notes |
|---|---|
| PREV_RETAIL_SALES | The item's retail sales in its most recent earlier month |
| PREV_RETAIL_TRANSFERS | Transfers to stores in that month |
| PREV_WAREHOUSE_SALES | Warehouse sales in that month |
| ITEM_MEAN_RETAIL_PRIOR | The item's average retail sales over all earlier months |
| ITEM_MONTHS_OBSERVED | How many earlier months the item appears in (0 = new item) |
| PERIODS_SINCE_LAST | How many months of data ago that was (months missing from the source are not counted) |
| MONTH | Calendar month of the forecast |
| ITEM TYPE | LIQUOR / WINE / BEER / NON-ALCOHOL |
| SUPPLIER | One-hot; rare and unseen suppliers are grouped |

### Target
`RETAIL SALES` — log-transformed (`np.log1p`) during training to handle right skew; forecasts are inverse-transformed (`np.expm1`) before display. Months of net returns (negative sales) are floored at zero.

### Evaluation
- **Split by time:** the earliest ~60% of months for training, the next ~20% for validation, the latest ~20% for testing
- **Baselines:** "same as the item's last recorded month" and "the item's historical average"
- **Models compared on validation:** Ridge Regression, Random Forest, Gradient Boosting (scikit-learn), XGBoost
- **Metrics:** R² on the log scale, MAE in units, and WAPE (total absolute error ÷ total units sold)
- The selected model is refit on train + validation and scored once on the test months

The preprocessing and the model are saved as a single scikit-learn pipeline, so training and the app cannot drift apart.

---

## Setup

```bash
git clone https://github.com/sejalnimkar/retail-sales-ml-pipeline.git
cd retail-sales-ml-pipeline
pip install -r requirements.txt
```

Download the dataset from [Montgomery County Open Data](https://data.montgomerycountymd.gov/d/v76h-r7br) and place it at:
```
data/Warehouse_and_Retail_Sales.csv
```

**Train and evaluate:**
```bash
python -m src.pipeline.train_pipeline --data data/Warehouse_and_Retail_Sales.csv
```
This cleans the data, builds features, selects a model, writes `reports/results.md`, and saves `artifacts/model.pkl` and `artifacts/item_state.csv`.

**Run the app:**
```bash
streamlit run app.py
```

**Run the tests** (no dataset needed; they use synthetic data):
```bash
pip install -r requirements-dev.txt
pytest -q
```

---

## Limitations

- **Thin margin under change.** The model's edge over "same as last month" is small on the 2020 test months (see Results). For a real decision, the baseline is a serious competitor.
- **Short, gappy history.** The source covers parts of four years and skips months, so yearly seasonality cannot be learned well.
- **New items.** An item with no history is forecast from its type and supplier only.
- **One month ahead.** Forecasting further out reuses the same last-known history, so accuracy will fall with distance.
