# Montgomery County Warehouse & Retail Sales Forecaster

[![CI](https://github.com/sejalnimkar/retail-sales-ml-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/sejalnimkar/retail-sales-ml-pipeline/actions/workflows/ci.yml)

A machine learning pipeline that forecasts next-month retail sales for individual liquor, wine, beer and non-alcohol products in Montgomery County, MD, using only what is known before the month starts. Built on public county data (352K records, 2017–2026), evaluated on months the model never saw, with a Streamlit app for interactive forecasts.

---

## Results

<!-- RESULTS:START -->
Sales are in **cases**. The county file has two stretches of data: June 2017 to September 2020, then nothing until June 2026 to September 2026. The model is scored separately on each.

### After the gap: 2026, one month at a time

Each month is forecast by a Random Forest trained on every earlier month, which is how it would be used as new data arrives. June 2026 is not scored: it is the first month after the gap, so no product has recent history.

| Method | R² (log scale) | MAE (cases) | WAPE |
|---|---|---|---|
| **Random Forest** | 0.943 | 1.48 | 24.4% |
| Baseline: same as last recorded month | 0.933 | 1.56 | 25.8% |
| Baseline: average of last 3 months | 0.933 | 1.56 | 25.8% |

| Month | Model MAE | Last-month baseline MAE | 3-month-average baseline MAE |
|---|---|---|---|
| 2026-07 | 1.72 | 1.63 | 1.63 |
| 2026-08 | 1.40 | 1.69 | 1.56 |
| 2026-09 | 1.33 | 1.37 | 1.50 |

The model's average miss is 5% lower than the better baseline overall. It loses in July and wins in August and September.

### Before the gap: 2019-11 to 2020-09

The model was chosen on validation months (2019-06 to 2019-10), then scored once on these 57,314 item-months.

| Method | R² (log scale) | MAE (cases) | WAPE |
|---|---|---|---|
| **Random Forest** | 0.896 | 3.09 | 38.9% |
| Baseline: average of last 3 months | 0.881 | 3.00 | 37.8% |
| Baseline: same as last recorded month | 0.872 | 3.48 | 43.9% |

Here the picture is mixed: the model has the best R², but the 3-month average has a slightly lower average miss (3.00 vs 3.09 cases). The model leads clearly in 2019-11 and 2020-01 and is roughly level from March 2020, when pandemic-era buying changed.

### How much to trust these numbers

The features were revised twice after looking at test results (see "Second round of changes" below), so neither table is a perfectly untouched test. The clean check is the next month the county publishes: `python -m src.pipeline.score_month` scores it once, with the model type fixed in advance. That result will be added here.

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

### Second round of changes (November 2026)

A newer download of the county file added June to September 2026, after a gap of almost six years. Run unchanged, the model's average miss on the 2026 months was about double that of "same as last month". Looking at the data showed why:

- Only 47% of the products sold in 2026 also appear in 2017–2020.
- Products present in both periods sell 42% fewer cases per month in 2026.
- The model leaned on each product's all-time average, which in 2026 was mostly made of six-year-old sales levels.

The fixes:

| Change | Why |
|---|---|
| A hole of more than 12 months in the data is a break; history does not carry across it | Sales from 2020 are not "last month" for June 2026 |
| The all-time average is replaced by the average of the last 3 months | It follows changes in sales level instead of anchoring on old ones |
| Results are reported separately for each stretch of data, and the 2026 months are forecast one at a time | The model gets to learn from 2026 before being scored on it |
| A `score_month` command scores a newly published month once, without re-tuning | Gives one clean test that no design decision has seen |

The newer file also writes negative numbers in accounting style, like `(2.00)`, and stores some numbers as text; the loader now reads both.

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
│   │   ├── score_month.py          # Scores one newly published month, once
│   │   └── predict_pipeline.py     # Forecast for one item and month
│   ├── logger.py
│   └── utils.py                    # save/load, scoring (R², MAE, WAPE)
├── notebook/
│   └── montgomery_eda.ipynb        # Exploratory analysis of the county data
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

- 352,419 records of monthly item-level sales across suppliers and product categories, in cases
- Covers LIQUOR, WINE, BEER, NON-ALCOHOL, KEGS, and store supplies
- 28 months of data: June 2017 to September 2020 with several missing months, then nothing until June to September 2026
- Updated monthly by the county

**Cleaning:**
- Dropped KEGS, DUNNAGE, STR_SUPPLIES, REF (not retail products)
- Removed null and "Default" supplier rows
- Combined duplicate rows for the same item and month
- Kept months with zero retail sales
- Read numbers stored as text, including accounting-style negatives like `(2.00)`

---

## Approach

### Features
Every history feature describes months **before** the month being forecast, within the same stretch of data.

| Feature | Notes |
|---|---|
| PREV_RETAIL_SALES | The item's retail sales in its most recent earlier month |
| PREV_RETAIL_TRANSFERS | Transfers to stores in that month |
| PREV_WAREHOUSE_SALES | Warehouse sales in that month |
| RETAIL_MEAN_LAST3 | The item's average retail sales over its last 3 earlier months |
| ITEM_MONTHS_OBSERVED | How many earlier months the item appears in since the last break (0 = no history) |
| PERIODS_SINCE_LAST | How many months of data ago that was (months missing from the source are not counted) |
| MONTH | Calendar month of the forecast |
| ITEM TYPE | LIQUOR / WINE / BEER / NON-ALCOHOL |
| SUPPLIER | One-hot; rare and unseen suppliers are grouped |

### Target
`RETAIL SALES` — log-transformed (`np.log1p`) during training to handle right skew; forecasts are inverse-transformed (`np.expm1`) before display. Months of net returns (negative sales) are floored at zero.

### Evaluation
- **Before the gap:** split by time inside 2017–2020: the earliest ~60% of months for training, the next ~20% for validation, the latest ~20% for testing
- **After the gap:** each 2026 month is forecast by the selected model trained on every earlier month
- **Baselines:** "same as the item's last recorded month" and "the item's average over its last 3 months"
- **Models compared on validation:** Ridge Regression, Random Forest, Gradient Boosting (scikit-learn), XGBoost
- **Metrics:** R² on the log scale, MAE in cases, and WAPE (total absolute error ÷ total cases sold)
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

A full run takes about 10 minutes on a laptop, most of it the Random Forest refits in the 2026 evaluation.

**Score a newly published month** (once, with the model type fixed in `reports/metrics.json`):
```bash
python -m src.pipeline.score_month --data data/Warehouse_and_Retail_Sales.csv --month 2026-10
```

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

- **Thin margins.** The model's edge over simple baselines is small, and before the gap the 3-month average has a slightly lower average miss. For a real decision, the baselines are serious competitors.
- **Short, gappy history.** Only four 2026 months exist, and the source skips months, so yearly seasonality cannot be learned.
- **Large model file.** The selected Random Forest saves to about 650 MB. A smaller model (XGBoost was a close second on validation) would suit deployment better.
- **New items.** An item with no history is forecast from its type and supplier only; about half the 2026 products are new since 2020.
- **One month ahead.** Forecasting further out reuses the same last-known history, so accuracy will fall with distance.
