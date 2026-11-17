"""
Feature engineering and preprocessing.

Every history feature for a row describes months BEFORE that row's month,
so nothing about the month being predicted (including its warehouse sales,
which the first version of this project used) can reach the model.

History also stops at a break in the source: when more than
MAX_GAP_MONTHS calendar months pass with no data at all (the county file
jumps from September 2020 to June 2026), sales from before the break are not
treated as an item's "last month". Sales quantities are in cases.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET_COLUMN = "RETAIL SALES"

HISTORY_FEATURES = [
    "PREV_RETAIL_SALES",       # the item's retail sales in its most recent earlier month
    "PREV_RETAIL_TRANSFERS",
    "PREV_WAREHOUSE_SALES",
    "RETAIL_MEAN_LAST3",       # the item's average retail sales over its last 3 earlier months
    "ITEM_MONTHS_OBSERVED",    # how many earlier months the item appears in (0 = new item)
    "PERIODS_SINCE_LAST",      # how many months of data ago that was (1 = the previous month in the data)
]
SALES_HISTORY_COLUMNS = HISTORY_FEATURES[:4]
OTHER_NUMERICAL_COLUMNS = HISTORY_FEATURES[4:] + ["MONTH"]
NUMERICAL_COLUMNS = SALES_HISTORY_COLUMNS + OTHER_NUMERICAL_COLUMNS
CATEGORICAL_COLUMNS = ["ITEM TYPE", "SUPPLIER"]
FEATURE_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS

# Suppliers beyond the most common ones are grouped together by the encoder.
MAX_SUPPLIER_CATEGORIES = 31

# A hole in the source longer than this starts a new segment with no history.
# The county file has an 11-month hole (2018-02 to 2019-01), kept as one
# segment, and a 69-month hole (2020-09 to 2026-06), which is a break.
MAX_GAP_MONTHS = 12


def assign_segments(periods: pd.Series, max_gap: int = MAX_GAP_MONTHS) -> pd.Series:
    """Number the stretches of data between breaks: 0, 1, 2, ..."""
    distinct = np.sort(periods.unique())
    segment_of = dict(zip(distinct, np.concatenate([[0], np.cumsum(np.diff(distinct) > max_gap)]), strict=True))
    return periods.map(segment_of).astype(int)


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the HISTORY_FEATURES columns. Expects one row per (ITEM CODE, PERIOD)."""
    df = df.sort_values(["ITEM CODE", "PERIOD"]).copy()
    df["PERIOD_INDEX"] = df["PERIOD"].rank(method="dense").astype(int)
    df["SEGMENT"] = assign_segments(df["PERIOD"])
    by_item = df.groupby(["ITEM CODE", "SEGMENT"], sort=False)

    df["PREV_RETAIL_SALES"] = by_item["RETAIL SALES"].shift(1)
    df["PREV_RETAIL_TRANSFERS"] = by_item["RETAIL TRANSFERS"].shift(1)
    df["PREV_WAREHOUSE_SALES"] = by_item["WAREHOUSE SALES"].shift(1)
    df["ITEM_MONTHS_OBSERVED"] = by_item.cumcount()
    last3 = pd.concat([by_item["RETAIL SALES"].shift(k) for k in (1, 2, 3)], axis=1)
    df["RETAIL_MEAN_LAST3"] = last3.mean(axis=1, skipna=True)
    # Counted in months that exist in the data, not calendar months. The source
    # skips some months entirely; an item present on both sides of such a hole
    # has not been "away", and a calendar gap would wrongly say it had.
    df["PERIODS_SINCE_LAST"] = df["PERIOD_INDEX"] - by_item["PERIOD_INDEX"].shift(1)
    return df.sort_values(["PERIOD", "ITEM CODE"]).reset_index(drop=True)


def to_target(retail_sales) -> np.ndarray:
    """log1p of sales, with returns (negative months) floored at zero."""
    return np.log1p(np.clip(np.asarray(retail_sales, dtype=float), 0, None))


def from_target(log_sales) -> np.ndarray:
    return np.clip(np.expm1(np.asarray(log_sales, dtype=float)), 0, None)


def signed_log1p(values):
    """log1p that also accepts the occasional negative (a month of net returns)."""
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def build_preprocessor() -> ColumnTransformer:
    """
    Sales history: log scale (it is as skewed as the target) -> median impute -> scale.
    Other numerical: median impute (new items have no history) -> scale.
    Categorical: one-hot; rare suppliers are grouped, and a supplier never
    seen in training is treated as rare rather than causing an error.
    Fitted inside the model pipeline, so it only ever learns from training rows.
    """
    sales_history = Pipeline(
        steps=[
            ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    numerical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        max_categories=MAX_SUPPLIER_CATEGORIES,
        sparse_output=False,
    )
    return ColumnTransformer(
        transformers=[
            ("sales_history", sales_history, SALES_HISTORY_COLUMNS),
            ("numerical", numerical, OTHER_NUMERICAL_COLUMNS),
            ("categorical", categorical, CATEGORICAL_COLUMNS),
        ]
    )


def latest_item_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per item summarising everything known about it so far. This is
    what the app needs to forecast a future month: it holds the same
    quantities add_history_features would compute for a new row appended
    after the item's last month.
    """
    df = df.sort_values(["ITEM CODE", "PERIOD"]).copy()
    df["PERIOD_INDEX"] = df["PERIOD"].rank(method="dense").astype(int)
    df["SEGMENT"] = assign_segments(df["PERIOD"])
    data_end = df["PERIOD"].max()
    newest = df["PERIOD_INDEX"].max()
    # Only the latest stretch of data counts as an item's history.
    df = df[df["SEGMENT"] == df["SEGMENT"].max()]
    by_item = df.groupby("ITEM CODE", sort=False)
    last = by_item.tail(1).set_index("ITEM CODE")
    return pd.DataFrame(
        {
            "ITEM DESCRIPTION": last["ITEM DESCRIPTION"],
            "ITEM TYPE": last["ITEM TYPE"],
            "SUPPLIER": last["SUPPLIER"],
            "LAST_PERIOD": last["PERIOD"],
            "DATA_END_PERIOD": data_end,
            # If the forecast month were the next month of data, this would be its PERIODS_SINCE_LAST.
            "PERIODS_SINCE_LAST": newest - last["PERIOD_INDEX"] + 1,
            "PREV_RETAIL_SALES": last["RETAIL SALES"],
            "PREV_RETAIL_TRANSFERS": last["RETAIL TRANSFERS"],
            "PREV_WAREHOUSE_SALES": last["WAREHOUSE SALES"],
            "RETAIL_MEAN_LAST3": by_item.tail(3).groupby("ITEM CODE", sort=False)["RETAIL SALES"].mean(),
            "ITEM_MONTHS_OBSERVED": by_item.size(),
        }
    ).reset_index()


def features_for_month(item_state: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """
    Turn item-state rows into model inputs for a forecast of (year, month).
    The forecast is treated as the next month of data after the data ends.
    """
    period = int(year) * 12 + int(month) - 1
    frame = item_state.copy()
    if (frame["DATA_END_PERIOD"] >= period).any():
        raise ValueError("the forecast month must be after the last month in the data")
    if (period - frame["DATA_END_PERIOD"] > MAX_GAP_MONTHS).any():
        raise ValueError(f"the forecast month must be within {MAX_GAP_MONTHS} months of the end of the data")
    frame["MONTH"] = int(month)
    return frame[FEATURE_COLUMNS]
