"""
Feature engineering and preprocessing.

Every history feature for a row describes months BEFORE that row's month,
so nothing about the month being predicted (including its warehouse sales,
which the first version of this project used) can reach the model.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "RETAIL SALES"

HISTORY_FEATURES = [
    "PREV_RETAIL_SALES",       # the item's retail sales in its most recent earlier month
    "PREV_RETAIL_TRANSFERS",
    "PREV_WAREHOUSE_SALES",
    "ITEM_MEAN_RETAIL_PRIOR",  # the item's average retail sales over all earlier months
    "ITEM_MONTHS_OBSERVED",    # how many earlier months the item appears in (0 = new item)
    "MONTHS_SINCE_LAST",       # gap to that most recent month (the data skips some months)
]
NUMERICAL_COLUMNS = HISTORY_FEATURES + ["MONTH"]
CATEGORICAL_COLUMNS = ["ITEM TYPE", "SUPPLIER"]
FEATURE_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS

# Suppliers beyond the most common ones are grouped together by the encoder.
MAX_SUPPLIER_CATEGORIES = 31


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the HISTORY_FEATURES columns. Expects one row per (ITEM CODE, PERIOD)."""
    df = df.sort_values(["ITEM CODE", "PERIOD"]).copy()
    by_item = df.groupby("ITEM CODE", sort=False)

    df["PREV_RETAIL_SALES"] = by_item["RETAIL SALES"].shift(1)
    df["PREV_RETAIL_TRANSFERS"] = by_item["RETAIL TRANSFERS"].shift(1)
    df["PREV_WAREHOUSE_SALES"] = by_item["WAREHOUSE SALES"].shift(1)
    df["ITEM_MONTHS_OBSERVED"] = by_item.cumcount()
    # cumsum includes the current row, so subtract it to get "earlier months only".
    prior_total = by_item["RETAIL SALES"].cumsum() - df["RETAIL SALES"]
    df["ITEM_MEAN_RETAIL_PRIOR"] = (prior_total / df["ITEM_MONTHS_OBSERVED"]).where(df["ITEM_MONTHS_OBSERVED"] > 0)
    df["MONTHS_SINCE_LAST"] = df["PERIOD"] - by_item["PERIOD"].shift(1)
    return df.sort_values(["PERIOD", "ITEM CODE"]).reset_index(drop=True)


def to_target(retail_sales) -> np.ndarray:
    """log1p of sales, with returns (negative months) floored at zero."""
    return np.log1p(np.clip(np.asarray(retail_sales, dtype=float), 0, None))


def from_target(log_sales) -> np.ndarray:
    return np.clip(np.expm1(np.asarray(log_sales, dtype=float)), 0, None)


def build_preprocessor() -> ColumnTransformer:
    """
    Numerical: median impute (new items have no history) -> scale.
    Categorical: one-hot; rare suppliers are grouped, and a supplier never
    seen in training is treated as rare rather than causing an error.
    Fitted inside the model pipeline, so it only ever learns from training rows.
    """
    numerical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
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
            ("numerical", numerical, NUMERICAL_COLUMNS),
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
    df = df.sort_values(["ITEM CODE", "PERIOD"])
    by_item = df.groupby("ITEM CODE", sort=False)
    last = by_item.tail(1).set_index("ITEM CODE")
    return pd.DataFrame(
        {
            "ITEM DESCRIPTION": last["ITEM DESCRIPTION"],
            "ITEM TYPE": last["ITEM TYPE"],
            "SUPPLIER": last["SUPPLIER"],
            "LAST_PERIOD": last["PERIOD"],
            "PREV_RETAIL_SALES": last["RETAIL SALES"],
            "PREV_RETAIL_TRANSFERS": last["RETAIL TRANSFERS"],
            "PREV_WAREHOUSE_SALES": last["WAREHOUSE SALES"],
            "ITEM_MEAN_RETAIL_PRIOR": by_item["RETAIL SALES"].mean(),
            "ITEM_MONTHS_OBSERVED": by_item.size(),
        }
    ).reset_index()


def features_for_month(item_state: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """Turn item-state rows into model inputs for a forecast of (year, month)."""
    period = int(year) * 12 + int(month) - 1
    frame = item_state.copy()
    if (frame["LAST_PERIOD"] >= period).any():
        raise ValueError("the forecast month must be after the item's last recorded month")
    frame["MONTH"] = int(month)
    frame["MONTHS_SINCE_LAST"] = period - frame["LAST_PERIOD"]
    return frame[FEATURE_COLUMNS]
