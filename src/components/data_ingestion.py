"""
Loading, cleaning and time-based splitting of the Montgomery County
warehouse and retail sales data.

The split is by calendar period, never random: the model is always scored on
months that come after every month it was trained on, which is how it would
be used in practice.
"""

import pandas as pd

from src.logger import logger

RAW_COLUMNS = [
    "YEAR", "MONTH", "SUPPLIER", "ITEM CODE", "ITEM DESCRIPTION", "ITEM TYPE",
    "RETAIL SALES", "RETAIL TRANSFERS", "WAREHOUSE SALES",
]
SALES_COLUMNS = ["RETAIL SALES", "RETAIL TRANSFERS", "WAREHOUSE SALES"]

# Kegs, dunnage, store supplies and reference rows are not retail products.
VALID_ITEM_TYPES = {"WINE", "BEER", "LIQUOR", "NON-ALCOHOL"}


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ITEM CODE": str})
    df.columns = [c.strip().upper() for c in df.columns]
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {sorted(missing)}")
    logger.info("Raw data loaded: %s rows", f"{len(df):,}")
    return df[RAW_COLUMNS]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per item per month.

    Unlike the first version of this project, months where an item sold
    nothing at retail are KEPT. Whether an item sells is not known in advance,
    so dropping those rows would mean filtering on the answer.
    """
    df = df.copy()
    df["SUPPLIER"] = df["SUPPLIER"].astype("string").str.strip()
    df = df[df["SUPPLIER"].notna() & ~df["SUPPLIER"].str.lower().eq("default")]
    df = df[df["ITEM TYPE"].isin(VALID_ITEM_TYPES) & df["ITEM CODE"].notna()]
    df[SALES_COLUMNS] = df[SALES_COLUMNS].fillna(0.0)

    # A few items appear more than once in a month; combine them.
    df = df.groupby(["YEAR", "MONTH", "ITEM CODE"], as_index=False).agg(
        {
            "SUPPLIER": "first",
            "ITEM DESCRIPTION": "first",
            "ITEM TYPE": "first",
            "RETAIL SALES": "sum",
            "RETAIL TRANSFERS": "sum",
            "WAREHOUSE SALES": "sum",
        }
    )
    df["SUPPLIER"] = df["SUPPLIER"].astype(object)
    # Months since year 0: lets us measure gaps, since the source skips some months.
    df["PERIOD"] = df["YEAR"].astype(int) * 12 + df["MONTH"].astype(int) - 1
    df = df.sort_values(["PERIOD", "ITEM CODE"]).reset_index(drop=True)
    logger.info("Cleaned data: %s item-months over %d months", f"{len(df):,}", df["PERIOD"].nunique())
    return df


def split_periods(periods, val_fraction: float = 0.2, test_fraction: float = 0.2):
    """Split the sorted distinct periods into (train, validation, test), oldest first."""
    periods = sorted(set(periods))
    if len(periods) < 3:
        raise ValueError("need at least three distinct months to make a train/validation/test split")
    n_test = max(1, round(len(periods) * test_fraction))
    n_val = max(1, round(len(periods) * val_fraction))
    if n_test + n_val >= len(periods):
        raise ValueError("not enough months left for training after the validation and test splits")
    train = periods[: len(periods) - n_val - n_test]
    val = periods[len(periods) - n_val - n_test : len(periods) - n_test]
    test = periods[len(periods) - n_test :]
    return train, val, test


def period_label(period: int) -> str:
    return f"{period // 12}-{period % 12 + 1:02d}"
