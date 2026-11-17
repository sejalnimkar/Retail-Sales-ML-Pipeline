import numpy as np
import pandas as pd
import pytest

# Mirrors the real data: some calendar months are missing from the source.
# Like the real file: an 11-month hole that is not a break, then a multi-year break.
PERIODS = (
    [(2017, m) for m in range(6, 13)] + [(2018, 1), (2018, 2)] + [(2019, m) for m in range(1, 8)]
    + [(2026, m) for m in range(6, 10)]
)
SUPPLIERS = [f"SUPPLIER {i:02d}" for i in range(40)]
ITEM_TYPES = ["WINE", "BEER", "LIQUOR", "NON-ALCOHOL"]


def make_raw(n_items: int = 250, seed: int = 0) -> pd.DataFrame:
    """Synthetic data in the raw file's layout, including the mess clean() has to handle."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        code = str(10000 + i)
        base = rng.gamma(shape=1.2, scale=8.0)  # typical monthly retail sales for this item
        supplier, item_type = SUPPLIERS[i % len(SUPPLIERS)], ITEM_TYPES[i % len(ITEM_TYPES)]
        for year, month in PERIODS:
            if rng.random() < 0.15:  # item not stocked this month
                continue
            retail = 0.0 if rng.random() < 0.2 else round(base * rng.lognormal(0, 0.3), 2)
            rows.append([year, month, supplier, code, f"ITEM {code}", item_type,
                         retail, round(retail * 0.9, 2), round(base * rng.lognormal(0, 0.5), 2)])
    cols = ["YEAR", "MONTH", "SUPPLIER", "ITEM CODE", "ITEM DESCRIPTION", "ITEM TYPE",
            "RETAIL SALES", "RETAIL TRANSFERS", "WAREHOUSE SALES"]
    df = pd.DataFrame(rows, columns=cols)
    extra = pd.DataFrame(
        [
            [2019, 1, None, "90001", "NO SUPPLIER", "WINE", 5.0, 5.0, 1.0],
            [2019, 1, "Default", "90002", "DEFAULT SUPPLIER", "WINE", 5.0, 5.0, 1.0],
            [2019, 1, "SUPPLIER 00", "90003", "A KEG", "KEGS", 0.0, 0.0, 4.0],
            [2019, 1, "SUPPLIER 00", "90004", "RETURNED ITEM", "WINE", -2.0, 0.0, 0.0],
            [2019, 1, "SUPPLIER 00", "90005", "SPLIT ROW", "BEER", 3.0, 1.0, 2.0],
            [2019, 1, "SUPPLIER 00", "90005", "SPLIT ROW", "BEER", 4.0, 1.0, 2.0],
        ],
        columns=cols,
    )
    return pd.concat([df, extra], ignore_index=True)


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return make_raw()


@pytest.fixture(scope="session")
def raw_csv(raw, tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("data") / "sales.csv"
    raw.to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="session")
def trained(raw_csv, tmp_path_factory):
    """Run the full training pipeline once on the synthetic data."""
    from src.pipeline.train_pipeline import run

    out = tmp_path_factory.mktemp("run")
    results = run(raw_csv, artifacts_dir=str(out / "artifacts"), reports_dir=str(out / "reports"))
    return out, results
