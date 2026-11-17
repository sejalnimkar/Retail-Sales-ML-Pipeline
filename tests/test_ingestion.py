import pandas as pd
import pytest

from src.components.data_ingestion import clean, load_raw, period_label, split_periods, to_number


def test_clean_drops_invalid_rows_and_keeps_zero_sales(raw):
    cleaned = clean(raw)
    codes = set(cleaned["ITEM CODE"])
    assert not {"90001", "90002", "90003"} & codes   # null supplier, "Default" supplier, kegs
    assert "90004" in codes                          # a month of returns is kept
    assert (cleaned["RETAIL SALES"] == 0).any()      # zero-sales months are kept
    assert not cleaned.duplicated(["ITEM CODE", "PERIOD"]).any()


def test_clean_combines_duplicate_item_months(raw):
    row = clean(raw).query("`ITEM CODE` == '90005'").iloc[0]
    assert (row["RETAIL SALES"], row["WAREHOUSE SALES"]) == (7.0, 4.0)


def test_load_raw_rejects_a_file_with_missing_columns(tmp_path, raw):
    path = tmp_path / "bad.csv"
    raw.drop(columns=["RETAIL SALES"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="RETAIL SALES"):
        load_raw(str(path))


def test_split_periods_is_ordered_and_disjoint():
    train, val, test = split_periods(range(100, 116))
    assert max(train) < min(val) <= max(val) < min(test)
    assert len(train) + len(val) + len(test) == 16
    assert (len(val), len(test)) == (3, 3)


def test_split_periods_needs_enough_months():
    with pytest.raises(ValueError):
        split_periods([1, 2])


def test_period_label():
    assert period_label(2019 * 12 + 0) == "2019-01"
    assert period_label(2020 * 12 + 11) == "2020-12"


def test_to_number_reads_text_sales_values():
    raw = pd.Series(["1,234.50", " 7 ", "", "N/A", "-3", "abc"], dtype="string")
    out = to_number(raw, "RETAIL SALES")
    assert out.iloc[:2].tolist() == [1234.5, 7.0]
    assert out.iloc[2:4].isna().all() and out.iloc[4] == -3.0 and pd.isna(out.iloc[5])


def test_clean_handles_a_file_where_every_column_is_text(raw, tmp_path):
    """Some downloads of the county file come with numbers stored as text."""
    as_text = raw.astype(str)
    as_text.loc[as_text.index[0], "RETAIL SALES"] = "1,000.00"
    as_text.loc[as_text.index[1], "MONTH"] = ""
    path = tmp_path / "text.csv"
    as_text.to_csv(path, index=False)
    loaded = pd.read_csv(path, dtype=str)
    cleaned = clean(loaded)
    assert cleaned["RETAIL SALES"].dtype == float
    assert cleaned["RETAIL SALES"].max() >= 1000.0
    assert len(cleaned) > 0
