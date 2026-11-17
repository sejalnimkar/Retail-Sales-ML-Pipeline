import numpy as np
import pandas as pd
import pytest

from src.components.data_ingestion import clean
from src.components.data_transformation import (
    FEATURE_COLUMNS,
    HISTORY_FEATURES,
    add_history_features,
    build_preprocessor,
    features_for_month,
    from_target,
    latest_item_state,
    to_target,
)


def tiny():
    """
    Item A is in the data for periods 100, 101 and 105. Item B is there for
    101, 104 and 105. Nothing at all exists for periods 102 and 103: the
    source skipped them.
    """
    return pd.DataFrame(
        {
            "ITEM CODE": ["A", "A", "A", "B", "B", "B"],
            "PERIOD": [100, 101, 105, 101, 104, 105],
            "RETAIL SALES": [10.0, 20.0, 60.0, 1.0, 2.0, 3.0],
            "RETAIL TRANSFERS": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            "WAREHOUSE SALES": [5.0, 6.0, 7.0, 0.0, 0.0, 0.0],
        }
    )


def item(frame, code):
    return frame[frame["ITEM CODE"] == code].sort_values("PERIOD").reset_index(drop=True)


def test_history_features_by_hand():
    out = item(add_history_features(tiny()), "A")
    first, second, third = (out.iloc[i] for i in range(3))
    assert first[HISTORY_FEATURES].drop("ITEM_MONTHS_OBSERVED").isna().all()  # new item: nothing known
    assert first["ITEM_MONTHS_OBSERVED"] == 0
    assert (second["PREV_RETAIL_SALES"], second["ITEM_MEAN_RETAIL_PRIOR"], second["PERIODS_SINCE_LAST"]) == (10, 10, 1)
    assert (third["PREV_RETAIL_SALES"], third["PREV_WAREHOUSE_SALES"]) == (20, 6)
    assert third["ITEM_MEAN_RETAIL_PRIOR"] == 15      # mean of 10 and 20, not including 60
    # A missed one month of data (104). The months the source skipped (102, 103) do not count.
    assert (third["ITEM_MONTHS_OBSERVED"], third["PERIODS_SINCE_LAST"]) == (2, 2)


def test_months_missing_from_the_source_do_not_count_as_absence():
    b = item(add_history_features(tiny()), "B")
    # 101 -> 104 spans the hole in the source, but 104 is the very next month of data.
    assert list(b["PERIODS_SINCE_LAST"].iloc[1:]) == [1, 1]


def test_a_rows_features_do_not_depend_on_that_rows_own_sales():
    """The leakage guard: change the month being predicted; its features must not move."""
    base = item(add_history_features(tiny()), "A")
    changed = tiny()
    changed.loc[2, ["RETAIL SALES", "RETAIL TRANSFERS", "WAREHOUSE SALES"]] = [9999.0, 9999.0, 9999.0]
    changed = item(add_history_features(changed), "A")
    pd.testing.assert_series_equal(base.iloc[2][HISTORY_FEATURES], changed.iloc[2][HISTORY_FEATURES])


def test_same_month_sales_columns_are_not_model_inputs():
    assert not {"RETAIL SALES", "RETAIL TRANSFERS", "WAREHOUSE SALES"} & set(FEATURE_COLUMNS)


def test_items_do_not_share_history(raw):
    out = add_history_features(clean(raw))
    first_rows = out.sort_values("PERIOD").groupby("ITEM CODE").head(1)
    assert first_rows["PREV_RETAIL_SALES"].isna().all()


def test_item_state_matches_training_features(raw):
    """Serving must build the same numbers training would for the next month."""
    cleaned = clean(raw)
    code = cleaned["ITEM CODE"].iloc[0]
    future = cleaned[cleaned["ITEM CODE"] == code].tail(1).copy()
    future["PERIOD"] = cleaned["PERIOD"].max() + 3
    future["MONTH"] = future["PERIOD"] % 12 + 1
    expected = add_history_features(pd.concat([cleaned, future])).iloc[-1]

    state = latest_item_state(cleaned)
    period = int(future["PERIOD"].iloc[0])
    served = features_for_month(state[state["ITEM CODE"] == code], period // 12, period % 12 + 1).iloc[0]
    for col in HISTORY_FEATURES + ["MONTH"]:
        assert served[col] == pytest.approx(expected[col]), col


def test_cannot_forecast_a_month_already_in_the_data(raw):
    state = latest_item_state(clean(raw)).head(1)
    last = int(state["DATA_END_PERIOD"].iloc[0])
    with pytest.raises(ValueError):
        features_for_month(state, last // 12, last % 12 + 1)


def test_target_round_trip_and_returns_floor_at_zero():
    assert from_target(to_target([0.0, 5.0, 120.5])) == pytest.approx([0.0, 5.0, 120.5])
    assert to_target([-3.0])[0] == 0.0


def test_preprocessor_accepts_unseen_supplier_and_new_items(raw):
    featured = add_history_features(clean(raw))
    pre = build_preprocessor().fit(featured[FEATURE_COLUMNS])
    row = featured[FEATURE_COLUMNS].head(1).copy()
    row["SUPPLIER"] = "A SUPPLIER NOBODY HAS SEEN"
    row[HISTORY_FEATURES] = np.nan
    assert np.isfinite(pre.transform(row)).all()
