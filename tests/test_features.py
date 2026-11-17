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
    """One item over three months, with a two-month gap before the last."""
    return pd.DataFrame(
        {
            "ITEM CODE": ["A", "A", "A"],
            "PERIOD": [100, 101, 103],
            "RETAIL SALES": [10.0, 20.0, 60.0],
            "RETAIL TRANSFERS": [1.0, 2.0, 3.0],
            "WAREHOUSE SALES": [5.0, 6.0, 7.0],
        }
    )


def test_history_features_by_hand():
    out = add_history_features(tiny())
    first, second, third = (out.iloc[i] for i in range(3))
    assert first[HISTORY_FEATURES].drop("ITEM_MONTHS_OBSERVED").isna().all()  # new item: nothing known
    assert first["ITEM_MONTHS_OBSERVED"] == 0
    assert (second["PREV_RETAIL_SALES"], second["ITEM_MEAN_RETAIL_PRIOR"], second["MONTHS_SINCE_LAST"]) == (10, 10, 1)
    assert (third["PREV_RETAIL_SALES"], third["PREV_WAREHOUSE_SALES"]) == (20, 6)
    assert third["ITEM_MEAN_RETAIL_PRIOR"] == 15      # mean of 10 and 20, not including 60
    assert (third["ITEM_MONTHS_OBSERVED"], third["MONTHS_SINCE_LAST"]) == (2, 2)


def test_a_rows_features_do_not_depend_on_that_rows_own_sales():
    """The leakage guard: change the month being predicted; its features must not move."""
    base = add_history_features(tiny())
    changed = tiny()
    changed.loc[2, ["RETAIL SALES", "RETAIL TRANSFERS", "WAREHOUSE SALES"]] = [9999.0, 9999.0, 9999.0]
    changed = add_history_features(changed)
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
    item = cleaned[cleaned["ITEM CODE"] == code]
    future = item.tail(1).copy()
    future["PERIOD"] = item["PERIOD"].max() + 3
    future["MONTH"] = future["PERIOD"] % 12 + 1
    expected = add_history_features(pd.concat([item, future])).iloc[-1]

    state = latest_item_state(cleaned)
    period = int(future["PERIOD"].iloc[0])
    served = features_for_month(state[state["ITEM CODE"] == code], period // 12, period % 12 + 1).iloc[0]
    for col in HISTORY_FEATURES + ["MONTH"]:
        assert served[col] == pytest.approx(expected[col]), col


def test_cannot_forecast_a_month_already_in_the_data(raw):
    state = latest_item_state(clean(raw)).head(1)
    last = int(state["LAST_PERIOD"].iloc[0])
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
