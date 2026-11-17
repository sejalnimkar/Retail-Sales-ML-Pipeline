"""
Model selection and final evaluation.

Candidates are compared on the validation months. The winner is then refit
on train + validation and scored ONCE on the test months, which play no part
in choosing it. Two model-free baselines are scored the same way, so the
result reads as "how much better than the obvious guess", not as a bare R².
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.components.data_transformation import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
    from_target,
    to_target,
)
from src.logger import logger
from src.utils import score

BASELINES = {
    "Baseline: last observed month": "PREV_RETAIL_SALES",
    "Baseline: item's historical average": "ITEM_MEAN_RETAIL_PRIOR",
}


def candidate_models() -> dict:
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=42),
        "Gradient Boosting (sklearn)": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, n_jobs=-1, random_state=42
        ),
    }


def make_pipeline(model) -> Pipeline:
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])


def fit(pipeline: Pipeline, frame: pd.DataFrame) -> Pipeline:
    return pipeline.fit(frame[FEATURE_COLUMNS], to_target(frame[TARGET_COLUMN]))


def predict_units(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    return from_target(pipeline.predict(frame[FEATURE_COLUMNS]))


def baseline_scores(frame: pd.DataFrame) -> dict:
    # An item with no history gets a baseline forecast of zero.
    return {name: score(frame[TARGET_COLUMN], frame[col].fillna(0.0)) for name, col in BASELINES.items()}


def train_and_evaluate(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """Returns (fitted final pipeline, results dict)."""
    validation = baseline_scores(val)
    for name, model in candidate_models().items():
        pipeline = fit(make_pipeline(model), train)
        validation[name] = score(val[TARGET_COLUMN], predict_units(pipeline, val))
        logger.info("%s: validation R² (log) = %.4f", name, validation[name]["r2_log"])

    best_name = max(candidate_models(), key=lambda n: validation[n]["r2_log"])
    logger.info("Selected on validation: %s", best_name)

    final = fit(make_pipeline(candidate_models()[best_name]), pd.concat([train, val], ignore_index=True))
    test_predictions = predict_units(final, test)
    test_scores = baseline_scores(test)
    test_scores[best_name] = score(test[TARGET_COLUMN], test_predictions)

    sold = (test[TARGET_COLUMN] > 0).to_numpy()
    results = {
        "selected_model": best_name,
        "validation": validation,
        "test": test_scores,
        # Comparable to the first version of the project, which only kept rows with sales.
        "test_rows_with_sales_only": score(test.loc[sold, TARGET_COLUMN], test_predictions[sold]),
        "test_by_item_type": {
            item_type: {"rows": int(mask.sum()), **score(test.loc[mask, TARGET_COLUMN], test_predictions[mask])}
            for item_type in sorted(test["ITEM TYPE"].unique())
            for mask in [(test["ITEM TYPE"] == item_type).to_numpy()]
        },
    }
    return final, results
