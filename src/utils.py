"""Small helpers: saving/loading objects and scoring forecasts."""

import os
import pickle

import numpy as np
from sklearn.metrics import r2_score


def save_object(file_path: str, obj) -> None:
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)


def load_object(file_path: str):
    with open(file_path, "rb") as file_obj:
        return pickle.load(file_obj)


def score(actual_cases, predicted_cases) -> dict:
    """
    r2_log - R² on log1p(sales), the scale the model is trained on
    mae    - average miss in cases per item-month
    wape   - total absolute miss / total cases sold
    """
    actual = np.clip(np.asarray(actual_cases, dtype=float), 0, None)
    predicted = np.clip(np.asarray(predicted_cases, dtype=float), 0, None)
    if len(actual) == 0:
        raise ValueError("cannot score an empty set")
    abs_err = np.abs(actual - predicted)
    total = actual.sum()
    return {
        "r2_log": float(r2_score(np.log1p(actual), np.log1p(predicted))),
        "mae": float(abs_err.mean()),
        "wape": float(abs_err.sum() / total) if total > 0 else float("nan"),
    }
