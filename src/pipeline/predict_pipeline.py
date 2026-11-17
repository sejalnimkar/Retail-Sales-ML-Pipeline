"""Inference: forecast an item's retail sales for a future month from its saved history."""

import os

import pandas as pd

from src.components.data_transformation import features_for_month, from_target
from src.utils import load_object

MODEL_PATH = os.path.join("artifacts", "model.pkl")
ITEM_STATE_PATH = os.path.join("artifacts", "item_state.csv")


class PredictPipeline:
    def __init__(self, model_path: str = MODEL_PATH, item_state_path: str = ITEM_STATE_PATH):
        self.model = load_object(model_path)
        self.item_state = pd.read_csv(item_state_path, dtype={"ITEM CODE": str})

    def items(self) -> pd.DataFrame:
        """Items the model has history for, for populating a picker."""
        return self.item_state[["ITEM CODE", "ITEM DESCRIPTION", "ITEM TYPE", "SUPPLIER"]]

    def history(self, item_code: str) -> pd.Series:
        match = self.item_state[self.item_state["ITEM CODE"] == str(item_code)]
        if match.empty:
            raise KeyError(f"no history for item code {item_code!r}")
        return match.iloc[0]

    def predict(self, item_code: str, year: int, month: int) -> float:
        """Forecast retail sales (cases) for one item in (year, month)."""
        self.history(item_code)  # raises a clear error for an unknown item
        state = self.item_state[self.item_state["ITEM CODE"] == str(item_code)].head(1)
        features = features_for_month(state, year, month)
        return float(from_target(self.model.predict(features))[0])
