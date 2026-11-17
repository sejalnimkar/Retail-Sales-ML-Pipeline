"""
Score the model on a month it has never seen, without changing anything.

Run this when the county publishes a new month. The model type and its
settings are the ones recorded in reports/metrics.json by the last full
training run; nothing is re-tuned or re-selected. The model is trained on
every month before the new one and scored once against the baselines.

Usage:
    python -m src.pipeline.score_month --data data/Warehouse_and_Retail_Sales.csv --month 2026-10
"""

import argparse
import json
import os

from src.components.data_ingestion import clean, load_raw
from src.components.data_transformation import add_history_features
from src.components.model_trainer import candidate_models, walk_forward
from src.logger import logger
from src.pipeline.train_pipeline import REPORTS_DIR, walk_forward_lines


def parse_month(text: str) -> int:
    year, month = (int(part) for part in text.split("-"))
    if not 1 <= month <= 12:
        raise ValueError(f"month must look like YYYY-MM, got {text!r}")
    return year * 12 + month - 1


def run(data_path: str, month: str, model_name: str | None = None, reports_dir: str = REPORTS_DIR) -> dict:
    if model_name is None:
        with open(os.path.join(reports_dir, "metrics.json"), encoding="utf-8") as f:
            model_name = json.load(f)["selected_model"]
    if model_name not in candidate_models():
        raise ValueError(f"unknown model {model_name!r}; choose from {sorted(candidate_models())}")

    period = parse_month(month)
    featured = add_history_features(clean(load_raw(data_path)))
    target = featured[featured["PERIOD"] == period]
    if target.empty:
        raise ValueError(f"{month} is not in {data_path}")
    if target["ITEM_MONTHS_OBSERVED"].eq(0).all():
        raise ValueError(f"{month} is the first month after a break in the data, so no item has history")

    result = walk_forward(featured, [period], model_name)
    lines = ["# Untouched month", "", *walk_forward_lines(result, f"{month}, scored once")]
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, f"score_{month}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(reports_dir, f"score_{month}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Wrote %s", os.path.join(reports_dir, f"score_{month}.md"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score the model once on a newly published month.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--month", required=True, help="Month to score, YYYY-MM")
    parser.add_argument("--model", default=None, help="Defaults to the model in reports/metrics.json")
    args = parser.parse_args()
    run(args.data, args.month, args.model)
