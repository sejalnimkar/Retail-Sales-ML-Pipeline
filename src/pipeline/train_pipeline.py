"""
Runs the whole pipeline: load -> clean -> features -> time split ->
model selection on validation -> one final score on test -> save artifacts.

Usage:
    python -m src.pipeline.train_pipeline --data data/Warehouse_and_Retail_Sales.csv
"""

import argparse
import json
import os

from src.components.data_ingestion import clean, load_raw, period_label, split_periods
from src.components.data_transformation import add_history_features, latest_item_state
from src.components.model_trainer import candidate_models, fit, make_pipeline, train_and_evaluate, walk_forward
from src.logger import logger
from src.utils import save_object

ARTIFACTS_DIR = "artifacts"
REPORTS_DIR = "reports"


def write_report(results: dict, path: str) -> None:
    s = results["split"]
    lines = [
        "# Results",
        "",
        "Sales are in cases. Part 1 is the first stretch of data; part 2 (if present) is after the break.",
        "",
        f"Train {s['train'][0]} to {s['train'][1]} ({s['train_rows']:,} item-months), "
        f"validation {s['validation'][0]} to {s['validation'][1]} ({s['validation_rows']:,}), "
        f"test {s['test'][0]} to {s['test'][1]} ({s['test_rows']:,}).",
        "",
        "## Part 1: model selection (validation months)",
        "",
        "| Method | R² (log scale) | MAE (cases) | WAPE |",
        "|---|---|---|---|",
    ]
    for name, m in sorted(results["validation"].items(), key=lambda kv: -kv[1]["r2_log"]):
        lines.append(f"| {name} | {m['r2_log']:.4f} | {m['mae']:.2f} | {m['wape']:.1%} |")
    lines += [
        "",
        f"## Part 1: final score (test months; selected model: {results['selected_model']})",
        "",
        "| Method | R² (log scale) | MAE (cases) | WAPE |",
        "|---|---|---|---|",
    ]
    for name, m in sorted(results["test"].items(), key=lambda kv: -kv[1]["r2_log"]):
        lines.append(f"| {name} | {m['r2_log']:.4f} | {m['mae']:.2f} | {m['wape']:.1%} |")
    m = results["test_rows_with_sales_only"]
    lines += [
        "",
        f"On test rows with non-zero sales only: R² (log) {m['r2_log']:.4f}, MAE {m['mae']:.2f}, WAPE {m['wape']:.1%}.",
        "",
        "## Part 1: selected model vs. last-month baseline, by test month",
        "",
        "| Month | Item-months | Baseline R² (log) | Model R² (log) | Baseline MAE | Model MAE |",
        "|---|---|---|---|---|---|",
        *[
            f"| {r['month']} | {r['rows']:,} | {r['baseline_last_month']['r2_log']:.4f} | {r['model']['r2_log']:.4f} "
            f"| {r['baseline_last_month']['mae']:.2f} | {r['model']['mae']:.2f} |"
            for r in results["test_by_month"]
        ],
        "",
        "## Part 1: selected model by item type (test months)",
        "",
        "| Item type | Item-months | R² (log scale) | MAE (cases) | WAPE |",
        "|---|---|---|---|---|",
    ]
    for item_type, m in results["test_by_item_type"].items():
        lines.append(f"| {item_type} | {m['rows']:,} | {m['r2_log']:.4f} | {m['mae']:.2f} | {m['wape']:.1%} |")
    if results.get("walk_forward"):
        title = "Part 2: after the break in the data, month by month"
        lines += ["", *walk_forward_lines(results["walk_forward"], title)]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def walk_forward_lines(wf: dict, title: str) -> list[str]:
    """Markdown for a walk-forward result: overall table, then one row per month."""
    lines = [
        f"## {title}",
        "",
        f"Each month is forecast by {wf['model']} trained on every earlier month.",
        "",
        "| Method | R² (log scale) | MAE (cases) | WAPE |",
        "|---|---|---|---|",
    ]
    for name, m in sorted(wf["overall"].items(), key=lambda kv: kv[1]["mae"]):
        lines.append(f"| {name} | {m['r2_log']:.4f} | {m['mae']:.2f} | {m['wape']:.1%} |")
    lines += [
        "",
        "| Month | Item-months | Model MAE | Last-month baseline MAE | 3-month-average baseline MAE |",
        "|---|---|---|---|---|",
    ]
    for r in wf["by_month"]:
        last_month = r["Baseline: last observed month"]["mae"]
        last3 = r["Baseline: average of last 3 months"]["mae"]
        lines.append(f"| {r['month']} | {r['rows']:,} | {r['model']['mae']:.2f} | {last_month:.2f} | {last3:.2f} |")
    return lines


def run(data_path: str, artifacts_dir: str = ARTIFACTS_DIR, reports_dir: str = REPORTS_DIR) -> dict:
    cleaned = clean(load_raw(data_path))
    featured = add_history_features(cleaned)

    # Part 1: model selection and a held-out test inside the first stretch of
    # data (2017-06 to 2020-09 in the county file).
    first_segment = featured[featured["SEGMENT"] == 0]
    train_p, val_p, test_p = split_periods(first_segment["PERIOD"])
    train = featured[featured["PERIOD"].isin(train_p)]
    val = featured[featured["PERIOD"].isin(val_p)]
    test = featured[featured["PERIOD"].isin(test_p)]
    logger.info("Split rows  train: %s  validation: %s  test: %s", f"{len(train):,}", f"{len(val):,}", f"{len(test):,}")

    _, results = train_and_evaluate(train, val, test)
    results["split"] = {
        "train": [period_label(train_p[0]), period_label(train_p[-1])],
        "validation": [period_label(val_p[0]), period_label(val_p[-1])],
        "test": [period_label(test_p[0]), period_label(test_p[-1])],
        "train_rows": len(train),
        "validation_rows": len(val),
        "test_rows": len(test),
    }

    # Part 2: every later stretch of data (2026 in the county file), forecast
    # one month at a time with the model chosen in part 1. The first month
    # after a break is skipped: no item has any history there.
    later = featured[featured["SEGMENT"] > 0]
    months = sorted(set(later["PERIOD"]) - set(later.groupby("SEGMENT")["PERIOD"].min()))
    results["walk_forward"] = walk_forward(featured, months, results["selected_model"]) if months else None

    # The app forecasts months after the data ends, so the saved model is the
    # selected one refit on every month available. The scores above are from
    # the model that had not seen the test months.
    deployed = fit(make_pipeline(candidate_models()[results["selected_model"]]), featured)

    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    save_object(os.path.join(artifacts_dir, "model.pkl"), deployed)
    latest_item_state(cleaned).to_csv(os.path.join(artifacts_dir, "item_state.csv"), index=False)
    with open(os.path.join(reports_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_report(results, os.path.join(reports_dir, "results.md"))

    best = results["test"][results["selected_model"]]
    logger.info("Test: %s  R² (log) %.4f  MAE %.2f  WAPE %.1f%%",
                results["selected_model"], best["r2_log"], best["mae"], 100 * best["wape"])
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the retail sales model.")
    parser.add_argument("--data", default=os.path.join("data", "Warehouse_and_Retail_Sales.csv"))
    args = parser.parse_args()
    run(args.data)
