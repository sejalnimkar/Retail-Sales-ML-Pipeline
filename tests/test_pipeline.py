import json
import math
import os

import numpy as np
import pytest

from src.components.model_trainer import BASELINES, candidate_models
from src.pipeline.predict_pipeline import PredictPipeline
from src.utils import score


def test_score_known_values():
    s = score([10, 0, 30], [12, 0, 24])
    assert s["mae"] == pytest.approx(8 / 3)
    assert s["wape"] == pytest.approx(8 / 40)
    assert score([1, 2, 3], [1, 2, 3])["r2_log"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        score([], [])


def test_training_run_reports_every_method_and_an_ordered_split(trained):
    _, results = trained
    assert set(results["validation"]) == set(BASELINES) | set(candidate_models())
    assert results["selected_model"] in candidate_models()
    assert set(results["test"]) == set(BASELINES) | {results["selected_model"]}
    split = results["split"]
    assert split["train"][1] < split["validation"][0] <= split["validation"][1] < split["test"][0]
    for metrics in results["test"].values():
        assert all(math.isfinite(v) for v in metrics.values())


def test_selected_model_is_the_best_on_validation_not_test(trained):
    _, results = trained
    val = {name: results["validation"][name]["r2_log"] for name in candidate_models()}
    assert results["selected_model"] == max(val, key=val.get)


def test_run_writes_artifacts_and_reports(trained):
    out, _ = trained
    for path in ["artifacts/model.pkl", "artifacts/item_state.csv", "reports/metrics.json", "reports/results.md"]:
        assert (out / path).exists(), path
    assert "selected_model" in json.loads((out / "reports" / "metrics.json").read_text())


def test_predict_pipeline_end_to_end(trained):
    out, _ = trained
    pipeline = PredictPipeline(str(out / "artifacts" / "model.pkl"), str(out / "artifacts" / "item_state.csv"))
    code = pipeline.items()["ITEM CODE"].iloc[0]
    last = int(pipeline.history(code)["LAST_PERIOD"])
    prediction = pipeline.predict(code, year=(last + 1) // 12, month=(last + 1) % 12 + 1)
    assert np.isfinite(prediction) and prediction >= 0
    with pytest.raises(KeyError):
        pipeline.predict("not-a-real-item", 2030, 1)


def test_streamlit_app_runs_and_forecasts(trained, monkeypatch):
    """Drives the real app.py headlessly: load, pick the defaults, click the button."""
    from streamlit.testing.v1 import AppTest

    out, _ = trained
    app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
    monkeypatch.chdir(out)  # the app looks for ./artifacts
    app = AppTest.from_file(app_path, default_timeout=60).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert "Forecast:" in app.success[0].value
