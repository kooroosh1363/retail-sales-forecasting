from pathlib import Path
import json
import pandas as pd

from src.forecast import main


def test_forecasting_pipeline_end_to_end():
    main()
    root = Path(__file__).resolve().parents[1]
    metrics = json.loads((root / "artifacts" / "metrics.json").read_text())

    assert metrics["validation_design"]["backtest_folds"] == 4
    assert metrics["validation_design"]["final_test_horizon_days"] == 28
    assert metrics["selection"]["selected_model"] in {"seasonal_naive", "ridge", "random_forest"}
    assert metrics["test_metrics"]["mae"] > 0
    assert metrics["test_metrics"]["rmse"] >= metrics["test_metrics"]["mae"]
    assert 0 <= metrics["test_metrics"]["wape"] < 2.0
    assert metrics["uncertainty"]["absolute_error_q90"] > 0

    series_audit = metrics["series_audit"]
    assert series_audit["terminal_day_excluded_as_potentially_incomplete"] is True
    assert series_audit["excluded_terminal_date"] == "2011-12-09"
    assert series_audit["series_end"] == "2011-12-08"
    assert series_audit["source_last_timestamp"] == "2011-12-09T12:50:00"

    expected = [
        "daily_sales.csv",
        "backtest_metrics.csv",
        "backtest_summary.csv",
        "backtest_predictions.csv",
        "test_forecast.csv",
        "horizon_errors.csv",
        "feature_importance.csv",
        "metrics.json",
    ]
    for name in expected:
        assert (root / "artifacts" / name).exists()

    test = pd.read_csv(root / "artifacts" / "test_forecast.csv")
    assert len(test) == 28
    assert test["date"].max() == "2011-12-08"
    assert (test["forecast"] >= 0).all()
    assert (test["lower_90_heuristic"] >= 0).all()
    assert (test["upper_90_heuristic"] >= test["forecast"]).all()
