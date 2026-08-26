from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import build_daily_sales, clean_transactions, load_raw
from .features import feature_columns, make_supervised, one_step_features

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
RANDOM_STATE = 42
BACKTEST_FOLDS = 4
BACKTEST_HORIZON = 14
TEST_HORIZON = 28


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    abs_err = np.abs(err)
    denom = float(np.abs(y_true).sum())
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "wape": float(abs_err.sum() / denom) if denom > 0 else float("nan"),
    }


def make_models() -> dict[str, object]:
    return {
        "ridge": Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def recursive_model_forecast(model: object, history: list[float], dates: pd.DatetimeIndex) -> np.ndarray:
    hist = [float(x) for x in history]
    rows = []
    for date in dates:
        row = one_step_features(pd.Timestamp(date), hist)
        X = pd.DataFrame([row], columns=feature_columns())
        pred = max(0.0, float(model.predict(X)[0]))
        rows.append(pred)
        hist.append(pred)
    return np.asarray(rows, dtype=float)


def seasonal_naive_forecast(history: list[float], horizon: int, season: int = 7) -> np.ndarray:
    hist = [float(x) for x in history]
    preds = []
    for _ in range(horizon):
        pred = max(0.0, float(hist[-season]))
        preds.append(pred)
        hist.append(pred)
    return np.asarray(preds, dtype=float)


def fit_model(name: str, daily: pd.DataFrame, forecast_start: int) -> object:
    supervised = make_supervised(daily.iloc[:forecast_start].copy())
    X = supervised[feature_columns()].astype(float)
    y = supervised["sales"].astype(float)
    model = make_models()[name]
    model.fit(X, y)
    return model


def backtest(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(daily)
    test_start = n - TEST_HORIZON
    first_start = test_start - BACKTEST_FOLDS * BACKTEST_HORIZON
    if first_start <= 60:
        raise ValueError("Series is too short for configured backtest")

    metric_rows = []
    prediction_rows = []
    model_names = ["seasonal_naive", "ridge", "random_forest"]

    for fold in range(BACKTEST_FOLDS):
        start = first_start + fold * BACKTEST_HORIZON
        end = start + BACKTEST_HORIZON
        true = daily.iloc[start:end]["sales"].to_numpy(dtype=float)
        dates = pd.DatetimeIndex(daily.iloc[start:end]["date"])
        history = daily.iloc[:start]["sales"].tolist()

        for name in model_names:
            if name == "seasonal_naive":
                pred = seasonal_naive_forecast(history, len(true))
            else:
                model = fit_model(name, daily, start)
                pred = recursive_model_forecast(model, history, dates)
            score = metrics(true, pred)
            metric_rows.append({"fold": fold + 1, "model": name, **score})
            for horizon, (date, actual, forecast) in enumerate(zip(dates, true, pred), start=1):
                prediction_rows.append({
                    "fold": fold + 1,
                    "model": name,
                    "date": pd.Timestamp(date),
                    "horizon": horizon,
                    "actual": float(actual),
                    "forecast": float(forecast),
                    "absolute_error": float(abs(actual - forecast)),
                })

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def select_model(backtest_metrics: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    summary = (
        backtest_metrics.groupby("model")[["mae", "rmse", "wape"]]
        .mean()
        .sort_values(["wape", "mae"])
        .reset_index()
    )
    return str(summary.iloc[0]["model"]), summary


def feature_importance(model: object) -> pd.DataFrame:
    names = feature_columns()
    if isinstance(model, Pipeline):
        coef = np.abs(model.named_steps["model"].coef_)
        values = coef / coef.sum() if coef.sum() else coef
        method = "absolute standardized Ridge coefficient share"
    else:
        values = model.feature_importances_
        values = values / values.sum() if values.sum() else values
        method = "Random Forest impurity importance share"
    out = pd.DataFrame({"feature": names, "importance": values})
    out["method"] = method
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def main() -> None:
    ART.mkdir(exist_ok=True)
    raw = load_raw()
    clean, transaction_audit = clean_transactions(raw)
    daily, series_audit = build_daily_sales(clean)

    backtest_metrics, backtest_predictions = backtest(daily)
    selected_model, backtest_summary = select_model(backtest_metrics)

    test_start = len(daily) - TEST_HORIZON
    test = daily.iloc[test_start:].copy()
    history = daily.iloc[:test_start]["sales"].tolist()
    test_dates = pd.DatetimeIndex(test["date"])
    actual = test["sales"].to_numpy(dtype=float)

    if selected_model == "seasonal_naive":
        forecast = seasonal_naive_forecast(history, TEST_HORIZON)
        fitted = None
        importance = pd.DataFrame(columns=["feature", "importance", "method"])
    else:
        fitted = fit_model(selected_model, daily, test_start)
        forecast = recursive_model_forecast(fitted, history, test_dates)
        importance = feature_importance(fitted)
        joblib.dump(fitted, ART / "model.joblib")

    test_scores = metrics(actual, forecast)
    selected_bt = backtest_predictions.loc[backtest_predictions["model"] == selected_model]
    error_q90 = float(selected_bt["absolute_error"].quantile(0.90))
    lower = np.maximum(0.0, forecast - error_q90)
    upper = forecast + error_q90

    test_predictions = pd.DataFrame({
        "date": test_dates,
        "horizon": np.arange(1, TEST_HORIZON + 1),
        "actual": actual,
        "forecast": forecast,
        "lower_90_heuristic": lower,
        "upper_90_heuristic": upper,
    })
    test_predictions["absolute_error"] = np.abs(test_predictions["actual"] - test_predictions["forecast"])

    horizon_errors = test_predictions[["horizon", "absolute_error"]].copy()

    daily.to_csv(ART / "daily_sales.csv", index=False)
    backtest_metrics.to_csv(ART / "backtest_metrics.csv", index=False)
    backtest_summary.to_csv(ART / "backtest_summary.csv", index=False)
    backtest_predictions.to_csv(ART / "backtest_predictions.csv", index=False)
    test_predictions.to_csv(ART / "test_forecast.csv", index=False)
    horizon_errors.to_csv(ART / "horizon_errors.csv", index=False)
    importance.to_csv(ART / "feature_importance.csv", index=False)

    report = {
        "transaction_audit": transaction_audit,
        "series_audit": series_audit,
        "validation_design": {
            "backtest_folds": BACKTEST_FOLDS,
            "backtest_horizon_days": BACKTEST_HORIZON,
            "final_test_horizon_days": TEST_HORIZON,
            "split_policy": "expanding history with forward-only recursive multi-step forecasting",
            "test_policy": "final 28 calendar days excluded from model selection",
        },
        "feature_policy": {
            "lags": [1, 7, 14, 28],
            "rolling_windows": [7, 14, 28],
            "rolling_shift": "all rolling statistics use sales shifted by one day",
        },
        "selection": {
            "primary_metric": "mean backtest WAPE",
            "secondary_metric": "mean backtest MAE",
            "selected_model": selected_model,
            "backtest_summary": backtest_summary.to_dict(orient="records"),
        },
        "test_metrics": test_scores,
        "uncertainty": {
            "method": "symmetric heuristic band from 90th percentile of selected-model backtest absolute errors",
            "absolute_error_q90": error_q90,
            "claim_boundary": "diagnostic uncertainty approximation; not a calibrated 90% prediction interval",
        },
        "claim_boundary": "historical daily revenue forecasting methodology; no guarantee of future retail demand or revenue",
    }
    (ART / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
