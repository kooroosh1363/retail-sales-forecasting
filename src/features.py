from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]


def make_supervised(series: pd.DataFrame) -> pd.DataFrame:
    frame = series.copy().sort_values("date").reset_index(drop=True)
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["day_of_month"] = frame["date"].dt.day
    frame["month"] = frame["date"].dt.month
    frame["week_of_year"] = frame["date"].dt.isocalendar().week.astype(int)
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)

    for lag in LAGS:
        frame[f"lag_{lag}"] = frame["sales"].shift(lag)
    shifted = frame["sales"].shift(1)
    for window in ROLLING_WINDOWS:
        frame[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        frame[f"rolling_std_{window}"] = shifted.rolling(window).std(ddof=0)

    feature_cols = [
        "day_of_week", "day_of_month", "month", "week_of_year", "is_weekend",
        *[f"lag_{lag}" for lag in LAGS],
        *[f"rolling_mean_{window}" for window in ROLLING_WINDOWS],
        *[f"rolling_std_{window}" for window in ROLLING_WINDOWS],
    ]
    frame = frame.dropna(subset=feature_cols).reset_index(drop=True)
    if not np.isfinite(frame[feature_cols].to_numpy(dtype=float)).all():
        raise ValueError("Features must be finite")
    return frame


def feature_columns() -> list[str]:
    return [
        "day_of_week", "day_of_month", "month", "week_of_year", "is_weekend",
        *[f"lag_{lag}" for lag in LAGS],
        *[f"rolling_mean_{window}" for window in ROLLING_WINDOWS],
        *[f"rolling_std_{window}" for window in ROLLING_WINDOWS],
    ]


def one_step_features(target_date: pd.Timestamp, history: list[float]) -> dict[str, float]:
    if len(history) < max(max(LAGS), max(ROLLING_WINDOWS)):
        raise ValueError("Insufficient history for recursive forecasting")
    row: dict[str, float] = {
        "day_of_week": float(target_date.dayofweek),
        "day_of_month": float(target_date.day),
        "month": float(target_date.month),
        "week_of_year": float(target_date.isocalendar().week),
        "is_weekend": float(target_date.dayofweek >= 5),
    }
    for lag in LAGS:
        row[f"lag_{lag}"] = float(history[-lag])
    for window in ROLLING_WINDOWS:
        values = np.asarray(history[-window:], dtype=float)
        row[f"rolling_mean_{window}"] = float(values.mean())
        row[f"rolling_std_{window}"] = float(values.std(ddof=0))
    return row
