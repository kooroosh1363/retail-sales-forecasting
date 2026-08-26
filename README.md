# DS-05 — Retail Sales Forecasting

Portfolio-grade time-series forecasting project that converts historical retail transactions into a leakage-safe daily revenue forecasting workflow.

## What this project demonstrates

- reproducible acquisition from the official UCI Machine Learning Repository
- transaction cleaning and daily calendar aggregation
- explicit calendar-gap handling
- exclusion of the potentially incomplete terminal source day from evaluation
- leakage-safe lag and rolling features
- seasonal naive baseline
- Ridge and Random Forest challengers
- expanding-history walk-forward backtesting
- recursive multi-step forecasting
- model selection by backtest WAPE
- untouched final 28-complete-day holdout
- MAE, RMSE, WAPE and horizon-level error diagnostics
- heuristic uncertainty bands with explicit claim boundaries
- tests and GitHub Actions CI

## Data

The project uses the UCI **Online Retail** dataset with 541,909 raw invoice-line rows covering December 2010 through December 2011.

The source ends on **2011-12-09 at 12:50**. Because that terminal date is not observed through a full end-of-day boundary, the forecasting series excludes 2011-12-09 from model selection and evaluation rather than scoring a full-day forecast against a partial-day target. The final modeled calendar day is therefore **2011-12-08**.

Missing dates inside the modeled window are filled with zero observed positive sales so lag offsets and forecast horizons remain true calendar-day distances. The public source does not reveal whether those dates indicate closure, no sales, or missing capture, so zero-filled dates are a modeling convention rather than proof of zero demand.

See `DATA_SOURCE.md`, `DATA_DICTIONARY.md`, and `METHOD_CARD.md` for provenance, feature definitions, validation rules, and limitations.

## Architecture

```text
official UCI Online Retail ZIP
    -> schema validation
    -> remove cancellations / non-positive rows / exact duplicates
    -> line revenue
    -> aggregate daily sales
    -> exclude potentially incomplete terminal source day
    -> fill internal calendar gaps with zero observed positive sales
    -> lag + shifted rolling + calendar features
    -> expanding-history 4-fold backtest
    -> 14-day recursive horizon per fold
    -> seasonal naive / Ridge / Random Forest
    -> select by mean backtest WAPE
    -> lock model choice
    -> final untouched 28-day recursive holdout
    -> MAE / RMSE / WAPE
    -> horizon diagnostics + heuristic uncertainty band
    -> artifacts + pytest + GitHub Actions
```

## Leakage controls

Random train/test splitting is intentionally not used.

The final 28 complete calendar days are excluded from model selection. Candidate methods are compared only on earlier walk-forward folds. Lag features use only prior observations, rolling statistics are shifted by one day, and multi-step forecasts are recursive: later horizons consume earlier predictions rather than future actual sales.

## Candidate methods

- `seasonal_naive`: previous-week value recursively repeated as a 7-day seasonal baseline
- `ridge`: linear model with feature standardization
- `random_forest`: nonlinear tree ensemble

The simple seasonal baseline is allowed to win if the learned models do not outperform it.

## Features

Calendar features:
- day of week
- day of month
- month
- ISO week of year
- weekend flag

Lag features:
- 1, 7, 14 and 28 days

Rolling features:
- shifted 7-, 14- and 28-day mean
- shifted 7-, 14- and 28-day standard deviation

## Validation policy

- four backtest folds
- expanding history
- 14-day forecast horizon per fold
- recursive forecasting
- primary selection metric: mean WAPE
- secondary selection metric: mean MAE
- final test: last 28 complete calendar days, never used for candidate selection

WAPE is preferred over day-level MAPE because zero-sales days can make percentage error undefined or unstable.

## Uncertainty policy

The pipeline creates a symmetric diagnostic band from the 90th percentile of absolute backtest errors for the selected method. It is **not** a calibrated 90% prediction interval and is documented only as an uncertainty approximation.

## Generated artifacts

Running the pipeline writes ignored outputs to `artifacts/`:

- `metrics.json`
- `daily_sales.csv`
- `backtest_metrics.csv`
- `backtest_summary.csv`
- `backtest_predictions.csv`
- `test_forecast.csv`
- `horizon_errors.csv`
- `feature_importance.csv`
- `model.joblib` when a learned model is selected

## Claim boundary

This project demonstrates historical retail forecasting methodology on one public dataset. It does not guarantee future revenue, demand, inventory requirements, or business outcomes.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.forecast
```
