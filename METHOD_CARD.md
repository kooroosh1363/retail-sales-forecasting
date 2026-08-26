# Forecasting Method Card

## Intended use

Educational/portfolio demonstration of leakage-safe retail time-series forecasting using historical daily revenue.

## Unit and target

- unit: calendar day
- target: aggregate daily positive transaction revenue
- horizon: multi-step daily forecasting

## Temporal validation design

Random train/test splitting is not used. The final 28 calendar days are reserved as an untouched model-selection holdout. Before that holdout, four expanding-history backtest folds are evaluated, each with a 14-day recursive forecasting horizon.

The model-selection metric is mean backtest WAPE, with mean MAE as a secondary tie-breaker.

## Features

Calendar features are known for future dates. Lag and rolling features are generated only from historical sales available before the forecasted day. Rolling statistics use a one-day shift to prevent target leakage.

During multi-step inference, model predictions are recursively appended to history, so later forecast horizons cannot accidentally use actual future target values.

## Candidate methods

- 7-day seasonal naive baseline
- Ridge regression with standardized features
- Random Forest regression

The baseline participates in model selection rather than being reported only as decoration.

## Metrics

- MAE: interpretable average absolute revenue error
- RMSE: emphasizes large misses
- WAPE: total absolute error divided by total absolute actual revenue, avoiding the zero-denominator problems of day-level MAPE

## Forecast uncertainty

A symmetric heuristic band is generated from the 90th percentile of selected-model absolute errors observed during backtesting. This is a diagnostic uncertainty approximation and must not be described as a calibrated 90% prediction interval.

## Diagnostics

The project emits backtest metrics, fold-level predictions, final holdout predictions, error by forecast horizon, and feature importance for learned models.

## Limitations

- only one historical retailer and roughly one year of data;
- no external or temporal validation beyond the historical window;
- no promotions, stock availability, marketing, weather, holidays, macroeconomics, or competitor information;
- cancellation/return behavior is excluded from the positive-sales target rather than forecast separately;
- recursive forecasting can accumulate error across longer horizons;
- uncertainty bands are heuristic, not calibrated probabilistic intervals;
- feature importance is predictive rather than causal.

## Production extensions

A production system would add stronger calendar/event features, exogenous drivers, calibrated probabilistic forecasts, rolling retraining, drift monitoring, hierarchical/store-level forecasts when identifiers exist, business-cost-aware metrics, and monitored forecast-vs-actual dashboards.
