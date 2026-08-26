# Data Dictionary

## Raw source fields

- `InvoiceNo`: invoice identifier; cancellations begin with `C`.
- `StockCode`: product identifier.
- `Description`: product text description.
- `Quantity`: units on the invoice line.
- `InvoiceDate`: transaction timestamp.
- `UnitPrice`: unit price.
- `CustomerID`: customer identifier when available.
- `Country`: customer country.

## Derived transaction field

- `line_revenue`: `Quantity * UnitPrice` after cleaning.

## Daily target table

- `date`: calendar date.
- `sales`: total valid positive `line_revenue` for that date; missing calendar dates are filled as zero-sales days.

## Forecast features

Calendar features:
- `day_of_week`
- `day_of_month`
- `month`
- `week_of_year`
- `is_weekend`

Lag features:
- `lag_1`
- `lag_7`
- `lag_14`
- `lag_28`

Rolling features, all computed after shifting sales by one day:
- `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_28`
- `rolling_std_7`, `rolling_std_14`, `rolling_std_28`

The one-day shift is critical: the target day itself is never included in its rolling statistics.
