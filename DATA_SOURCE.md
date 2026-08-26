# Data Source

## Canonical source

UCI Machine Learning Repository — Online Retail dataset.

- Dataset: Online Retail
- Source: https://archive.ics.uci.edu/dataset/352/online+retail
- Raw rows expected by the pipeline: 541,909
- Coverage: December 2010 through December 2011
- Unit of raw data: invoice line

The pipeline downloads the official UCI ZIP directly and extracts the single XLSX file inside it.

## Forecast target construction

This project forecasts aggregate daily gross sales revenue derived from valid positive transaction lines:

`line_revenue = Quantity * UnitPrice`

Daily sales are then created by calendar-day aggregation. Missing calendar dates inside the observed date range are explicitly reindexed and represented as zero-sales days rather than silently dropped.

## Cleaning policy

- cancelled invoices whose `InvoiceNo` starts with `C` are excluded;
- rows with non-positive quantity or unit price are excluded;
- exact full-row duplicates are removed;
- customer ID is not required because the forecasting unit is calendar day, not customer.

## Claim boundaries

The source is one historical retailer over roughly one year. Revenue is derived from listed transaction values in the dataset and is not a guarantee of realized future demand. The dataset has no explicit promotion calendar, inventory availability, marketing spend, competitor pricing, macroeconomic variables, or future exogenous signals. Forecasting results therefore demonstrate methodology rather than a production retail planning guarantee.
