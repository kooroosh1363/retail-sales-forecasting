from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import io

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_XLSX = RAW_DIR / "Online Retail.xlsx"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
EXPECTED_RAW_ROWS = 541_909
EXPECTED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]


def download_raw() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_XLSX.exists():
        return RAW_XLSX
    response = requests.get(UCI_ZIP_URL, timeout=120)
    response.raise_for_status()
    with ZipFile(io.BytesIO(response.content)) as zf:
        matches = [name for name in zf.namelist() if name.lower().endswith(".xlsx")]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one XLSX in UCI archive, found: {matches}")
        RAW_XLSX.write_bytes(zf.read(matches[0]))
    return RAW_XLSX


def load_raw() -> pd.DataFrame:
    path = download_raw()
    df = pd.read_excel(path, engine="openpyxl")
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected schema: {list(df.columns)}")
    if len(df) != EXPECTED_RAW_ROWS:
        raise ValueError(f"Unexpected raw row count: {len(df)}")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="raise")
    return df


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    work = df.copy()
    audit = {"raw_rows": int(len(work))}

    work["InvoiceNo"] = work["InvoiceNo"].astype(str)
    cancel_mask = work["InvoiceNo"].str.startswith("C", na=False)
    audit["cancelled_rows_removed"] = int(cancel_mask.sum())
    work = work.loc[~cancel_mask].copy()

    positive_mask = (work["Quantity"] > 0) & (work["UnitPrice"] > 0)
    audit["non_positive_rows_removed"] = int((~positive_mask).sum())
    work = work.loc[positive_mask].copy()

    before_dedup = len(work)
    work = work.drop_duplicates().copy()
    audit["exact_duplicates_removed"] = int(before_dedup - len(work))

    work["line_revenue"] = work["Quantity"].astype(float) * work["UnitPrice"].astype(float)
    if not np.isfinite(work["line_revenue"]).all() or (work["line_revenue"] <= 0).any():
        raise ValueError("Cleaned line revenue must be finite and positive")

    audit["clean_rows"] = int(len(work))
    audit["first_timestamp"] = work["InvoiceDate"].min().isoformat()
    audit["last_timestamp"] = work["InvoiceDate"].max().isoformat()
    return work.reset_index(drop=True), audit


def build_daily_sales(clean: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    daily = (
        clean.assign(date=clean["InvoiceDate"].dt.normalize())
        .groupby("date", as_index=True)["line_revenue"]
        .sum()
        .sort_index()
    )
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    missing_dates = full_index.difference(daily.index)
    out = daily.reindex(full_index, fill_value=0.0).rename("sales").to_frame()
    out.index.name = "date"
    out = out.reset_index()

    if out["date"].duplicated().any():
        raise ValueError("Daily series must have one row per date")
    if (out["sales"] < 0).any() or not np.isfinite(out["sales"]).all():
        raise ValueError("Daily sales must be finite and non-negative")

    audit = {
        "calendar_days": int(len(out)),
        "observed_sales_days": int((out["sales"] > 0).sum()),
        "zero_sales_days": int((out["sales"] == 0).sum()),
        "calendar_gaps_filled": int(len(missing_dates)),
        "series_start": out["date"].min().date().isoformat(),
        "series_end": out["date"].max().date().isoformat(),
    }
    return out, audit
