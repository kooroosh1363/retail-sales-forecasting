from src.data import build_daily_sales, clean_transactions, load_raw
from src.features import feature_columns, make_supervised


def test_data_contract_and_daily_series():
    raw = load_raw()
    assert len(raw) == 541_909
    clean, audit = clean_transactions(raw)
    assert len(clean) == audit["clean_rows"]
    assert (clean["line_revenue"] > 0).all()

    daily, series_audit = build_daily_sales(clean)
    assert daily["date"].is_monotonic_increasing
    assert daily["date"].nunique() == len(daily)
    assert (daily["sales"] >= 0).all()
    assert series_audit["calendar_days"] == len(daily)

    supervised = make_supervised(daily)
    assert supervised[feature_columns()].notna().all().all()
    assert supervised["date"].is_monotonic_increasing
