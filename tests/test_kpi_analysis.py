"""
Tests for analysis/kpi_analysis.py

All tests use synthetic DataFrames — no database connection required.
"""
import pandas as pd
import pytest
from datetime import date, timedelta

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analysis.kpi_analysis import (
    calculate_kpis,
    get_top_sellers,
    get_worst_performers,
    calculate_day_over_day_change,
    calculate_week_over_week_change,
    get_kpi_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n_products=3, days=14):
    """
    Build a simple multi-product DataFrame with predictable values
    so we can assert exact results.
    """
    records = []
    latest = date.today()
    products = [
        {"name": "Product A", "category": "Electronics", "revenue": 300, "ad_spend": 60,  "clicks": 200, "impressions": 5000, "units_sold": 20},
        {"name": "Product B", "category": "Audio",       "revenue": 150, "ad_spend": 100, "clicks": 100, "impressions": 4000, "units_sold": 8},
        {"name": "Product C", "category": "Office",      "revenue": 500, "ad_spend": 50,  "clicks": 400, "impressions": 8000, "units_sold": 40},
    ]
    for i in range(days):
        day = latest - timedelta(days=days - 1 - i)
        for p in products[:n_products]:
            records.append({"date": pd.Timestamp(day), **p})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# calculate_kpis
# ---------------------------------------------------------------------------

class TestCalculateKPIs:
    def test_adds_all_metric_columns(self):
        df = _make_df()
        result = calculate_kpis(df)
        for col in ["CTR", "conversion_rate", "ROAS", "ACOS", "CPC"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_ctr_calculation(self):
        df = _make_df(n_products=1)
        result = calculate_kpis(df)
        # Product A: clicks=200, impressions=5000 → CTR=0.04
        assert abs(result["CTR"].iloc[0] - 0.04) < 1e-6

    def test_roas_calculation(self):
        df = _make_df(n_products=1)
        result = calculate_kpis(df)
        # Product A: revenue=300, ad_spend=60 → ROAS=5.0
        assert abs(result["ROAS"].iloc[0] - 5.0) < 1e-6

    def test_acos_calculation(self):
        df = _make_df(n_products=1)
        result = calculate_kpis(df)
        # Product A: ad_spend=60, revenue=300 → ACOS=0.2
        assert abs(result["ACOS"].iloc[0] - 0.2) < 1e-6

    def test_cpc_calculation(self):
        df = _make_df(n_products=1)
        result = calculate_kpis(df)
        # Product A: ad_spend=60, clicks=200 → CPC=0.3
        assert abs(result["CPC"].iloc[0] - 0.3) < 1e-6

    def test_no_division_by_zero_on_zero_clicks(self):
        df = _make_df(n_products=1)
        df["clicks"] = 0
        result = calculate_kpis(df)
        assert result["conversion_rate"].notna().all()
        assert result["CPC"].notna().all()

    def test_does_not_modify_original_df(self):
        df = _make_df()
        original_cols = list(df.columns)
        calculate_kpis(df)
        assert list(df.columns) == original_cols, "calculate_kpis should not mutate input"


# ---------------------------------------------------------------------------
# get_top_sellers
# ---------------------------------------------------------------------------

class TestGetTopSellers:
    def test_returns_correct_top_product(self):
        df = calculate_kpis(_make_df())
        top = get_top_sellers(df, top_n=1)
        # Product C has revenue=500/day → highest total
        assert top.index[0] == "Product C"

    def test_respects_top_n(self):
        df = calculate_kpis(_make_df())
        top = get_top_sellers(df, top_n=2)
        assert len(top) == 2

    def test_sorted_descending(self):
        df = calculate_kpis(_make_df())
        top = get_top_sellers(df, top_n=3)
        assert list(top.values) == sorted(top.values, reverse=True)


# ---------------------------------------------------------------------------
# get_worst_performers
# ---------------------------------------------------------------------------

class TestGetWorstPerformers:
    def test_returns_lowest_roas_first(self):
        df = calculate_kpis(_make_df())
        worst = get_worst_performers(df, top_n=1)
        # Product B: ROAS = 150/100 = 1.5 → worst
        assert worst.index[0] == "Product B"

    def test_respects_top_n(self):
        df = calculate_kpis(_make_df())
        worst = get_worst_performers(df, top_n=2)
        assert len(worst) == 2

    def test_sorted_ascending(self):
        df = calculate_kpis(_make_df())
        worst = get_worst_performers(df, top_n=3)
        assert list(worst.values) == sorted(worst.values)


# ---------------------------------------------------------------------------
# calculate_day_over_day_change
# ---------------------------------------------------------------------------

class TestDayOverDayChange:
    def test_returns_none_with_single_day(self):
        df = _make_df(days=1)
        result = calculate_day_over_day_change(df)
        assert result is None

    def test_positive_change_when_revenue_increases(self):
        latest = date.today()
        df = pd.DataFrame([
            {"date": pd.Timestamp(latest - timedelta(days=1)), "name": "P", "category": "X", "revenue": 100, "ad_spend": 10, "clicks": 50, "impressions": 1000, "units_sold": 5},
            {"date": pd.Timestamp(latest),                     "name": "P", "category": "X", "revenue": 200, "ad_spend": 10, "clicks": 50, "impressions": 1000, "units_sold": 10},
        ])
        result = calculate_day_over_day_change(df)
        assert result["P"] == pytest.approx(100.0)  # +100%

    def test_uses_sorted_dates(self):
        # Dates inserted in reverse order — function should still pick last two correctly
        latest = date.today()
        df = pd.DataFrame([
            {"date": pd.Timestamp(latest),                     "name": "P", "category": "X", "revenue": 200, "ad_spend": 10, "clicks": 50, "impressions": 1000, "units_sold": 10},
            {"date": pd.Timestamp(latest - timedelta(days=1)), "name": "P", "category": "X", "revenue": 100, "ad_spend": 10, "clicks": 50, "impressions": 1000, "units_sold": 5},
        ])
        result = calculate_day_over_day_change(df)
        assert result["P"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# calculate_week_over_week_change
# ---------------------------------------------------------------------------

class TestWeekOverWeekChange:
    def test_returns_dataframe(self):
        df = _make_df(days=20)
        result = calculate_week_over_week_change(df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        df = _make_df(days=20)
        result = calculate_week_over_week_change(df)
        for col in ["name", "this_week", "last_week", "change_pct"]:
            assert col in result.columns

    def test_positive_change_when_recent_week_higher(self):
        latest = date.today()
        records = []
        # Last 7 days: revenue=200/day
        for i in range(7):
            records.append({"date": pd.Timestamp(latest - timedelta(days=i)), "name": "P", "category": "X", "revenue": 200, "ad_spend": 20, "clicks": 100, "impressions": 2000, "units_sold": 10})
        # Previous 7 days: revenue=100/day
        for i in range(7, 14):
            records.append({"date": pd.Timestamp(latest - timedelta(days=i)), "name": "P", "category": "X", "revenue": 100, "ad_spend": 20, "clicks": 100, "impressions": 2000, "units_sold": 5})
        df = pd.DataFrame(records)
        result = calculate_week_over_week_change(df)
        assert result.loc[result["name"] == "P", "change_pct"].values[0] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# get_kpi_summary
# ---------------------------------------------------------------------------

class TestGetKPISummary:
    def test_returns_dict(self):
        df = _make_df()
        result = get_kpi_summary(df)
        assert isinstance(result, dict)

    def test_has_all_keys(self):
        df = _make_df()
        result = get_kpi_summary(df)
        expected_keys = {
            "total_revenue", "total_ad_spend", "total_clicks",
            "total_impressions", "total_units_sold", "avg_roas",
            "avg_acos", "avg_ctr", "avg_cpc",
            "avg_conversion_rate", "revenue_per_unit",
        }
        assert expected_keys.issubset(result.keys())

    def test_total_revenue_is_correct(self):
        df = _make_df(n_products=1, days=1)
        # Product A: revenue=300 for 1 day
        result = get_kpi_summary(df)
        assert result["total_revenue"] == pytest.approx(300.0)

    def test_avg_roas_uses_totals_not_mean_of_means(self):
        # Correct ROAS = sum(revenue) / sum(ad_spend), not mean of per-row ROAS
        df = _make_df(n_products=1, days=1)
        result = get_kpi_summary(df)
        # Product A: revenue=300, ad_spend=60 → ROAS=5.0
        assert result["avg_roas"] == pytest.approx(5.0)

    def test_no_division_by_zero_on_empty_ish_data(self):
        df = _make_df(n_products=1, days=1)
        df["clicks"] = 0
        df["ad_spend"] = 0
        df["revenue"] = 0
        result = get_kpi_summary(df)
        # Should not raise, all values should be 0 or safe defaults
        assert result["avg_roas"] == 0.0
        assert result["avg_ctr"] == 0.0
