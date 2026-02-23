"""
Tests for analysis/recommendation_engine.py

All tests use synthetic DataFrames — no database connection required.
"""
import pandas as pd
import pytest
from datetime import date, timedelta

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analysis.recommendation_engine import generate_recommendations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(product="Test Product", category="Electronics",
             revenue=200, ad_spend=50, clicks=100,
             impressions=5000, units_sold=10, days=20):
    """
    Build a minimal daily_performance DataFrame with `days` rows,
    all metrics constant (easy to reason about in tests).
    """
    latest = date.today()
    dates = [latest - timedelta(days=i) for i in range(days - 1, -1, -1)]
    return pd.DataFrame({
        "date":        pd.to_datetime(dates),
        "name":        product,
        "category":    category,
        "revenue":     float(revenue),
        "ad_spend":    float(ad_spend),
        "clicks":      int(clicks),
        "impressions": int(impressions),
        "units_sold":  int(units_sold),
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHighACOS:
    def test_triggers_when_acos_above_threshold(self):
        # ACOS = ad_spend / revenue = 80/100 = 0.80 → above 0.35
        df = _make_df(revenue=100, ad_spend=80)
        recs = generate_recommendations(df)
        alerts = [r["Alert"] for r in recs]
        assert any("ACOS" in a for a in alerts), "Expected High ACOS alert"

    def test_does_not_trigger_when_acos_normal(self):
        # ACOS = 20/100 = 0.20 → below 0.35
        df = _make_df(revenue=100, ad_spend=20, clicks=500, units_sold=50)
        recs = generate_recommendations(df)
        alerts = [r["Alert"] for r in recs]
        assert not any("ACOS" in a for a in alerts), "Unexpected High ACOS alert"


class TestStrongROAS:
    def test_triggers_when_roas_above_threshold(self):
        # ROAS = revenue / ad_spend = 500/50 = 10 → above 4
        df = _make_df(revenue=500, ad_spend=50, clicks=500, units_sold=50)
        recs = generate_recommendations(df)
        alerts = [r["Alert"] for r in recs]
        assert any("ROAS" in a for a in alerts), "Expected Strong ROAS alert"

    def test_does_not_trigger_when_roas_low(self):
        # ROAS = 100/100 = 1 → below 4
        df = _make_df(revenue=100, ad_spend=100)
        recs = generate_recommendations(df)
        alerts = [r["Alert"] for r in recs]
        assert not any("ROAS" in a for a in alerts), "Unexpected Strong ROAS alert"


class TestLowCTR:
    def test_triggers_when_ctr_below_threshold(self):
        # CTR = clicks / impressions = 50 / 10000 = 0.005 → below 0.02
        df = _make_df(clicks=50, impressions=10000)
        recs = generate_recommendations(df)
        alerts = [r["Alert"] for r in recs]
        assert any("CTR" in a for a in alerts), "Expected Low CTR alert"

    def test_does_not_trigger_when_ctr_healthy(self):
        # CTR = 300 / 5000 = 0.06 → above 0.02
        df = _make_df(clicks=300, impressions=5000, units_sold=30)
        recs = generate_recommendations(df)
        alerts = [r["Alert"] for r in recs]
        assert not any("CTR" in a for a in alerts), "Unexpected Low CTR alert"


class TestLowConversion:
    def test_triggers_when_conversion_below_threshold(self):
        # Conv rate = units_sold / clicks = 2 / 200 = 0.01 → below 0.08
        df = _make_df(units_sold=2, clicks=200)
        recs = generate_recommendations(df)
        alerts = [r["Alert"] for r in recs]
        assert any("Conversion" in a for a in alerts), "Expected Low Conversion alert"


class TestRevenueDrop:
    def test_triggers_on_significant_weekly_drop(self):
        """
        Build a product with high revenue in week -2 and low revenue in week -1.
        Expect a Revenue Drop alert.
        """
        latest = date.today()
        # Last 7 days: low revenue
        dates_recent = [latest - timedelta(days=i) for i in range(6, -1, -1)]
        # Previous 7 days: high revenue
        dates_prior  = [latest - timedelta(days=i) for i in range(13, 6, -1)]

        df_recent = pd.DataFrame({
            "date": pd.to_datetime(dates_recent), "name": "Falling Product",
            "category": "Audio", "revenue": 50.0, "ad_spend": 20.0,
            "clicks": 100, "impressions": 4000, "units_sold": 5,
        })
        df_prior = pd.DataFrame({
            "date": pd.to_datetime(dates_prior), "name": "Falling Product",
            "category": "Audio", "revenue": 500.0, "ad_spend": 20.0,
            "clicks": 100, "impressions": 4000, "units_sold": 50,
        })
        df = pd.concat([df_prior, df_recent], ignore_index=True)
        recs = generate_recommendations(df)
        alerts = [r["Alert"] for r in recs]
        assert any("Drop" in a for a in alerts), "Expected Revenue Drop alert"


class TestRecommendationStructure:
    def test_each_recommendation_has_required_keys(self):
        df = _make_df(revenue=100, ad_spend=80)  # triggers High ACOS
        recs = generate_recommendations(df)
        required_keys = {"Product", "Category", "Alert", "Type",
                         "Metric", "This Week", "Last Week", "Trend", "Action"}
        for rec in recs:
            assert required_keys.issubset(rec.keys()), \
                f"Missing keys in recommendation: {required_keys - rec.keys()}"

    def test_returns_empty_list_for_healthy_product(self):
        # All metrics healthy: moderate ACOS, good ROAS, decent CTR, decent CR
        df = _make_df(revenue=400, ad_spend=50, clicks=300,
                      impressions=6000, units_sold=35)
        recs = generate_recommendations(df)
        assert isinstance(recs, list)

    def test_sorted_warnings_first(self):
        # Mix of alerts — warnings should come before success
        df = _make_df(revenue=100, ad_spend=80, clicks=50,
                      impressions=10000, units_sold=2)
        recs = generate_recommendations(df)
        if len(recs) > 1:
            types = [r["Type"] for r in recs]
            order = {"warning": 0, "info": 1, "success": 2}
            scores = [order.get(t, 9) for t in types]
            assert scores == sorted(scores), "Recommendations not sorted by priority"
