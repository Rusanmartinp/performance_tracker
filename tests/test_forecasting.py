"""
Tests for analysis/forecasting.py

These tests mock the database connection so no PostgreSQL instance is needed.
We test the forecast output shape, column names, and edge cases.
"""
import pandas as pd
import pytest
import numpy as np
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_revenue_series(days=90, base=10000.0, noise=0.08, seed=42):
    """
    Generate a realistic synthetic revenue series with weekly seasonality
    and mild trend — suitable for ARIMA fitting in tests.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=date.today(), periods=days, freq="D")
    trend = np.linspace(1.0, 1.15, days)
    weekly = np.array([1.0, 0.9, 0.95, 1.0, 1.05, 1.25, 1.20] * (days // 7 + 1))[:days]
    noise_arr = rng.normal(1.0, noise, days)
    revenue = base * trend * weekly * noise_arr
    return pd.DataFrame({"date": dates, "revenue": revenue})


def _patch_db(mock_df):
    """
    Returns a context manager that patches pd.read_sql to return mock_df.
    Also patches create_engine to avoid real DB connections.
    """
    return patch("analysis.forecasting.pd.read_sql", return_value=mock_df)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestForecastOutputShape:
    def test_returns_dataframe(self):
        mock_df = _make_revenue_series()
        with patch("analysis.forecasting.create_engine"), _patch_db(mock_df):
            from analysis.forecasting import forecast_revenue
            result = forecast_revenue(days=7)
        assert isinstance(result, pd.DataFrame)

    def test_forecast_has_required_columns(self):
        mock_df = _make_revenue_series()
        with patch("analysis.forecasting.create_engine"), _patch_db(mock_df):
            from analysis.forecasting import forecast_revenue
            result = forecast_revenue(days=7)
        for col in ["ds", "yhat", "yhat_lower", "yhat_upper"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_forecast_period_length(self):
        days = 14
        mock_df = _make_revenue_series(days=90)
        with patch("analysis.forecasting.create_engine"), _patch_db(mock_df):
            from analysis.forecasting import forecast_revenue
            result = forecast_revenue(days=days)
        # Result includes history + forecast rows
        future = result[result["ds"] > pd.Timestamp("today").normalize()]
        assert len(future) == days, \
            f"Expected {days} future rows, got {len(future)}"

    def test_confidence_interval_ordering(self):
        """yhat_lower <= yhat <= yhat_upper for all forecast rows."""
        mock_df = _make_revenue_series()
        with patch("analysis.forecasting.create_engine"), _patch_db(mock_df):
            from analysis.forecasting import forecast_revenue
            result = forecast_revenue(days=10)
        future = result[result["ds"] > pd.Timestamp("today").normalize()].dropna()
        assert (future["yhat_lower"] <= future["yhat"]).all(), \
            "yhat_lower should be <= yhat"
        assert (future["yhat"] <= future["yhat_upper"]).all(), \
            "yhat should be <= yhat_upper"


class TestForecastEdgeCases:
    def test_raises_on_insufficient_data(self):
        # Only 5 rows — below the 10-row minimum
        mock_df = _make_revenue_series(days=5)
        with patch("analysis.forecasting.create_engine"), _patch_db(mock_df):
            from analysis.forecasting import forecast_revenue
            with pytest.raises(ValueError, match="Not enough data"):
                forecast_revenue(days=30)

    def test_history_rows_have_null_confidence_interval(self):
        """Historical rows should not have confidence interval values."""
        mock_df = _make_revenue_series()
        with patch("analysis.forecasting.create_engine"), _patch_db(mock_df):
            from analysis.forecasting import forecast_revenue
            result = forecast_revenue(days=10)
        history = result[result["ds"] <= pd.Timestamp("today").normalize()]
        assert history["yhat_lower"].isna().all(), \
            "Historical rows should have NaN for yhat_lower"


class TestForecastFilters:
    def test_product_filter_passed_through(self):
        """
        When a product_filter is set, the function should filter the DataFrame
        before training. We verify by checking that read_sql is called once
        and the result still has the right shape.
        """
        mock_df = _make_revenue_series(days=60)
        # Simulate the SQL returning rows for a single product already
        mock_df["name"] = "Wireless Mouse"
        mock_df["category"] = "Accessories"
        mock_df.rename(columns={"date": "date", "revenue": "revenue"}, inplace=True)

        with patch("analysis.forecasting.create_engine"), \
             patch("analysis.forecasting.pd.read_sql", return_value=mock_df) as mock_sql:
            from analysis.forecasting import forecast_revenue
            result = forecast_revenue(days=7, product_filter="Wireless Mouse")

        mock_sql.assert_called_once()
        assert isinstance(result, pd.DataFrame)
