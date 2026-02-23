"""
Tests for the anomaly detection logic in dashboard/app.py

The detect_anomalies function is inlined in app.py, so we redefine it here
to test it in isolation — no Streamlit or DB needed.
"""
import pandas as pd
import pytest
from datetime import date, timedelta

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Replicate the function under test (from dashboard/app.py)
# ---------------------------------------------------------------------------

def detect_anomalies(df, z_threshold=1.8, window_days=7):
    anomalies = []
    latest_date = df["date"].max()
    cutoff = latest_date - pd.Timedelta(days=window_days - 1)

    for product, df_product in df.groupby("name"):
        df_product = df_product.sort_values("date")
        if len(df_product) < 14:
            continue
        mean = df_product["revenue"].mean()
        std  = df_product["revenue"].std()
        if std == 0:
            continue
        recent = df_product[df_product["date"] >= cutoff]
        for _, row in recent.iterrows():
            z = (row["revenue"] - mean) / std
            if abs(z) > z_threshold:
                anomalies.append({
                    "Product":  product,
                    "Category": row["category"],
                    "Date":     row["date"].strftime("%b %d"),
                    "Revenue":  f"${row['revenue']:,.0f}",
                    "Expected": f"${mean:,.0f} ± ${std:,.0f}",
                    "Z-Score":  round(z, 2),
                    "Type":     "📈 Spike" if z > 0 else "📉 Drop",
                })

    return sorted(anomalies, key=lambda x: abs(x["Z-Score"]), reverse=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_df(product="Stable Product", days=30, revenue=200.0):
    """All days have the same revenue — no anomalies expected."""
    latest = date.today()
    dates = [latest - timedelta(days=i) for i in range(days - 1, -1, -1)]
    return pd.DataFrame({
        "date":     pd.to_datetime(dates),
        "name":     product,
        "category": "Electronics",
        "revenue":  revenue,
    })


def _df_with_spike(product="Spike Product", base=200.0, spike_value=2000.0, days=30):
    """Last day is a massive spike."""
    df = _stable_df(product=product, days=days, revenue=base)
    df.loc[df.index[-1], "revenue"] = spike_value
    return df


def _df_with_drop(product="Drop Product", base=200.0, drop_value=5.0, days=30):
    """Last day is a severe drop."""
    df = _stable_df(product=product, days=days, revenue=base)
    df.loc[df.index[-1], "revenue"] = drop_value
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoAnomalies:
    def test_stable_series_returns_empty(self):
        df = _stable_df()
        result = detect_anomalies(df)
        assert result == [], "Expected no anomalies for a stable series"

    def test_insufficient_history_skipped(self):
        # Only 10 days — below the 14-day minimum
        df = _stable_df(days=10)
        result = detect_anomalies(df)
        assert result == [], "Products with < 14 days of history should be skipped"


class TestSpikeDetection:
    def test_detects_revenue_spike(self):
        df = _df_with_spike(spike_value=5000.0)
        result = detect_anomalies(df)
        assert len(result) >= 1, "Expected at least one anomaly for a spike"
        assert result[0]["Type"] == "📈 Spike"

    def test_spike_has_positive_z_score(self):
        df = _df_with_spike(spike_value=5000.0)
        result = detect_anomalies(df)
        assert result[0]["Z-Score"] > 0


class TestDropDetection:
    def test_detects_revenue_drop(self):
        df = _df_with_drop(drop_value=1.0)
        result = detect_anomalies(df)
        assert len(result) >= 1, "Expected at least one anomaly for a drop"
        assert result[0]["Type"] == "📉 Drop"

    def test_drop_has_negative_z_score(self):
        df = _df_with_drop(drop_value=1.0)
        result = detect_anomalies(df)
        assert result[0]["Z-Score"] < 0


class TestSortingAndStructure:
    def test_sorted_by_absolute_z_score_descending(self):
        df_spike = _df_with_spike("Product A", spike_value=9000.0)
        df_drop  = _df_with_drop("Product B", drop_value=1.0)
        df = pd.concat([df_spike, df_drop], ignore_index=True)
        result = detect_anomalies(df)
        if len(result) > 1:
            z_scores = [abs(r["Z-Score"]) for r in result]
            assert z_scores == sorted(z_scores, reverse=True), \
                "Anomalies should be sorted by |Z-Score| descending"

    def test_result_has_required_keys(self):
        df = _df_with_spike()
        result = detect_anomalies(df)
        required = {"Product", "Category", "Date", "Revenue", "Expected", "Z-Score", "Type"}
        for item in result:
            assert required.issubset(item.keys())

    def test_multiple_products(self):
        df_a = _df_with_spike("Product A")
        df_b = _stable_df("Product B")
        df = pd.concat([df_a, df_b], ignore_index=True)
        result = detect_anomalies(df)
        products_flagged = {r["Product"] for r in result}
        assert "Product A" in products_flagged
        assert "Product B" not in products_flagged
