import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.recommendation_engine import get_recommendations
from analysis.kpi_analysis import (
    calculate_kpis,
    get_kpi_summary,
    get_top_sellers,
    get_worst_performers,
    calculate_week_over_week_change,
    calculate_day_over_day_change,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
load_dotenv()

st.set_page_config(page_title="Performance Tracker", layout="wide")
st.title("📦 E-commerce Performance Tracker")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    DB_URL = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
    engine = create_engine(DB_URL)
    query = """
        SELECT dp.date, p.name, p.category,
               dp.impressions, dp.clicks, dp.ad_spend,
               dp.units_sold, dp.revenue
        FROM daily_performance dp
        JOIN products p ON dp.product_id = p.id
        ORDER BY dp.date;
    """
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])
    return df

df_full = load_data()

# ── SIDEBAR FILTERS ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")
selected_product  = st.sidebar.selectbox("Product",  ["All"] + sorted(df_full["name"].unique().tolist()))
category_options  = df_full["category"].unique() if selected_product == "All" else df_full[df_full["name"] == selected_product]["category"].unique()
selected_category = st.sidebar.selectbox("Category", ["All"] + sorted(category_options.tolist()))

df = df_full.copy()
if selected_product  != "All": df = df[df["name"]     == selected_product]
if selected_category != "All": df = df[df["category"] == selected_category]

df = calculate_kpis(df)

# ── KPI SUMMARY CARDS ─────────────────────────────────────────────────────────
st.subheader("📊 KPI Summary")

summary      = get_kpi_summary(df)
summary_prev = get_kpi_summary(calculate_kpis(
    df[df["date"] < df["date"].max() - pd.Timedelta(days=6)]
)) if len(df["date"].unique()) > 7 else None

def _delta(key, fmt=".2f", suffix=""):
    if summary_prev is None:
        return None
    prev = summary_prev.get(key, 0)
    curr = summary.get(key, 0)
    if prev == 0:
        return None
    pct = (curr - prev) / abs(prev) * 100
    return f"{pct:+.1f}% vs prev week"

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Revenue",     f"${summary['total_revenue']:,.0f}",    _delta("total_revenue"))
col2.metric("Avg ROAS",          f"{summary['avg_roas']:.2f}x",          _delta("avg_roas"))
col3.metric("Avg ACOS",          f"{summary['avg_acos']:.1%}",           _delta("avg_acos"))
col4.metric("Avg CTR",           f"{summary['avg_ctr']:.2%}",            _delta("avg_ctr"))
col5.metric("Avg CPC",           f"${summary['avg_cpc']:.2f}",           _delta("avg_cpc"))
col6.metric("Avg Conv. Rate",    f"{summary['avg_conversion_rate']:.2%}", _delta("avg_conversion_rate"))

# ── REVENUE TREND ─────────────────────────────────────────────────────────────
st.subheader("📈 Revenue Trend by Category")
revenue_trend = df.groupby(["date", "category"])["revenue"].sum().reset_index()
revenue_trend = revenue_trend.pivot(index="date", columns="category", values="revenue")
st.line_chart(revenue_trend)

# ── ARIMA FORECAST ────────────────────────────────────────────────────────────
from analysis.forecasting import forecast_revenue
import plotly.graph_objects as go

st.subheader("🔮 Revenue Forecast (Next 30 Days)")
active_filter = selected_product if selected_product != "All" else (selected_category if selected_category != "All" else "All Products")
st.caption(f"Forecast for: **{active_filter}**")

@st.cache_data
def get_forecast(days, product_filter, category_filter):
    return forecast_revenue(days, product_filter=product_filter, category_filter=category_filter)

forecast = get_forecast(30, selected_product, selected_category)
today    = pd.Timestamp("today").normalize()
history  = forecast[forecast["ds"] <= today]
future   = forecast[forecast["ds"] >  today]

fig = go.Figure()
fig.add_trace(go.Scatter(x=history["ds"], y=history["yhat"], mode="lines",
    name="Historical Revenue", line=dict(color="#1f77b4", width=2)))
fig.add_trace(go.Scatter(
    x=pd.concat([future["ds"], future["ds"][::-1]]),
    y=pd.concat([future["yhat_upper"], future["yhat_lower"][::-1]]),
    fill="toself", fillcolor="rgba(255,127,14,0.2)",
    line=dict(color="rgba(255,255,255,0)"),
    name="80% Confidence Interval", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=future["ds"], y=future["yhat"], mode="lines",
    name="Forecast", line=dict(color="#ff7f0e", width=2, dash="dash")))
fig.update_layout(
    title="Revenue Forecast — Next 30 Days (ARIMA)",
    xaxis_title="Date", yaxis_title="Revenue ($)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified", plot_bgcolor=None)
st.plotly_chart(fig, use_container_width=True)

# ── TOP & WORST PERFORMERS ────────────────────────────────────────────────────
st.subheader("🏆 Product Performance")
perf_col1, perf_col2 = st.columns(2)

with perf_col1:
    st.markdown("**Top Sellers — Total Revenue**")
    top = get_top_sellers(df, top_n=5).reset_index()
    top.columns = ["Product", "Revenue"]
    top["Revenue"] = top["Revenue"].apply(lambda x: f"${x:,.0f}")
    st.dataframe(top, use_container_width=True, hide_index=True)

with perf_col2:
    st.markdown("**Worst Performers — ROAS**")
    worst = get_worst_performers(df, top_n=5).reset_index()
    worst.columns = ["Product", "ROAS"]
    worst["ROAS"] = worst["ROAS"].apply(lambda x: f"{x:.2f}x")
    st.dataframe(worst, use_container_width=True, hide_index=True)

# ── WEEK-OVER-WEEK ────────────────────────────────────────────────────────────
st.subheader("📅 Week-over-Week Revenue Change")

wow = calculate_week_over_week_change(df)
wow["this_week"] = wow["this_week"].apply(lambda x: f"${x:,.0f}")
wow["last_week"] = wow["last_week"].apply(lambda x: f"${x:,.0f}")
wow["change_pct"] = wow["change_pct"].apply(
    lambda x: f"{'📈' if x > 0 else '📉'} {x:+.1f}%"
)
wow.columns = ["Product", "This Week", "Last Week", "Change"]
st.dataframe(wow, use_container_width=True, hide_index=True,
    column_config={
        "Product":   st.column_config.TextColumn(width="medium"),
        "This Week": st.column_config.TextColumn(width="small"),
        "Last Week": st.column_config.TextColumn(width="small"),
        "Change":    st.column_config.TextColumn(width="small"),
    })

# ── ANOMALY DETECTION ─────────────────────────────────────────────────────────
st.subheader("🔍 Anomaly Detection")

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

anomalies = detect_anomalies(df)

if not anomalies:
    st.success("✅ No anomalies detected in the last 7 days.")
else:
    an_df   = pd.DataFrame(anomalies)
    spikes  = sum(1 for a in anomalies if a["Type"] == "📈 Spike")
    drops   = sum(1 for a in anomalies if a["Type"] == "📉 Drop")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Anomalies", len(anomalies))
    c2.metric("📈 Spikes", spikes)
    c3.metric("📉 Drops",  drops)
    st.dataframe(an_df, use_container_width=True, hide_index=True,
        column_config={
            "Product":  st.column_config.TextColumn("Product",  width="medium"),
            "Category": st.column_config.TextColumn("Category", width="small"),
            "Date":     st.column_config.TextColumn("Date",     width="small"),
            "Revenue":  st.column_config.TextColumn("Revenue",  width="small"),
            "Expected": st.column_config.TextColumn("Expected (mean ± std)", width="medium"),
            "Z-Score":  st.column_config.NumberColumn("Z-Score", format="%.2f", width="small"),
            "Type":     st.column_config.TextColumn("Type",     width="small"),
        })

# ── RECOMMENDATIONS ───────────────────────────────────────────────────────────
st.subheader("🎯 Automated Recommendations")

recommendations = get_recommendations()

if not recommendations:
    st.success("✅ No optimization recommendations for today. All products look healthy!")
elif isinstance(recommendations[0], str):
    st.warning("⚠️ **recommendation_engine.py** is outdated. Please replace it with the latest version.")
    for rec in recommendations:
        st.info(rec)
else:
    rec_df = pd.DataFrame(recommendations)
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        all_types = ["All"] + sorted(rec_df["Alert"].unique().tolist())
        sel_type  = st.selectbox("Filter by Alert",    all_types)
    with f_col2:
        all_cats  = ["All"] + sorted(rec_df["Category"].unique().tolist())
        sel_cat   = st.selectbox("Filter by Category", all_cats)
    with f_col3:
        search = st.text_input("Search product", placeholder="e.g. Wireless Mouse")

    filtered = rec_df.copy()
    if sel_type != "All": filtered = filtered[filtered["Alert"]    == sel_type]
    if sel_cat  != "All": filtered = filtered[filtered["Category"] == sel_cat]
    if search:            filtered = filtered[filtered["Product"].str.contains(search, case=False, na=False)]

    st.caption(f"Showing {len(filtered)} of {len(rec_df)} recommendation(s) · last 7 days vs previous 7 days")
    display_df = filtered.drop(columns=["Type"]).reset_index(drop=True)
    st.dataframe(display_df, use_container_width=True, hide_index=True,
        column_config={
            "Product":   st.column_config.TextColumn("Product",   width="medium"),
            "Category":  st.column_config.TextColumn("Category",  width="small"),
            "Alert":     st.column_config.TextColumn("Alert",     width="medium"),
            "Metric":    st.column_config.TextColumn("Metric",    width="small"),
            "This Week": st.column_config.TextColumn("This Week", width="small"),
            "Last Week": st.column_config.TextColumn("Last Week", width="small"),
            "Trend":     st.column_config.TextColumn("Trend",     width="small"),
            "Action":    st.column_config.TextColumn("Action",    width="large"),
        })