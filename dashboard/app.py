import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.recommendation_engine import get_recommendations
from analysis.anomaly_detection import detect_anomalies
from analysis.kpi_analysis import (
    calculate_kpis, get_kpi_summary, get_top_sellers,
    get_worst_performers, calculate_week_over_week_change,
)
from analysis.forecasting import forecast_revenue, forecast_revenue_ma

load_dotenv()

st.set_page_config(page_title="Performance Tracker", layout="wide")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("", ["📊 Overview", "🔍 Product Detail"])
    st.markdown("---")
    st.header("Filters")

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
with st.sidebar:
    selected_product  = st.selectbox("Product",  ["All"] + sorted(df_full["name"].unique().tolist()))
    category_options  = df_full["category"].unique() if selected_product == "All" else df_full[df_full["name"] == selected_product]["category"].unique()
    selected_category = st.selectbox("Category", ["All"] + sorted(category_options.tolist()))

df = df_full.copy()
if selected_product  != "All": df = df[df["name"]     == selected_product]
if selected_category != "All": df = df[df["category"] == selected_category]
df = calculate_kpis(df)

active_filter = selected_product if selected_product != "All" else (selected_category if selected_category != "All" else "All Products")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📦 E-commerce Performance Tracker")

    # ── KPI CARDS ─────────────────────────────────────────────────────────────
    st.subheader("📊 KPI Summary")
    summary      = get_kpi_summary(df)
    summary_prev = get_kpi_summary(calculate_kpis(
        df[df["date"] < df["date"].max() - pd.Timedelta(days=6)]
    )) if len(df["date"].unique()) > 7 else None

    def _delta(key):
        if summary_prev is None: return None
        prev = summary_prev.get(key, 0)
        curr = summary.get(key, 0)
        if prev == 0: return None
        return f"{(curr - prev) / abs(prev) * 100:+.1f}% vs prev week"

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Revenue",  f"${summary['total_revenue']:,.0f}",     _delta("total_revenue"))
    c2.metric("Avg ROAS",       f"{summary['avg_roas']:.2f}x",           _delta("avg_roas"))
    c3.metric("Avg ACOS",       f"{summary['avg_acos']:.1%}",            _delta("avg_acos"))
    c4.metric("Avg CTR",        f"{summary['avg_ctr']:.2%}",             _delta("avg_ctr"))
    c5.metric("Avg CPC",        f"${summary['avg_cpc']:.2f}",            _delta("avg_cpc"))
    c6.metric("Avg Conv. Rate", f"{summary['avg_conversion_rate']:.2%}", _delta("avg_conversion_rate"))

    # ── REVENUE TREND ─────────────────────────────────────────────────────────
    st.subheader("📈 Revenue Trend by Category")
    revenue_trend = df.groupby(["date","category"])["revenue"].sum().reset_index()
    revenue_trend = revenue_trend.pivot(index="date", columns="category", values="revenue")
    st.line_chart(revenue_trend)

    # ── DUAL FORECAST ─────────────────────────────────────────────────────────
    import plotly.graph_objects as go

    st.subheader("🔮 Revenue Forecast (Next 30 Days)")
    st.caption(f"Forecast for: **{active_filter}** · ARIMA vs Moving Average (7-day window)")

    @st.cache_data
    def get_forecasts(days, pf, cf):
        return (
            forecast_revenue(days, product_filter=pf, category_filter=cf),
            forecast_revenue_ma(days, product_filter=pf, category_filter=cf),
        )

    fc_arima, fc_ma = get_forecasts(30, selected_product, selected_category)
    today = pd.Timestamp("today").normalize()

    def _split(fc):
        return fc[fc["ds"] <= today], fc[fc["ds"] > today]

    hist_a, fut_a = _split(fc_arima)
    _,       fut_m = _split(fc_ma)

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(x=hist_a["ds"], y=hist_a["yhat"], mode="lines",
        name="Historical Revenue", line=dict(color="#1f77b4", width=2)))

    # ARIMA CI + line
    fig.add_trace(go.Scatter(
        x=pd.concat([fut_a["ds"], fut_a["ds"][::-1]]),
        y=pd.concat([fut_a["yhat_upper"], fut_a["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(255,127,14,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="ARIMA 80% CI", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fut_a["ds"], y=fut_a["yhat"], mode="lines",
        name="ARIMA Forecast", line=dict(color="#ff7f0e", width=2, dash="dash")))

    # MA CI + line
    fig.add_trace(go.Scatter(
        x=pd.concat([fut_m["ds"], fut_m["ds"][::-1]]),
        y=pd.concat([fut_m["yhat_upper"], fut_m["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(44,160,44,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="MA 80% CI", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fut_m["ds"], y=fut_m["yhat"], mode="lines",
        name="MA Forecast (7d)", line=dict(color="#2ca02c", width=2, dash="dot")))

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Revenue ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", plot_bgcolor=None)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🟠 ARIMA captures trend & seasonality patterns. 🟢 Moving Average is a simpler baseline. When they agree, the forecast is more reliable.")

    # ── TOP & WORST ───────────────────────────────────────────────────────────
    st.subheader("🏆 Product Performance")
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown("**Top Sellers — Total Revenue**")
        top = get_top_sellers(df, top_n=5).reset_index()
        top.columns = ["Product", "Revenue"]
        top["Revenue"] = top["Revenue"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(top, use_container_width=True, hide_index=True)
    with pc2:
        st.markdown("**Worst Performers — ROAS**")
        worst = get_worst_performers(df, top_n=5).reset_index()
        worst.columns = ["Product", "ROAS"]
        worst["ROAS"] = worst["ROAS"].apply(lambda x: f"{x:.2f}x")
        st.dataframe(worst, use_container_width=True, hide_index=True)

    # ── WEEK-OVER-WEEK ────────────────────────────────────────────────────────
    st.subheader("📅 Week-over-Week Revenue Change")
    wow = calculate_week_over_week_change(df)
    wow["this_week"]  = wow["this_week"].apply(lambda x: f"${x:,.0f}")
    wow["last_week"]  = wow["last_week"].apply(lambda x: f"${x:,.0f}")
    wow["change_pct"] = wow["change_pct"].apply(lambda x: f"{'📈' if x > 0 else '📉'} {x:+.1f}%")
    wow.columns = ["Product", "This Week", "Last Week", "Change"]
    st.dataframe(wow, use_container_width=True, hide_index=True)

    # ── ANOMALY DETECTION ─────────────────────────────────────────────────────
    st.subheader("🔍 Anomaly Detection")
    anomalies = detect_anomalies(df)
    if not anomalies:
        st.success("✅ No anomalies detected in the last 7 days.")
    else:
        an_df  = pd.DataFrame(anomalies)
        spikes = sum(1 for a in anomalies if a["Type"] == "📈 Spike")
        drops  = sum(1 for a in anomalies if a["Type"] == "📉 Drop")
        ac1, ac2, ac3 = st.columns(3)
        ac1.metric("Total Anomalies", len(anomalies))
        ac2.metric("📈 Spikes", spikes)
        ac3.metric("📉 Drops",  drops)
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

    # ── RECOMMENDATIONS ───────────────────────────────────────────────────────
    st.subheader("🎯 Automated Recommendations")
    recommendations = get_recommendations()
    if not recommendations:
        st.success("✅ No optimization recommendations for today.")
    elif isinstance(recommendations[0], str):
        st.warning("⚠️ recommendation_engine.py is outdated.")
        for rec in recommendations: st.info(rec)
    else:
        rec_df = pd.DataFrame(recommendations)
        rf1, rf2, rf3 = st.columns(3)
        with rf1:
            sel_type = st.selectbox("Filter by Alert",    ["All"] + sorted(rec_df["Alert"].unique().tolist()))
        with rf2:
            sel_cat  = st.selectbox("Filter by Category", ["All"] + sorted(rec_df["Category"].unique().tolist()))
        with rf3:
            search = st.text_input("Search product", placeholder="e.g. Wireless Mouse")
        filtered = rec_df.copy()
        if sel_type != "All": filtered = filtered[filtered["Alert"]    == sel_type]
        if sel_cat  != "All": filtered = filtered[filtered["Category"] == sel_cat]
        if search:            filtered = filtered[filtered["Product"].str.contains(search, case=False, na=False)]
        st.caption(f"Showing {len(filtered)} of {len(rec_df)} recommendation(s) · last 7 days vs previous 7 days")
        st.dataframe(filtered.drop(columns=["Type"]).reset_index(drop=True),
            use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PRODUCT DETAIL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Product Detail":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("🔍 Product Detail")

    product = st.selectbox("Select a product", sorted(df_full["name"].unique().tolist()))
    df_p = calculate_kpis(df_full[df_full["name"] == product].copy())

    if df_p.empty:
        st.warning("No data for this product.")
        st.stop()

    category = df_p["category"].iloc[0]
    st.markdown(f"**Category:** {category} &nbsp;·&nbsp; **Data range:** {df_p['date'].min().strftime('%b %d, %Y')} → {df_p['date'].max().strftime('%b %d, %Y')}")

    # ── KPI CARDS ─────────────────────────────────────────────────────────────
    summary_p = get_kpi_summary(df_p)
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total Revenue",   f"${summary_p['total_revenue']:,.0f}")
    k2.metric("Avg ROAS",        f"{summary_p['avg_roas']:.2f}x")
    k3.metric("Avg ACOS",        f"{summary_p['avg_acos']:.1%}")
    k4.metric("Avg CTR",         f"{summary_p['avg_ctr']:.2%}")
    k5.metric("Avg Conv. Rate",  f"{summary_p['avg_conversion_rate']:.2%}")

    # ── REVENUE + AD SPEND OVER TIME ──────────────────────────────────────────
    st.subheader("💰 Revenue & Ad Spend Over Time")
    fig_rev = go.Figure()
    fig_rev.add_trace(go.Scatter(x=df_p["date"], y=df_p["revenue"],
        mode="lines", name="Revenue", line=dict(color="#1f77b4", width=2)))
    fig_rev.add_trace(go.Scatter(x=df_p["date"], y=df_p["ad_spend"],
        mode="lines", name="Ad Spend", line=dict(color="#d62728", width=1.5, dash="dot")))
    fig_rev.update_layout(hovermode="x unified", plot_bgcolor=None,
        legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_rev, use_container_width=True)

    # ── METRICS OVER TIME ─────────────────────────────────────────────────────
    st.subheader("📐 Key Metrics Over Time")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        fig_ctr = px.line(df_p, x="date", y="CTR", title="CTR over time (Click-Through Rate = Clicks / Impressions)",
            color_discrete_sequence=["#9467bd"])
        fig_ctr.add_hline(y=0.02, line_dash="dash", line_color="red",
            annotation_text="Min threshold (2%)")
        fig_ctr.update_layout(plot_bgcolor=None)
        st.plotly_chart(fig_ctr, use_container_width=True)
    with m_col2:
        fig_roas = px.line(df_p, x="date", y="ROAS", title="ROAS over time (Return on Ad Spend = Revenue / Ad Spend)",
            color_discrete_sequence=["#2ca02c"])
        fig_roas.add_hline(y=4, line_dash="dash", line_color="green",
            annotation_text="Target (4x)")
        fig_roas.update_layout(plot_bgcolor=None)
        st.plotly_chart(fig_roas, use_container_width=True)

    # ── DUAL FORECAST FOR THIS PRODUCT ────────────────────────────────────────
    st.subheader("🔮 Revenue Forecast (Next 30 Days)")

    @st.cache_data
    def get_product_forecasts(prod, days=30):
        return (
            forecast_revenue(days, product_filter=prod),
            forecast_revenue_ma(days, product_filter=prod),
        )

    try:
        fc_a, fc_m = get_product_forecasts(product)
        today = pd.Timestamp("today").normalize()
        hist_a = fc_a[fc_a["ds"] <= today]
        fut_a  = fc_a[fc_a["ds"] >  today]
        fut_m  = fc_m[fc_m["ds"] >  today]

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=hist_a["ds"], y=hist_a["yhat"], mode="lines",
            name="Historical", line=dict(color="#1f77b4", width=2)))
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([fut_a["ds"], fut_a["ds"][::-1]]),
            y=pd.concat([fut_a["yhat_upper"], fut_a["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(255,127,14,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="ARIMA CI", hoverinfo="skip"))
        fig_fc.add_trace(go.Scatter(x=fut_a["ds"], y=fut_a["yhat"], mode="lines",
            name="ARIMA", line=dict(color="#ff7f0e", width=2, dash="dash")))
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([fut_m["ds"], fut_m["ds"][::-1]]),
            y=pd.concat([fut_m["yhat_upper"], fut_m["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(44,160,44,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="MA CI", hoverinfo="skip"))
        fig_fc.add_trace(go.Scatter(x=fut_m["ds"], y=fut_m["yhat"], mode="lines",
            name="Moving Avg", line=dict(color="#2ca02c", width=2, dash="dot")))
        fig_fc.update_layout(hovermode="x unified", plot_bgcolor=None,
            xaxis_title="Date", yaxis_title="Revenue ($)",
            legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_fc, use_container_width=True)
    except ValueError as e:
        st.warning(f"Cannot generate forecast: {e}")

    # ── ANOMALIES FOR THIS PRODUCT ────────────────────────────────────────────
    st.subheader("🔍 Recent Anomalies")
    anomalies_p = [a for a in detect_anomalies(df_p) if a["Product"] == product]
    if not anomalies_p:
        st.success("✅ No anomalies detected in the last 7 days.")
    else:
        st.dataframe(pd.DataFrame(anomalies_p), use_container_width=True, hide_index=True)

    # ── RECOMMENDATIONS ───────────────────────────────────────────────────────
    st.subheader("🎯 Recommendations")
    st.caption("Analysis based on this product's last 7 days vs previous 7 days.")

    if st.button("🔍 Analyse this product", type="primary"):
        from analysis.recommendation_engine import generate_recommendations
        recs = generate_recommendations(df_p)

        if not recs:
            st.success("✅ This product looks healthy — no actions needed.")
        else:
            for r in recs:
                icon_map = {"warning": "⚠️", "info": "💡", "success": "🚀"}
                alert_fn = {"warning": st.warning, "info": st.info, "success": st.success}
                icon  = icon_map.get(r["Type"], "📌")
                alert = alert_fn.get(r["Type"], st.info)

                alert(
                    f"**{r['Alert']}** · {r['Metric']}: {r['This Week']} "
                    f"(prev: {r['Last Week']}) {r['Trend']}"
                    f"👉 {r['Action']}")
    else:
        st.caption("Click the button to run the analysis.")