import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()


def load_data():
    DB_URL = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
    engine = create_engine(DB_URL)
    query = """
        SELECT
            dp.date,
            p.name,
            p.category,
            dp.impressions,
            dp.clicks,
            dp.ad_spend,
            dp.units_sold,
            dp.revenue
        FROM daily_performance dp
        JOIN products p ON dp.product_id = p.id
        ORDER BY dp.date;
    """
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])
    df["CTR"] = df["clicks"] / df["impressions"].replace(0, 1)
    df["conversion_rate"] = df["units_sold"] / df["clicks"].replace(0, 1)
    df["ROAS"] = df["revenue"] / df["ad_spend"].replace(0, 1)
    df["ACOS"] = df["ad_spend"] / df["revenue"].replace(0, 1)
    return df


def _week_avg(df_product, latest_date, week_offset=0):
    """Return the mean of each metric for a given 7-day window.
    Metrics are computed here from raw columns so the function works
    with any DataFrame regardless of whether KPIs were pre-calculated.
    week_offset=0 → last 7 days, week_offset=1 → previous 7 days."""
    end   = latest_date - pd.Timedelta(days=7 * week_offset)
    start = end - pd.Timedelta(days=6)
    w = df_product[(df_product["date"] >= start) & (df_product["date"] <= end)].copy()
    if w.empty:
        return None
    w["CTR"]             = w["clicks"]     / w["impressions"].replace(0, 1)
    w["conversion_rate"] = w["units_sold"] / w["clicks"].replace(0, 1)
    w["ROAS"]            = w["revenue"]    / w["ad_spend"].replace(0, 1)
    w["ACOS"]            = w["ad_spend"]   / w["revenue"].replace(0, 1)
    return w[["CTR", "conversion_rate", "ROAS", "ACOS", "revenue"]].mean()


def _trend_arrow(current, previous, higher_is_better=True):
    """Return a trend symbol and direction label."""
    if previous is None or previous == 0:
        return "➡️", "no prior data"
    pct = (current - previous) / abs(previous) * 100
    if abs(pct) < 1:
        return "➡️", "stable"
    if higher_is_better:
        return ("📈", f"+{pct:.1f}%") if pct > 0 else ("📉", f"{pct:.1f}%")
    else:
        return ("📉", f"+{pct:.1f}%") if pct > 0 else ("📈", f"{pct:.1f}%")


def generate_recommendations(df):
    latest_date = df["date"].max()
    recommendations = []

    for product, df_product in df.groupby("name"):
        category = df_product["category"].iloc[0]

        this_week = _week_avg(df_product, latest_date, week_offset=0)
        last_week = _week_avg(df_product, latest_date, week_offset=1)

        if this_week is None:
            continue

        lw = last_week  # shorthand — may be None

        # --- ACOS too high ---
        if this_week["ACOS"] > 0.35:
            arrow, trend_label = _trend_arrow(
                this_week["ACOS"],
                lw["ACOS"] if lw is not None else None,
                higher_is_better=False
            )
            recommendations.append({
                "Product": product,
                "Category": category,
                "Alert": "⚠️ High ACOS",
                "Type": "warning",
                "Metric": "ACOS",
                "This Week": f"{this_week['ACOS']:.1%}",
                "Last Week": f"{lw['ACOS']:.1%}" if lw is not None else "—",
                "Trend": f"{arrow} {trend_label}",
                "Action": "Reduce bids or pause low-converting keywords.",
            })

        # --- ROAS strong ---
        if this_week["ROAS"] > 4:
            arrow, trend_label = _trend_arrow(
                this_week["ROAS"],
                lw["ROAS"] if lw is not None else None,
                higher_is_better=True
            )
            recommendations.append({
                "Product": product,
                "Category": category,
                "Alert": "🚀 Strong ROAS",
                "Type": "success",
                "Metric": "ROAS",
                "This Week": f"{this_week['ROAS']:.2f}x",
                "Last Week": f"{lw['ROAS']:.2f}x" if lw is not None else "—",
                "Trend": f"{arrow} {trend_label}",
                "Action": "Scale ad budget to capture more demand.",
            })

        # --- CTR low ---
        if this_week["CTR"] < 0.02:
            arrow, trend_label = _trend_arrow(
                this_week["CTR"],
                lw["CTR"] if lw is not None else None,
                higher_is_better=True
            )
            recommendations.append({
                "Product": product,
                "Category": category,
                "Alert": "🖼️ Low CTR",
                "Type": "warning",
                "Metric": "CTR",
                "This Week": f"{this_week['CTR']:.2%}",
                "Last Week": f"{lw['CTR']:.2%}" if lw is not None else "—",
                "Trend": f"{arrow} {trend_label}",
                "Action": "A/B test main image or rewrite product title.",
            })

        # --- Conversion rate low ---
        if this_week["conversion_rate"] < 0.08:
            arrow, trend_label = _trend_arrow(
                this_week["conversion_rate"],
                lw["conversion_rate"] if lw is not None else None,
                higher_is_better=True
            )
            recommendations.append({
                "Product": product,
                "Category": category,
                "Alert": "💡 Low Conversion",
                "Type": "info",
                "Metric": "Conv. Rate",
                "This Week": f"{this_week['conversion_rate']:.2%}",
                "Last Week": f"{lw['conversion_rate']:.2%}" if lw is not None else "—",
                "Trend": f"{arrow} {trend_label}",
                "Action": "Review pricing, add reviews, or improve bullet points.",
            })

        # --- Revenue declining ---
        if lw is not None and this_week["revenue"] < lw["revenue"] * 0.85:
            pct = (this_week["revenue"] - lw["revenue"]) / lw["revenue"] * 100
            recommendations.append({
                "Product": product,
                "Category": category,
                "Alert": "📉 Revenue Drop",
                "Type": "warning",
                "Metric": "Revenue",
                "This Week": f"${this_week['revenue']:,.0f}",
                "Last Week": f"${lw['revenue']:,.0f}",
                "Trend": f"📉 {pct:.1f}%",
                "Action": "Investigate stock levels, pricing changes, or ad budget drops.",
            })

    # Sort: warnings first, then info, then success
    order = {"warning": 0, "info": 1, "success": 2}
    recommendations.sort(key=lambda r: order.get(r["Type"], 9))

    return recommendations


def get_recommendations():
    df = load_data()
    return generate_recommendations(df)