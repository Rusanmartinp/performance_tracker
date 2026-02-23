import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_performance_data():
    """Load full daily performance data including category."""
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
    return df


# ---------------------------------------------------------------------------
# KPI calculation
# ---------------------------------------------------------------------------

def calculate_kpis(df):
    """Add CTR, conversion rate, ROAS, ACOS and CPC columns to the dataframe."""
    df = df.copy()
    df["CTR"]             = df["clicks"]    / df["impressions"].replace({0: 1})
    df["conversion_rate"] = df["units_sold"] / df["clicks"].replace({0: 1})
    df["ROAS"]            = df["revenue"]   / df["ad_spend"].replace({0: 1})
    df["ACOS"]            = df["ad_spend"]  / df["revenue"].replace({0: 1})
    df["CPC"]             = df["ad_spend"]  / df["clicks"].replace({0: 1})  # Cost Per Click
    return df


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------

def get_top_sellers(df, top_n=5):
    """Top N products by total revenue."""
    return (
        df.groupby("name")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )


def get_worst_performers(df, top_n=5):
    """
    Bottom N products by ROAS (worst return on ad spend).
    Only includes products with non-zero ad spend to avoid skewed results.
    """
    product_roas = (
        df[df["ad_spend"] > 0]
        .groupby("name")
        .apply(lambda g: g["revenue"].sum() / g["ad_spend"].sum(), include_groups=False)
        .rename("ROAS")
        .sort_values(ascending=True)
        .head(top_n)
    )
    return product_roas


# ---------------------------------------------------------------------------
# Period-over-period comparisons
# ---------------------------------------------------------------------------

def calculate_day_over_day_change(df):
    """
    Revenue % change for each product: latest day vs previous day.
    Returns a Series sorted descending, or None if not enough data.
    """
    sorted_dates = sorted(df["date"].unique())
    if len(sorted_dates) < 2:
        return None

    latest_date   = sorted_dates[-1]
    previous_date = sorted_dates[-2]

    latest   = df[df["date"] == latest_date].groupby("name")["revenue"].sum()
    previous = df[df["date"] == previous_date].groupby("name")["revenue"].sum()
    change   = ((latest - previous) / previous.replace({0: 1})) * 100

    return change.sort_values(ascending=False)


def calculate_week_over_week_change(df):
    """
    Revenue % change per product: last 7 days vs previous 7 days.
    More stable than day-over-day because it smooths daily noise.
    Returns a DataFrame with columns: product, this_week, last_week, change_pct.
    """
    latest_date = df["date"].max()
    this_week_start = latest_date - pd.Timedelta(days=6)
    last_week_start = latest_date - pd.Timedelta(days=13)
    last_week_end   = latest_date - pd.Timedelta(days=7)

    this_week = (
        df[df["date"] >= this_week_start]
        .groupby("name")["revenue"].sum()
        .rename("this_week")
    )
    last_week = (
        df[(df["date"] >= last_week_start) & (df["date"] <= last_week_end)]
        .groupby("name")["revenue"].sum()
        .rename("last_week")
    )

    combined = pd.concat([this_week, last_week], axis=1).fillna(0)
    combined["change_pct"] = (
        (combined["this_week"] - combined["last_week"])
        / combined["last_week"].replace({0: 1})
    ) * 100

    return combined.sort_values("change_pct", ascending=False).reset_index()


# ---------------------------------------------------------------------------
# Aggregated summary (for dashboard consumption)
# ---------------------------------------------------------------------------

def get_kpi_summary(df):
    """
    Returns a flat dict of aggregated KPIs across all products,
    ready to be consumed directly by the dashboard or any other UI.

    Keys: total_revenue, total_ad_spend, total_clicks, total_impressions,
          total_units_sold, avg_roas, avg_acos, avg_ctr, avg_cpc,
          avg_conversion_rate, revenue_per_unit
    """
    df = calculate_kpis(df)

    total_revenue     = df["revenue"].sum()
    total_ad_spend    = df["ad_spend"].sum()
    total_clicks      = df["clicks"].sum()
    total_impressions = df["impressions"].sum()
    total_units_sold  = df["units_sold"].sum()

    return {
        "total_revenue":       round(total_revenue, 2),
        "total_ad_spend":      round(total_ad_spend, 2),
        "total_clicks":        int(total_clicks),
        "total_impressions":   int(total_impressions),
        "total_units_sold":    int(total_units_sold),
        "avg_roas":            round(total_revenue / max(total_ad_spend, 1), 2),
        "avg_acos":            round(total_ad_spend / max(total_revenue, 1), 4),
        "avg_ctr":             round(total_clicks / max(total_impressions, 1), 4),
        "avg_cpc":             round(total_ad_spend / max(total_clicks, 1), 2),
        "avg_conversion_rate": round(total_units_sold / max(total_clicks, 1), 4),
        "revenue_per_unit":    round(total_revenue / max(total_units_sold, 1), 2),
    }


# ---------------------------------------------------------------------------
# Main (CLI usage)
# ---------------------------------------------------------------------------

def run_kpi_analysis():
    print("Loading performance data...")
    df = load_performance_data()
    df = calculate_kpis(df)

    print("\n── KPI Summary ──────────────────────────")
    summary = get_kpi_summary(df)
    for key, value in summary.items():
        print(f"  {key:<25} {value}")

    print("\n── Top 5 Sellers ────────────────────────")
    print(get_top_sellers(df).to_string())

    print("\n── Worst 5 Performers (ROAS) ────────────")
    print(get_worst_performers(df).to_string())

    print("\n── Day-over-Day Revenue Change ──────────")
    dod = calculate_day_over_day_change(df)
    if dod is not None:
        print(dod.to_string())
    else:
        print("  Not enough data.")

    print("\n── Week-over-Week Revenue Change ────────")
    wow = calculate_week_over_week_change(df)
    print(wow.to_string(index=False))


if __name__ == "__main__":
    run_kpi_analysis()