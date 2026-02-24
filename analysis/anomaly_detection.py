import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    DB_URL = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
    engine = create_engine(DB_URL)
    query = """
        SELECT dp.date, p.name, p.category,
               dp.impressions, dp.clicks,
               dp.ad_spend, dp.units_sold, dp.revenue
        FROM daily_performance dp
        JOIN products p ON dp.product_id = p.id
        ORDER BY dp.date;
    """
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Core detection — used by both dashboard and CLI
# ---------------------------------------------------------------------------

def detect_anomalies(df, z_threshold=1.8, window_days=7):
    """
    For each product, flag any day in the last `window_days` where
    revenue deviates more than `z_threshold` standard deviations
    from that product's historical mean.

    Returns a list of dicts sorted by |Z-Score| descending, ready
    to be passed directly to pd.DataFrame() for display.
    """
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
# CLI entry point
# ---------------------------------------------------------------------------

def run_anomaly_detection():
    df = load_data()
    anomalies = detect_anomalies(df)
    if not anomalies:
        print("No anomalies detected.")
    else:
        for a in anomalies:
            print(f"\n{a['Type']} {a['Product']} on {a['Date']}")
            print(f"  Revenue:  {a['Revenue']}  (expected {a['Expected']})")
            print(f"  Z-Score:  {a['Z-Score']}")


if __name__ == "__main__":
    run_anomaly_detection()