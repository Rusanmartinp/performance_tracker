import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from scipy import stats
from dotenv import load_dotenv
import os

# CONFIG
load_dotenv()

#LOAD DATA

def load_data():
    DB_URL = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
    engine = create_engine(DB_URL)
    query = """
        SELECT
            dp.date,
            p.name,
            dp.impressions,
            dp.clicks,
            dp.ad_spend,
            dp.units_sold,
            dp.revenue
        FROM daily_performance dp
        JOIN products p ON dp.product_id = p.id
        ORDER BY dp.date;
    """
    return pd.read_sql(query, engine)

#Z-score calculation
def calculate_z_scores(series):
    mean = series.mean()
    std = series.std()

    if std == 0:
        return 0
    latest_value = series.iloc[-1]
    z_scores = (latest_value - mean) / std
    return z_scores


#Hypothesis testing
def generate_hypothesis(df_product):
    latest = df_product.iloc[-1]
    baseline = df_product.iloc[:-1]

    impressions_change = (
        (latest['impressions'] - baseline['impressions'].mean()) / baseline['impressions'].mean()
    )

    conversion_rate_latest = latest['units_sold'] / max(latest['clicks'], 1) if latest['clicks'] > 0 else 0
    conversion_rate_baseline = (
        baseline["units_sold"].sum() / max(baseline["clicks"].sum(), 1) if baseline["clicks"].sum() > 0 else 0
    )

    if impressions_change < -0.1:
        return "Revenue drop likely due to decreased impressions. Consider increasing ad spend or optimizing targeting."
    elif conversion_rate_latest < conversion_rate_baseline:
        return "Possible conversion issue. Review product listing, pricing, and ad creatives for potential improvements."
    else:
        return "Potential efficiency or bidding change."
    

    #MAIN ANOMALY DETECTION
def detect_anomalies(df, threshold = 2):
    alerts = []
    for product in df['name'].unique(): 
        df_product = df[df['name'] == product].sort_values('date')
        if len(df_product) < 7:
            continue
        z_revenue = calculate_z_scores(df_product['revenue'])
        z_roas = calculate_z_scores(
            df_product['revenue'] / df_product['ad_spend'].replace({0: 1})
        )

        if abs(z_revenue) > threshold or abs(z_roas) > threshold:
            hypothesis = generate_hypothesis(df_product)
            alerts.append({
                "product": product,
                "z_revenue": z_revenue,
                "z_roas": z_roas,
                "hypothesis": hypothesis
            })
    return alerts

#RUN

def run_anomaly_detection():
    df = load_data()
    alerts = detect_anomalies(df)
    if not alerts:
        print("No anomalies detected.")
    else:
        for alert in alerts:
            print(f"\nProduct: {alert['product']}")
            print(f"Z-Score Revenue: {alert['z_revenue']:.2f}")
            print(f"Z-Score ROAS: {alert['z_roas']:.2f}")
            print(f"Hypothesis: {alert['hypothesis']}")

if __name__ == "__main__":
    run_anomaly_detection()