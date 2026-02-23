import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

#CONFIG
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
engine = create_engine(DB_URL)

def load_data():
    query = """
        SELECT
            dp.date,
            p.name,
            p.category,
            dp.revenue,
            dp.ad_spend,
            pr.discount_percent,
            pr.start_date,
            pr.end_date
        FROM daily_performance dp
        JOIN products p ON dp.product_id = p.id
        JOIN promotions pr ON dp.product_id = pr.product_id
        ORDER BY dp.date;
    """
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])
    return df


def analyze_promotions(df):
    results = []
    df["date"] = pd.to_datetime(df["date"])
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    for product in df["name"].unique():
        df_product = df[df["name"] == product]
        promo_start = df_product["start_date"].iloc[0]
        promo_end = df_product["end_date"].iloc[0]

        before = df_product[df_product["date"] < promo_start]
        during = df_product[(df_product["date"] >= promo_start) & 
                            (df_product["date"] <= promo_end)]
        
        if len(before) < 3 or len(during) < 1:
            continue

        baseline = before["revenue"].mean()
        promo_avg = during["revenue"].mean()

        lift = ((promo_avg - baseline) / baseline) if baseline != 0 else 0

        results.append({
            "product": product,
            "baseline_revenue": round(baseline, 2),
            "promo_revenue": round(promo_avg, 2),
            "lift_percent": round(lift * 100, 2)
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    df = load_data()
    promo_analysis = analyze_promotions(df)
    print(promo_analysis)

