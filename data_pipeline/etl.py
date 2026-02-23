import requests
import pandas as pd
from sqlalchemy import create_engine, text
import psycopg2
import math
import random
from datetime import datetime

from dotenv import load_dotenv
import os

# CONFIG
load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# --- Simulation function for trend and seasonality effects ---
def product_trend_multiplier(product_id, date_str, category):
    """
    Applies a small category-level multiplier only.
    All major trend/seasonality/noise is now handled in the API layer
    to avoid stacking random effects that destroy ARIMA's signal.
    """
    if category == "Electronics":
        return 1.08
    elif category == "Audio":
        return 1.05
    elif category == "Office":
        return 0.97
    else:  # Accessories etc.
        return 1.00

# --- EXTRACT ---
def fetch_products():
    response = requests.get(f"{API_BASE_URL}/products")
    return response.json()

def fetch_daily_performance():
    response = requests.get(f"{API_BASE_URL}/daily-performance?days_back=90")
    return response.json()

# --- LOAD ---
def _get_engine():
    DB_URL = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
    return create_engine(DB_URL)

def load_products(products):
    with _get_engine().begin() as conn:
        for p in products:
            conn.execute(
                text("""
                    INSERT INTO products (id, name, price, category)
                    VALUES (:id, :name, :price, :category)
                    ON CONFLICT (id)
                     DO UPDATE SET name = EXCLUDED.name, 
                     price = EXCLUDED.price, 
                     category = EXCLUDED.category
                """),
                p
            )

def load_daily_performance(data, product_categories):
    with _get_engine().begin() as conn:
        for d in data:
            # Get category from product_id
            category = product_categories.get(d["product_id"], "Unknown")

            # Apply trend/seasonality multiplier
            multiplier = product_trend_multiplier(d["product_id"], d["date"], category)

            d["units_sold"] = int(d["units_sold"] * multiplier)
            d["revenue"] = round(d["revenue"] * multiplier, 2)

            conn.execute(
                text("""
                    INSERT INTO daily_performance (
                        date, product_id, impressions,
                        clicks, ad_spend, units_sold, revenue
                    )
                    VALUES (
                        :date, :product_id, :impressions,
                        :clicks, :ad_spend, :units_sold, :revenue
                    )
                    ON CONFLICT (date, product_id) DO NOTHING
                """),
                d
            )

# --- MAIN ETL ---
def run_etl():
    print("Fetching products...")
    products = fetch_products()
    load_products(products)

    # Build product_id -> category lookup
    product_categories = {p["id"]: p["category"] for p in products}

    print("Fetching daily performance data...")
    performance = fetch_daily_performance()
    load_daily_performance(performance, product_categories)

    print("ETL process completed successfully.")

if __name__ == "__main__":
    run_etl()