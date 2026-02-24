from fastapi import FastAPI
from datetime import date, timedelta
import random


app = FastAPI(title = "Simulated Ads API")

#Simulated product catalog

PRODUCTS = [
    {"id": 1, "name": "Wireless Mouse", "price": 25.99, "category": "Accessories"},
    {"id": 2, "name": "Mechanical Keyboard", "price": 89.99, "category": "Accessories"},
    {"id": 3, "name": "USB-C Hub", "price": 45.50, "category": "Accessories"},
    {"id": 4, "name": "Gaming Headset", "price": 79.90, "category": "Audio"},
    {"id": 5, "name": "Laptop Stand", "price": 39.99, "category": "Office"},
    {"id": 6, "name": "Webcam HD", "price": 59.99, "category": "Electronics"},
    {"id": 7, "name": "Bluetooth Speaker", "price": 99.99, "category": "Audio"},
    {"id": 8, "name": "External SSD 1TB", "price": 129.99, "category": "Electronics"},
    {"id": 9, "name": "Monitor 27 inch", "price": 249.99, "category": "Electronics"},
    {"id": 10, "name": "Office Chair", "price": 199.99, "category": "Office"},
    {"id": 11, "name": "Desk Lamp", "price": 29.99, "category": "Office"},
    {"id": 12, "name": "Smartphone Stand", "price": 19.99, "category": "Accessories"},
    {"id": 13, "name": "Wireless Charger", "price": 34.99, "category": "Accessories"},
    {"id": 14, "name": "Noise Cancelling Headphones", "price": 199.99, "category": "Audio"},
    {"id": 15, "name": "Tablet 10 inch", "price": 299.99, "category": "Electronics"},
]

import math

# Each product has a personality: base demand, price sensitivity, and growth stage
PRODUCT_PROFILES = {
    1:  {"base": 800,  "growth": 0.0010, "volatility": 0.06},  # Wireless Mouse       - stable mature
    2:  {"base": 300,  "growth": 0.0020, "volatility": 0.08},  # Mechanical Keyboard  - slow growth
    3:  {"base": 500,  "growth": 0.0008, "volatility": 0.07},  # USB-C Hub            - stable
    4:  {"base": 250,  "growth": 0.0015, "volatility": 0.10},  # Gaming Headset       - growing
    5:  {"base": 400,  "growth": 0.0005, "volatility": 0.06},  # Laptop Stand         - stable
    6:  {"base": 350,  "growth": 0.0012, "volatility": 0.09},  # Webcam HD            - moderate growth
    7:  {"base": 200,  "growth": 0.0018, "volatility": 0.11},  # Bluetooth Speaker    - growing
    8:  {"base": 150,  "growth": 0.0025, "volatility": 0.09},  # External SSD         - fast growth
    9:  {"base": 80,   "growth": 0.0030, "volatility": 0.12},  # Monitor 27"          - fast growth, high ticket
    10: {"base": 60,   "growth": 0.0010, "volatility": 0.08},  # Office Chair         - stable, high ticket
    11: {"base": 600,  "growth": 0.0003, "volatility": 0.05},  # Desk Lamp            - very stable
    12: {"base": 700,  "growth": 0.0005, "volatility": 0.06},  # Smartphone Stand     - stable
    13: {"base": 450,  "growth": 0.0015, "volatility": 0.07},  # Wireless Charger     - growing
    14: {"base": 100,  "growth": 0.0020, "volatility": 0.10},  # Noise Cancelling HP  - growing, high ticket
    15: {"base": 70,   "growth": 0.0028, "volatility": 0.13},  # Tablet               - fast growth, high ticket
}

def generate_daily_metrics(product, target_date):
    from datetime import date as date_type
    profile = PRODUCT_PROFILES[product["id"]]

    # --- Trend: slow linear growth over time ---
    days_since_start = (target_date - date_type(2025, 1, 1)).days
    trend = 1 + profile["growth"] * days_since_start

    # --- Weekly seasonality: weekends +25%, Monday slow ---
    weekday = target_date.weekday()
    weekly = {0: 0.88, 1: 0.95, 2: 1.00, 3: 1.02, 4: 1.05, 5: 1.25, 6: 1.20}.get(weekday, 1.0)

    # --- Monthly seasonality: mid-month dip, end-of-month boost ---
    monthly = 1 + 0.10 * math.sin((target_date.day / 31) * 2 * math.pi - math.pi / 2)

    # --- Controlled noise: gaussian, tight around 1.0 ---
    noise = max(0.80, min(1.20, random.gauss(1.0, profile["volatility"])))

    # --- Rare campaign spike (3% chance, max +50%) ---
    spike = random.uniform(1.25, 1.50) if random.random() < 0.03 else 1.0

    # --- Compose multiplier ---
    multiplier = trend * weekly * monthly * noise * spike

    # --- Impressions ---
    base_impressions = int(profile["base"] * 20 * multiplier)
    impressions = max(500, base_impressions)

    # --- CTR: stable per product with small daily variation ---
    base_ctr = 0.025 + (product["id"] % 5) * 0.004  # 0.025 to 0.041 depending on product
    ctr = max(0.010, min(0.08, random.gauss(base_ctr, 0.003)))
    clicks = int(impressions * ctr)

    # --- Conversion rate: stable, slight noise ---
    base_cr = 0.10 + (product["id"] % 4) * 0.02   # 0.10 to 0.16
    conversion_rate = max(0.05, min(0.30, random.gauss(base_cr, 0.015)))
    units_sold = max(0, int(clicks * conversion_rate))

    revenue = round(units_sold * product["price"], 2)
    ad_spend = round(clicks * random.gauss(0.55, 0.05), 2)

    return {
        "product_id": product["id"],
        "date": str(target_date),
        "impressions": impressions,
        "clicks": clicks,
        "ad_spend": max(0, ad_spend),
        "units_sold": units_sold,
        "revenue": revenue,
    }

@app.get("/products")
def get_products():
    return PRODUCTS

@app.get("/daily-performance")
def get_daily_performance(days_back: int = 0):
    data = []
    for i in range(days_back + 1):
        target_date = date.today() - timedelta(days=i)

        for product in PRODUCTS:
            metrics = generate_daily_metrics(product, target_date)
            metrics["date"] = str(target_date)
            data.append(metrics)
    return data