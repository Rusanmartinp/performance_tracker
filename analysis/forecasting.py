import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from pmdarima import auto_arima

# Load env
load_dotenv()

def forecast_revenue(days=30, product_filter="All", category_filter="All"):
    """
    Forecast revenue for the next `days` days using ARIMA.
    Optionally filter by product name or category.
    Returns a dataframe with: ds, yhat, yhat_lower, yhat_upper.
    """

    # --- Step 1: Load aggregated revenue data ---
    DB_URL = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
    engine = create_engine(DB_URL)

    query = """
        SELECT dp.date, p.name, p.category, dp.revenue
        FROM daily_performance dp
        JOIN products p ON dp.product_id = p.id
        ORDER BY dp.date
    """
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])

    # Apply filters
    if product_filter and product_filter != "All":
        df = df[df["name"] == product_filter]
    elif category_filter and category_filter != "All":
        df = df[df["category"] == category_filter]

    # Aggregate by date
    df = df.groupby("date")["revenue"].sum().reset_index()
    df.columns = ["ds", "y"]
    df = df.set_index("ds")
    df.index.freq = pd.infer_freq(df.index)

    # Validate minimum data requirements after filtering
    if len(df) < 10:
        raise ValueError(f"Not enough data to train ARIMA for the selected filter. Got {len(df)} rows, need at least 10.")

    # --- Step 2: Fit ARIMA model with automatic parameter selection ---
    # auto_arima tests different (p,d,q) combinations and picks the best one
    # using AIC. Much more robust than a fixed order on volatile data.
    model = auto_arima(
        df["y"],
        seasonal=True,
        m=7,                # weekly seasonality
        stepwise=True,      # faster search
        suppress_warnings=True,
        error_action="ignore",
        max_p=3, max_q=3,   # keep it simple to avoid overfitting
    )
    result = model

    # --- Step 3: Generate forecast with 80% confidence interval ---
    forecast_mean, conf_int = result.predict(n_periods=days, return_conf_int=True, alpha=0.20)

    # --- Step 4: Build future dataframe ---
    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=days,
        freq="D"
    )
    forecast_df = pd.DataFrame({
        "ds": future_dates,
        "yhat": forecast_mean,
        "yhat_lower": conf_int[:, 0],
        "yhat_upper": conf_int[:, 1],
        "is_forecast": True,
    })

    # Include historical data so the plot shows context
    history_df = pd.DataFrame({
        "ds": df.index,
        "yhat": df["y"].values,
        "yhat_lower": None,
        "yhat_upper": None,
        "is_forecast": False,
    })

    return pd.concat([history_df, forecast_df], ignore_index=True)