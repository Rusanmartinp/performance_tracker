# E-commerce Performance Tracker

An end-to-end analytics pipeline that simulates a real e-commerce advertising operation — from data generation to an interactive dashboard with forecasting, anomaly detection, and automated recommendations.

Built as a portfolio project to demonstrate data engineering, analysis, and visualization skills.

---

## Dashboard Preview

| Section | Description |
|---|---|
| KPI Cards | Revenue, ROAS, Clicks, Impressions for the latest day |
| Revenue Trend | Line chart by category over time |
| ARIMA Forecast | 30-day revenue forecast with 80% confidence interval, filterable by product/category |
| Anomaly Detection | Z-score based spike/drop detection over the last 7 days |
| Recommendations | Automated action table based on 7-day vs prior 7-day trends |

---

## Dashboard Preview

<p align="center">
  <img src="images/full_dashboard.png" width="900">
</p>

---

### 🔹 KPI Cards
<p align="center">
  <img src="images/kpi.png" width="700">
</p>

### 🔹 ARIMA Forecast
<p align="center">
  <img src="images/forecast.png" width="700">
</p>

## Architecture

```
┌─────────────────────┐
│   Simulated API     │  FastAPI — generates realistic daily metrics
│   (simulated_api/)  │  per product with trend, seasonality & noise
└────────┬────────────┘
         │ HTTP
┌────────▼────────────┐
│   ETL Pipeline      │  Fetches from API, applies category multipliers,
│   (data_pipeline/)  │  loads into PostgreSQL
└────────┬────────────┘
         │ SQL
┌────────▼────────────┐
│    PostgreSQL DB    │  Stores products + daily_performance tables
└────────┬────────────┘
         │
┌────────▼────────────┐
│  Analysis Modules   │  forecasting / recommendation_engine /
│  (analysis/)        │  anomaly_detection / kpi_analysis
└────────┬────────────┘
         │
┌────────▼────────────┐
│  Streamlit Dashboard│  Interactive UI with filters, charts, tables
│  (dashboard/)       │
└─────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| Database | PostgreSQL |
| ORM / Queries | SQLAlchemy, pandas |
| Forecasting | ARIMA via pmdarima (auto parameter selection) |
| Dashboard | Streamlit + Plotly |
| Containerization | Docker + Docker Compose |
| Testing | pytest |

---

## Quick Start (Docker — recommended)

**Prerequisites:** Docker and Docker Compose installed.

```bash
git clone https://github.com/youruser/performance-tracker.git
cd performance-tracker

# Copy and configure environment variables
cp .env.example .env

# Start all services (DB + API + Dashboard)
docker-compose up --build
```

Then open **http://localhost:8501** in your browser.

On first run, the ETL loads 90 days of simulated historical data automatically.

---

## Manual Setup (without Docker)

### 1. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your PostgreSQL credentials
```

### 4. Create the database tables

```sql
CREATE TABLE products (
    id       SERIAL PRIMARY KEY,
    name     TEXT NOT NULL,
    price    NUMERIC(10,2),
    category TEXT
);

CREATE TABLE daily_performance (
    date        DATE,
    product_id  INTEGER REFERENCES products(id),
    impressions INTEGER,
    clicks      INTEGER,
    ad_spend    NUMERIC(10,2),
    units_sold  INTEGER,
    revenue     NUMERIC(10,2),
    PRIMARY KEY (date, product_id)
);
```

### 5. Start the simulated API

```bash
uvicorn simulated_api.main:app --reload --port 8000
```

### 6. Run the ETL pipeline

```bash
python data_pipeline/etl.py
```

### 7. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
performance_tracker/
├── simulated_api/
│   └── main.py                   # FastAPI app — generates daily metrics per product
├── data_pipeline/
│   └── etl.py                    # Extract from API, transform, load to PostgreSQL
├── analysis/
│   ├── forecasting.py            # ARIMA revenue forecast (auto parameter selection)
│   ├── recommendation_engine.py  # 7-day trend comparison + action suggestions
│   ├── anomaly_detection.py      # Z-score anomaly detection with hypothesis
│   ├── kpi_analysis.py           # KPI calculations
│   └── promotion_analysis.py     # Promotion impact analysis
├── dashboard/
│   └── app.py                    # Streamlit dashboard
├── tests/
│   ├── test_recommendation_engine.py
│   ├── test_anomaly_detection.py
│   └── test_forecasting.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_NAME=performance_tracker
```

When running with Docker, `DB_HOST` should be `db` (the service name).

---

## Key Design Decisions

**Why a simulated API instead of a CSV?**
Simulating a real HTTP data source makes the ETL more representative of production pipelines, where data comes from external ad platforms (Google Ads, Amazon, Meta).

**Why ARIMA over Prophet?**
Prophet is powerful but heavy. ARIMA with `auto_arima` selects optimal parameters automatically and is faster for daily revenue series with clear weekly seasonality.

**Why Z-score for anomaly detection?**
Simple, interpretable, and effective for univariate time series without requiring labeled training data. The threshold is configurable.
