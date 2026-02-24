"""
ETL Scheduler — runs the ETL pipeline automatically every day at midnight.

Usage:
    python data_pipeline/scheduler.py

The scheduler keeps running in the foreground. Press Ctrl+C to stop.
For production, run it as a background service or inside Docker.
"""
import schedule
import time
import logging
from datetime import datetime
from etl import run_etl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_etl_job():
    logger.info("Starting scheduled ETL run...")
    try:
        run_etl()
        logger.info("ETL completed successfully.")
    except Exception as e:
        logger.error(f"ETL failed: {e}")


# Run once immediately on startup to populate data
run_etl_job()

# Then schedule daily at midnight
schedule.every().day.at("00:00").do(run_etl_job)

logger.info("Scheduler running — ETL will execute daily at 00:00. Press Ctrl+C to stop.")

while True:
    schedule.run_pending()
    time.sleep(60)