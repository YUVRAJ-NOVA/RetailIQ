"""
RetailIQ — Data Ingestion Script
Loads all 9 CSV files into PostgreSQL raw schema
"""

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import logging
from pathlib import Path

# Logging setup 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'retailiq')
DB_USER = os.getenv('DB_USER', 'retailiq_user')
DB_PASS = os.getenv('DB_PASSWORD', 'retailiq123')

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# File to table mapping 
FILE_TABLE_MAP = {
    "olist_customers_dataset.csv":            "customers",
    "olist_orders_dataset.csv":               "orders",
    "olist_order_items_dataset.csv":          "order_items",
    "olist_products_dataset.csv":             "products",
    "olist_order_reviews_dataset.csv":        "reviews",
    "olist_order_payments_dataset.csv":       "payments",
    "olist_sellers_dataset.csv":              "sellers",
    "olist_geolocation_dataset.csv":          "geolocation",
    "product_category_name_translation.csv":  "category_translation",
}

# Column renaming to match schema 
COLUMN_RENAMES = {
    "orders": {
        "order_purchase_timestamp":        "order_purchase_ts",
        "order_approved_at":               "order_approved_ts",
        "order_delivered_carrier_date":    "order_carrier_ts",
        "order_delivered_customer_date":   "order_delivered_ts",
        "order_estimated_delivery_date":   "order_estimated_ts",
    },
    "reviews": {
        "review_creation_date":            "review_creation_date",
        "review_answer_timestamp":         "review_answer_timestamp",
    }
}


def load_table(filepath: Path, table_name: str, engine) -> bool:
    """Load a single CSV into PostgreSQL."""
    try:
        logger.info(f"Loading {filepath.name}...")

        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f"  → {len(df):,} rows, {len(df.columns)} columns")

        # Rename columns if needed
        if table_name in COLUMN_RENAMES:
            df = df.rename(columns=COLUMN_RENAMES[table_name])

        # Load to PostgreSQL
        df.to_sql(
            name=table_name,
            con=engine,
            schema='raw',
            if_exists='replace',   
            index=False,
            method='multi',        
            chunksize=500
        )

        logger.info(f"   {table_name} loaded successfully")
        return True

    except Exception as e:
        logger.error(f"   Failed to load {table_name}: {e}")
        return False


def main():
    data_dir = Path("data/raw")
    
    # Check karo ki data folder hai
    if not data_dir.exists():
        logger.error("data/raw folder nahi mila! CSVs wahan rakh.")
        return
    
    # Database connection
    logger.info("Connecting to PostgreSQL...")
    engine = create_engine(DATABASE_URL)
    
    # Test connection
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(" Database connected!")
    except Exception as e:
        logger.error(f" Database connection failed: {e}")
        return
    
    # Load all tables
    logger.info("\n" + "="*50)
    logger.info("STARTING DATA LOAD")
    logger.info("="*50 + "\n")
    
    success_count = 0
    fail_count = 0
    
    for filename, table_name in FILE_TABLE_MAP.items():
        filepath = data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"    File not found: {filename} — skipping")
            fail_count += 1
            continue
        
        if load_table(filepath, table_name, engine):
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info(f"LOAD COMPLETE —  {success_count} tables |  {fail_count} failed")
    logger.info("="*50)


if __name__ == "__main__":
    main()