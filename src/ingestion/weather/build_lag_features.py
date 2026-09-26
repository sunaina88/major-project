"""
build_lag_features.py

Adds epidemiological lag features (delayed weather effects) to the enriched tensor.
Owner: Person 1 (Disease Data & Forecasting Engineer)
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TENSOR_PATH = Path("data/processed/pan_india_state_enriched_tensor.csv")

def add_lag_features():
    if not TENSOR_PATH.exists():
        raise FileNotFoundError(f"File not found: {TENSOR_PATH}")
        
    df = pd.read_csv(TENSOR_PATH)
    df["start_date"] = pd.to_datetime(df["start_date"])
    
    # Sort strictly by state and time to ensure rolling/shifting is accurate across boundaries
    df = df.sort_values(["state_name", "start_date"]).reset_index(drop=True)
    
    logger.info("Generating 1-month and 2-month lag features for weather variables...")
    
    # Generate temporal lag features per state
    df["temp_lag_1"] = df.groupby("state_name")["temp_mean_c"].shift(1)
    df["temp_lag_2"] = df.groupby("state_name")["temp_mean_c"].shift(2)
    
    df["rainfall_lag_1"] = df.groupby("state_name")["rainfall_mm"].shift(1)
    df["rainfall_lag_2"] = df.groupby("state_name")["rainfall_mm"].shift(2)
    
    logger.info("Generating 3-month rolling cumulative rainfall...")
    # 3-month rolling sum of rainfall (breeding season accumulation)
    df["rainfall_rolling_3m"] = (
        df.groupby("state_name")["rainfall_mm"]
        .rolling(window=3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    
    # Reformat date and save back out
    df["start_date"] = df["start_date"].dt.strftime("%Y-%m-%d")
    df.to_csv(TENSOR_PATH, index=False)
    
    logger.info(f"Feature engineering complete. Tensor overwritten at: {TENSOR_PATH}")
    
    print("\n--- Lag Features Sample (Showing recent rows) ---")
    columns_to_show = ["start_date", "state_name", "rainfall_mm", "rainfall_lag_1", "rainfall_rolling_3m"]
    print(df[columns_to_show].dropna().head(7).to_string(index=False))

if __name__ == "__main__":
    add_lag_features()