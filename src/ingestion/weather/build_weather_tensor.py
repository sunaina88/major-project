"""
build_weather_tensor.py (Offline Climatology Version)

Generates historical climate data (Temperature and Precipitation) for 36 Indian States/UTs 
using deterministic seasonal climatology models to bypass university network API blocks.
Merges features into the master spatiotemporal tensor.
Owner: Person 1 (Disease Data & Forecasting Engineer)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DISEASE_TENSOR_PATH = Path("data/processed/pan_india_state_spatiotemporal_tensor.csv")
ENRICHED_TENSOR_PATH = Path("data/processed/pan_india_state_enriched_tensor.csv")

# Regional climate modifiers: (base_annual_rain_mm, base_annual_temp_c, rain_variance)
STATE_CLIMATE_PROFILES = {
    "andaman_and_nicobar_islands": (2900, 28.0, 1.2),
    "andhra_pradesh": (900, 29.5, 1.0),
    "arunachal_pradesh": (3000, 20.0, 1.3),
    "assam": (2800, 24.0, 1.3),
    "bihar": (1200, 26.0, 1.1),
    "chandigarh": (1000, 24.5, 1.0),
    "chhattisgarh": (1300, 27.0, 1.1),
    "delhi": (700, 25.5, 0.9),
    "goa": (3000, 27.5, 1.3),
    "gujarat": (800, 28.5, 0.9),
    "haryana": (600, 25.0, 0.8),
    "himachal_pradesh": (1100, 15.0, 1.0),
    "jammu_and_kashmir": (900, 13.0, 1.0),
    "jharkhand": (1300, 26.5, 1.1),
    "karnataka": (1200, 26.5, 1.0),
    "kerala": (2900, 27.5, 1.2),
    "ladakh": (100, 5.0, 0.5),
    "lakshadweep": (1600, 28.0, 1.1),
    "madhya_pradesh": (1100, 27.0, 1.0),
    "maharashtra": (1300, 27.5, 1.1),
    "manipur": (1500, 21.0, 1.1),
    "meghalaya": (4000, 20.5, 1.4),
    "mizoram": (2500, 21.5, 1.2),
    "nagaland": (2000, 21.0, 1.1),
    "odisha": (1450, 27.5, 1.2),
    "puducherry": (1300, 29.0, 1.1),
    "punjab": (650, 24.5, 0.9),
    "rajasthan": (500, 28.5, 0.7),
    "sikkim": (2700, 16.0, 1.2),
    "tamil_nadu": (950, 29.0, 1.0),
    "telangana": (900, 28.5, 1.0),
    "the_dadra_and_nagar_haveli_and_daman_and_diu": (2000, 27.5, 1.1),
    "tripura": (2200, 24.5, 1.2),
    "uttar_pradesh": (950, 26.0, 1.0),
    "uttarakhand": (1500, 18.0, 1.1),
    "west_bengal": (1700, 26.5, 1.2)
}

# Monthly distribution (1=Jan, 12=Dec)
# Temperature peaks in May-June, Rainfall peaks in July-August
TEMP_CURVE = {1: -6, 2: -3, 3: 2, 4: 6, 5: 8, 6: 7, 7: 4, 8: 3, 9: 2, 10: -1, 11: -4, 12: -6}
RAIN_CURVE = {1: 0.01, 2: 0.02, 3: 0.02, 4: 0.03, 5: 0.05, 6: 0.15, 7: 0.25, 8: 0.25, 9: 0.12, 10: 0.06, 11: 0.03, 12: 0.01}

def generate_local_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    np.random.seed(42) # Ensure reproducibility
    
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["month"] = df["start_date"].dt.month
    df["year"] = df["start_date"].dt.year
    
    temp_list = []
    rain_list = []
    
    for _, row in df.iterrows():
        state = row["state_id"]
        month = row["month"]
        
        # Get baseline profile, default to national average if missing
        base_rain, base_temp, variance = STATE_CLIMATE_PROFILES.get(state, (1000, 25.0, 1.0))
        
        # Calculate temperature with seasonal curve + slight random inter-annual noise
        temp = base_temp + TEMP_CURVE[month] + np.random.normal(0, 0.5)
        
        # Calculate rainfall with monsoon curve + variance multiplier
        rain = (base_rain * RAIN_CURVE[month]) * np.random.normal(1.0, 0.15 * variance)
        
        temp_list.append(round(max(temp, -10.0), 2))
        rain_list.append(round(max(rain, 0.0), 2))
        
    df["temp_mean_c"] = temp_list
    df["rainfall_mm"] = rain_list
    
    # Drop temporary columns
    df.drop(columns=["month", "year"], inplace=True)
    return df

def run_weather_ingestion():
    if not DISEASE_TENSOR_PATH.exists():
        raise FileNotFoundError(f"Missing master tensor at {DISEASE_TENSOR_PATH}")

    logger.info("Loading master disease tensor...")
    disease_tensor = pd.read_csv(DISEASE_TENSOR_PATH)
    
    logger.info("Generating local climatological weather features (Offline Mode)...")
    enriched_tensor = generate_local_weather(disease_tensor)
    
    # Enforce date formatting
    enriched_tensor["start_date"] = pd.to_datetime(enriched_tensor["start_date"]).dt.strftime("%Y-%m-%d")
    
    ENRICHED_TENSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched_tensor.to_csv(ENRICHED_TENSOR_PATH, index=False)
    
    logger.info(f"Enriched tensor saved to: {ENRICHED_TENSOR_PATH}")
    print("\n--- Enriched Tensor Sample ---")
    print(enriched_tensor[["start_date", "state_name", "temp_mean_c", "rainfall_mm", "dengue_cases"]].head(10).to_string(index=False))

if __name__ == "__main__":
    run_weather_ingestion()