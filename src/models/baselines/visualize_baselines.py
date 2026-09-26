"""
visualize_baselines.py

Generates a comparative visualization of baseline model performance (MAE)
for top high-burden states. Outputs a PNG artifact for the project documentation.
Owner: Person 1 (Disease Data & Forecasting Engineer)
"""

import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

METRICS_PATH = Path("data/processed/baseline_model_metrics.csv")
OUTPUT_PLOT_PATH = Path("data/processed/baseline_mae_comparison.png")

# Define high-burden states to feature in the benchmark plot
HIGH_BURDEN_STATES = [
    "Maharashtra", 
    "West Bengal", 
    "Delhi", 
    "Karnataka", 
    "Andhra Pradesh"
]

def generate_visualization():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Metrics file missing at {METRICS_PATH}. Run train_baselines.py first.")

    logger.info("Loading baseline metrics...")
    df = pd.read_csv(METRICS_PATH)
    
    # Filter to high burden states and focus on Dengue for the primary visualization
    plot_df = df[(df["state_name"].isin(HIGH_BURDEN_STATES)) & (df["disease"] == "dengue_cases")].copy()
    
    if plot_df.empty:
        logger.warning("No matching data found for the selected states and disease. Check state names.")
        return

    # Set up the plotting style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Create a grouped bar chart
    ax = sns.barplot(
        data=plot_df,
        x="state_name",
        y="mae",
        hue="model",
        hue_order=["Seasonal_Naive", "Gradient_Boosting", "Gradient_Boosting_Weather"],
        palette={
            "Seasonal_Naive": "#e74c3c",
            "Gradient_Boosting": "#3498db",
            "Gradient_Boosting_Weather": "#2ecc71",
        }
    )
    
    # Formatting the chart
    plt.title("Baseline Performance Benchmark: Dengue Cases (MAE)", fontsize=14, pad=15)
    plt.xlabel("State", fontsize=12)
    plt.ylabel("Mean Absolute Error (MAE)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.legend(title="Model", title_fontsize="11", fontsize="10")
    
    # Auto-adjust layout and save
    plt.tight_layout()
    
    OUTPUT_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
    logger.info(f"Visualization successfully saved to: {OUTPUT_PLOT_PATH}")

if __name__ == "__main__":
    generate_visualization()