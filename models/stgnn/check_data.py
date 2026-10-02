import pandas as pd, numpy as np

PATH = "data/processed/pan_india_state_enriched_tensor.csv"
df = pd.read_csv(PATH, parse_dates=["start_date"])

print("shape:", df.shape)
print("columns:", df.columns.tolist())
print("date range:", df.start_date.min().date(), "->", df.start_date.max().date())
print("n states:", df.state_id.nunique())
print("duplicate (state, date) rows:", df.duplicated(["state_id", "start_date"]).sum())

# full grid would be 36 states x 288 months
full = df.state_id.nunique() * 288
print(f"rows: {len(df)} | full grid: {full} | missing rows: {full - len(df)}")

print("\nrows per state (lowest 10):")
print(df.groupby("state_id").size().sort_values().head(10))

print("\nNaN fraction per column:")
print(df.isna().mean().round(3))

dis = ["dengue_cases", "malaria_cases", "chikungunya_cases"]
print("\nobserved months per state per disease:")
print(df.groupby("state_id")[dis].count().to_string())

# save the fixed node order for every later step
states = sorted(df.state_id.unique())
with open("configs/states.txt", "w") as f:
    f.write("\n".join(states))
print("\nsaved", len(states), "states to configs/states.txt")
print(states)