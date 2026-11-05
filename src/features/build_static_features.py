import os
import numpy as np
import pandas as pd

states = [s.strip() for s in open("configs/states.txt") if s.strip()]
CEN = "data/raw/static/census_2011_state.csv"
if not os.path.exists(CEN):
    pd.DataFrame({"state_id": states, "population_2011": np.nan, "urban_population_2011": np.nan}).to_csv(CEN, index=False)
    raise SystemExit(f"Created {CEN}. Fill it from the Census of India 2011 and rerun. "
                     "Use post-reorganisation figures (sum of districts) for Telangana, Andhra Pradesh, "
                     "Jammu and Kashmir, Ladakh and Dadra and Nagar Haveli and Daman and Diu.")

df = pd.DataFrame({"state_id": states})
df = df.merge(pd.read_csv("data/processed/state_area_km2.csv"), on="state_id", how="left")
df = df.merge(pd.read_csv(CEN), on="state_id", how="left")
if os.path.exists("data/processed/state_elevation_m.csv"):
    df = df.merge(pd.read_csv("data/processed/state_elevation_m.csv"), on="state_id", how="left")
else:
    df["mean_elevation_m"] = np.nan
df = df.rename(columns={"population_2011": "population"})
df["population_year"] = 2011
df["population_density_per_km2"] = (df.population / df.area_km2).round(1)
df["urban_fraction"] = (df.urban_population_2011 / df.population).round(4)
df = df.drop(columns=["urban_population_2011"])

problems = []
miss = df[df.population.isna()].state_id.tolist()
if miss:
    problems.append(f"missing population for {miss}")
tot = df.population.sum()
if not 1.19e9 < tot < 1.23e9:
    problems.append(f"population total {tot:,.0f} is outside 1.19 to 1.23 billion")
bad = df[(df.urban_fraction <= 0) | (df.urban_fraction > 1)].state_id.tolist()
if bad:
    problems.append(f"urban_fraction outside (0, 1] for {bad}")
df.to_csv("data/processed/static_state_features.csv", index=False)
print(df.round(1).to_string(index=False))
print("PROBLEMS:" if problems else "no problems found", *problems, sep="\n  ")
