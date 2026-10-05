"""Build frontend/mocks/*.json from real Person 2 outputs. Run from repo root."""
import sys, json, glob, os
import numpy as np, pandas as pd
sys.path.insert(0, "models/stgnn")
from annual_data import load_annual

OUT = "frontend/mocks"
os.makedirs(OUT, exist_ok=True)
def dump(name, obj):
    json.dump(obj, open(f"{OUT}/{name}.json", "w"), indent=1, default=str)
def clean(df):
    return df.astype(object).where(df.notna(), None).to_dict("records")

dump("forecast", clean(pd.read_csv("data/processed/stgnn_forecast_final.csv")))

states, years, Y, W = load_annual("data/processed/pan_india_state_enriched_tensor_real.csv")
Y = np.asarray(Y, dtype=float)
dis = ["dengue", "malaria", "chikungunya"]
hist = [{"state_id": s, "disease": d, "year": int(y), "cases": float(Y[k, i, j])}
        for i, s in enumerate(states) for j, d in enumerate(dis)
        for k, y in enumerate(years) if not np.isnan(Y[k, i, j])]
dump("history", hist)

rf = os.getenv("REPLAY_FILE") or sorted(glob.glob("data/processed/stgnn_annual_*_h1_pairs.csv"))[0]
print("replay file:", rf)
dump("replay", clean(pd.read_csv(rf)))

dump("model_performance", [
 {"variant": "Persistence", "log_mae": 0.846, "diff": None, "ci": None},
 {"variant": "GCN, land-border graph, real weather", "log_mae": 0.828, "diff": -0.018, "ci": [-0.060, 0.026]},
 {"variant": "Without weather", "log_mae": 0.852, "diff": 0.006, "ci": [-0.014, 0.029]},
 {"variant": "No graph", "log_mae": 0.836, "diff": -0.009, "ci": [-0.036, 0.014]},
 {"variant": "kNN graph", "log_mae": 0.827, "diff": None, "ci": "includes 0"},
 {"variant": "Distance graph", "log_mae": 0.824, "diff": None, "ci": "includes 0"},
 {"variant": "Hybrid graph", "log_mae": 0.828, "diff": None, "ci": "includes 0"},
 {"variant": "GAT", "log_mae": 0.859, "diff": 0.031, "ci": [0.006, 0.058], "note": "worse than GCN"},
 {"variant": "Separate model per disease", "log_mae": 0.891, "diff": 0.063, "ci": [0.0, 0.131], "note": "worse than shared"},
 {"variant": "Horizon 2 (persistence 1.093)", "log_mae": 1.127, "diff": None, "ci": None},
 {"variant": "Horizon 3 (persistence 1.240)", "log_mae": 1.259, "diff": None, "ci": None}])
