"""Build frontend/mocks/*.json from current processed outputs. Run from repo root."""
import sys, json, glob, os
import numpy as np, pandas as pd
sys.path.insert(0, "models/stgnn")
from annual_data import load_annual
from src.api.performance import summarize

OUT = "frontend/mocks"
os.makedirs(OUT, exist_ok=True)
def dump(name, obj):
    json.dump(obj, open(f"{OUT}/{name}.json", "w"), indent=1, default=str)
def clean(df):
    return df.astype(object).where(df.notna(), None).to_dict("records")

dump("forecast", clean(pd.read_csv("data/processed/stgnn_forecast_final.csv")))

states, years, Y, W = load_annual("data/processed/pan_india_state_enriched_tensor.csv")
Y = np.asarray(Y, dtype=float)
dis = ["dengue", "malaria", "chikungunya"]
hist = [{"state_id": s, "disease": d, "year": int(y), "cases": float(Y[k, i, j])}
        for i, s in enumerate(states) for j, d in enumerate(dis)
        for k, y in enumerate(years) if not np.isnan(Y[k, i, j])]
dump("history", hist)

rf = os.getenv("REPLAY_FILE") or "data/processed/stgnn_annual_geo_w_h1_pairs.csv"
if not os.path.exists(rf):
    raise FileNotFoundError(f"Replay file missing: {rf}. Run models/stgnn/run_all.sh first.")
print("replay file:", rf)
dump("replay", clean(pd.read_csv(rf)))

frames = []
for path in sorted(glob.glob("data/processed/stgnn_annual_*_h*.csv")):
    if path.endswith("_pairs.csv"):
        continue
    stem = os.path.basename(path)[len("stgnn_annual_"):-len(".csv")]
    tag, horizon = stem.rsplit("_h", 1)
    frame = pd.read_csv(path)
    frame.insert(0, "horizon", int(horizon))
    frame.insert(0, "tag", tag)
    frames.append(frame)
if not frames:
    raise FileNotFoundError("No ST-GNN fold results found. Run models/stgnn/run_all.sh first.")
summary = summarize(pd.concat(frames, ignore_index=True))
headline = next(r for r in summary if r["tag"] == "geo_w" and r["horizon"] == 1)
performance = [{"variant": "Persistence", "log_mae": headline["log_mae_persistence"],
                "diff": None, "ci": None}]
performance.extend(
    {"variant": r["variant"], "log_mae": r["log_mae_model"],
     "diff": r["diff_vs_persistence"], "ci": r["ci_95"]}
    for r in summary
)
dump("model_performance", performance)
