"""Compare newest year with earlier years. Usage: python src/monitoring/drift.py"""
import sys, json, argparse
import numpy as np
sys.path.insert(0, "models/stgnn")
from annual_data import load_annual

def shift(ref, new):
    ref, new = ref[~np.isnan(ref)], new[~np.isnan(new)]
    if len(ref) == 0 or len(new) == 0: return None
    return float((np.median(new) - np.median(ref)) / (np.std(ref) + 1e-9))

def main(data, out, cfg):
    th = json.load(open(cfg))
    states, years, Y, W = load_annual(data)
    Y, W = np.asarray(Y, float), np.asarray(W, float)
    ref = np.arange(len(years)) < len(years) - 1
    rep = {"newest_year": int(years[-1]), "diseases": {}, "weather": {}, "flags": []}
    for j, d in enumerate(["dengue", "malaria", "chikungunya"]):
        mr, mn = float(np.isnan(Y[ref, :, j]).mean()), float(np.isnan(Y[-1, :, j]).mean())
        s = shift(np.log1p(Y[ref, :, j]), np.log1p(Y[-1, :, j]))
        rep["diseases"][d] = {"missing_ref": mr, "missing_new": mn, "log_median_shift_sd": s}
        if mn - mr > th["missing_increase"]: rep["flags"].append(f"{d}: missingness up")
        if s is not None and abs(s) > th["median_shift_sd"]: rep["flags"].append(f"{d}: log-count shift")
    for k, n in enumerate(["temperature", "rainfall"]):
        s = shift(W[ref, :, k], W[-1, :, k]); rep["weather"][n] = {"median_shift_sd": s}
        if s is not None and abs(s) > th["weather_shift_sd"]: rep["flags"].append(f"{n}: shift")
    rep["flagged"] = bool(rep["flags"])
    json.dump(rep, open(out, "w"), indent=1); print(json.dumps(rep, indent=1))

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--data", default="data/processed/pan_india_state_enriched_tensor_real.csv")
    a.add_argument("--out", default="frontend/mocks/drift_report.json")
    a.add_argument("--config", default="config/drift_thresholds.json")
    x = a.parse_args(); main(x.data, x.out, x.config)
