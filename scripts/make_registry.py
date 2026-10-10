"""Write models/registry.json with ONE entry: annual_final.pt.

The files in models/stgnn/archive_monthly/ are invalid and are never registered.
Run from the repository root:  python scripts/make_registry.py [--git-commit SHA]
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.performance import summarize  # noqa: E402


def git_commit(given):
    if given:
        return given
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def load_folds():
    frames = []
    for p in sorted((ROOT / "data" / "processed").glob("stgnn_annual_*_h*.csv")):
        if p.name.endswith("_pairs.csv"):
            continue
        stem = p.stem.replace("stgnn_annual_", "")
        tag, h = stem.rsplit("_h", 1)
        df = pd.read_csv(p)
        df.insert(0, "horizon", int(h))
        df.insert(0, "tag", tag)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--git-commit", default=None)
    a = ap.parse_args()

    ckpt = ROOT / "models" / "stgnn" / "annual_final.pt"
    ck = torch.load(ckpt, weights_only=False)
    args = ck["args"]
    years = ck["years"]
    summary_rows = summarize(load_folds())
    headline = [r for r in summary_rows if r["tag"] == "geo_w" and r["horizon"] == 1][0]

    entry = {
        "name": "annual_final",
        "version": "1.0.0",
        "file": "models/stgnn/annual_final.pt",
        "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "training_years": f"{years[0]}-{years[-1]}",
        "horizons_years_ahead": sorted(int(h) for h in ck["models"].keys()),
        "default_horizon": 1,
        "architecture": {"kind": args["kind"], "graph": args["graph"], "hidden": args["hidden"],
                         "cap": args["cap"], "input_window_years": args["K"], "seeds": args["seeds"],
                         "epochs": args["epochs"], "base": "last year's value (residual head)"},
        "features": ["log1p annual dengue cases (normalised)", "log1p annual malaria cases (normalised)",
                     "log1p annual chikungunya cases (normalised)",
                     "observed flag dengue", "observed flag malaria", "observed flag chikungunya",
                     "annual mean temperature (NASA POWER, normalised)",
                     "annual total rainfall (NASA POWER, normalised)"],
        "metrics": {
            "metric": "log-MAE, rolling origin, test years 2010-2022 (13 folds), 95% bootstrap CI over folds",
            "headline_horizon_1": headline,
            "variants": summary_rows,
            "variants_run": len({r["tag"] for r in summary_rows}),
        },
        "summary": "Annual forecasts are statistically tied with persistence (last year's value). "
                   "No evidence that the graph or weather helps. Not an outbreak or early-warning model.",
        "git_commit": git_commit(a.git_commit),
        "registered_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "not_registered": "models/stgnn/archive_monthly/ (monthly models are invalid)",
    }
    out = ROOT / "models" / "registry.json"
    with open(out, "w") as f:
        json.dump({"models": [entry]}, f, indent=2)
        f.write("\n")
    print("wrote", out, "| headline:", headline["log_mae_model"], "vs persistence", headline["log_mae_persistence"])


if __name__ == "__main__":
    main()
