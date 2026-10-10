"""Load observations, forecasts and evaluation files into SQLite. Idempotent.

Run from anywhere:   python -m src.api.ingest
Database file:       env VW_DB_PATH (default data/vectorwatch.db)
Running it twice drops and rebuilds every table, so the result is identical.
"""
import datetime
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from src.api import db
from src.api.config import DISEASES, REPO_ROOT, db_path, in_repo_root, state_name
from src.risk import engine, uncertainty

PROC = REPO_ROOT / "data" / "processed"
TENSOR = PROC / "pan_india_state_enriched_tensor_real.csv"


def clean(df, cols):
    """DataFrame -> list of tuples with NaN replaced by None."""
    out = []
    for i in range(len(df)):
        row = []
        for c in cols:
            v = df.iloc[i][c]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                row.append(None)
            elif isinstance(v, (np.integer,)):
                row.append(int(v))
            elif isinstance(v, (np.floating,)):
                row.append(float(v))
            else:
                row.append(v)
        out.append(tuple(row))
    return out


def insert(conn, table, cols, rows):
    marks = ",".join(["?"] * len(cols))
    names = ",".join(f'"{c}"' for c in cols)
    conn.executemany(f"INSERT INTO {table} ({names}) VALUES ({marks})", rows)


def load_observations(conn):
    """Annual observations through Person 2's load_annual (zeros-as-missing heuristic included)."""
    with in_repo_root():
        sys.path.insert(0, str(REPO_ROOT / "models" / "stgnn"))
        try:
            from annual_data import load_annual
            states, years, Y, _ = load_annual(str(TENSOR))
        finally:
            sys.path.pop(0)
    insert(conn, "states", ["state_id", "state_name"], [(s, state_name(s)) for s in states])
    rows = []
    for t in range(len(years)):
        for j in range(len(states)):
            for k in range(3):
                v = Y[t, j, k]
                if not np.isnan(v):
                    rows.append((states[j], DISEASES[k], int(years[t]), float(v)))
    insert(conn, "disease_observations", ["state_id", "disease", "year", "cases"], rows)
    return int(years[-1])


def load_forecasts(conn):
    df = pd.read_csv(PROC / "stgnn_forecast_final.csv")
    df["notes"] = df["notes"].fillna("")
    cols = ["state_id", "disease", "year", "horizon", "forecast", "persistence",
            "confidence", "notes", "n_years_observed"]
    insert(conn, "forecasts", cols, clean(df, cols))
    return df


def load_fold_files(conn):
    n_files = 0
    fold_cols = ["tag", "horizon", "year", "n", "mae_model", "mae_last", "mae_avg2",
                 "log_model", "log_last", "log_avg2"]
    pair_cols = ["tag", "horizon", "year", "state_id", "disease", "true", "model", "persistence"]
    for path in sorted(glob.glob(str(PROC / "stgnn_annual_*_h*.csv"))):
        name = os.path.basename(path)
        m = re.match(r"stgnn_annual_(.+)_h(\d+)(_pairs)?\.csv$", name)
        if not m:
            continue
        tag, horizon, is_pairs = m.group(1), int(m.group(2)), m.group(3) is not None
        df = pd.read_csv(path)
        df.insert(0, "horizon", horizon)
        df.insert(0, "tag", tag)
        if is_pairs:
            insert(conn, "pair_predictions", pair_cols, clean(df, pair_cols))
        else:
            insert(conn, "fold_results", fold_cols, clean(df, fold_cols))
        n_files += 1
    return n_files


def load_anomalies(conn):
    path = PROC / "anomaly_annual.csv"
    if not path.exists():
        return 0   # optional input from Person 3
    df = pd.read_csv(path)
    cols = ["state_id", "disease", "year", "observed", "expected", "zscore", "pctile"]
    insert(conn, "anomaly_annual", cols, clean(df, cols))
    return len(df)


def load_registry(conn):
    path = REPO_ROOT / "models" / "registry.json"
    if not path.exists():
        return 0
    reg = json.load(open(path))
    n = 0
    for m in reg["models"]:
        conn.execute(
            "INSERT INTO model_registry VALUES (?,?,?,?,?,?,?,?)",
            (m["name"], m["version"], m["file"], m["training_years"], m["git_commit"],
             m["registered_at"], m["summary"], json.dumps(m)))
        n += 1
    return n


def build_risk_scores(conn, forecasts, cfg):
    """Risk level + interval for every forecast row. Uses only years up to the latest observed year."""
    obs = pd.read_sql_query("SELECT state_id, disease, year, cases FROM disease_observations", conn)
    prior = {}
    for key, g in obs.groupby(["state_id", "disease"]):
        prior[key] = g.sort_values("year")["cases"].values

    ucfg = cfg["uncertainty"]
    pairs = pd.read_sql_query(
        'SELECT disease, "true", model FROM pair_predictions WHERE tag=? AND horizon=?',
        conn, params=(ucfg["tag"], ucfg["horizon"]))
    interval_model = None
    if len(pairs) > 0:
        interval_model = uncertainty.fit_interval_model(pairs, ucfg["nominal"], ucfg["by_disease"])

    rows = []
    for i in range(len(forecasts)):
        r = forecasts.iloc[i]
        notes = [n for n in str(r["notes"]).split(";") if n]
        past = prior.get((r["state_id"], r["disease"]), np.array([]))
        pers = None if pd.isna(r["persistence"]) else float(r["persistence"])
        res = engine.assess(float(r["forecast"]), pers, r["confidence"], notes, past, cfg)
        lo = hi = None
        if interval_model is not None and int(r["horizon"]) == ucfg["horizon"]:
            lo, hi = uncertainty.interval_for(interval_model, r["disease"], float(r["forecast"]))
        rows.append((r["state_id"], r["disease"], int(r["year"]), int(r["horizon"]),
                     res["risk_level"], res["percentile"], res["growth_vs_persistence_pct"],
                     res["n_prior_years"], res["reason"], lo, hi))
    cols = ["state_id", "disease", "year", "horizon", "risk_level", "percentile",
            "growth_vs_persistence_pct", "n_prior_years", "reason", "interval_80_low", "interval_80_high"]
    insert(conn, "risk_scores", cols, rows)
    return len(rows)


def data_version():
    """Short hash of the input files, so /health shows when the data changed."""
    h = hashlib.sha256()
    for p in [TENSOR, PROC / "stgnn_forecast_final.csv"]:
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def main():
    path = db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfg = engine.load_config()
    conn = db.connect(path)
    try:
        with conn:   # one transaction: either everything loads or nothing changes
            db.recreate(conn)
            latest = load_observations(conn)
            forecasts = load_forecasts(conn)
            n_files = load_fold_files(conn)
            n_anom = load_anomalies(conn)
            n_reg = load_registry(conn)
            n_risk = build_risk_scores(conn, forecasts, cfg)
            meta = {"latest_observed_year": str(latest), "data_version": data_version(),
                    "data_granularity": "annual",
                    "ingested_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "risk_cutoffs": json.dumps(cfg["risk"]["cutoffs"])}
            for k, v in meta.items():
                conn.execute("INSERT INTO meta VALUES (?,?)", (k, v))
    finally:
        conn.close()
    print(f"database {path}: observations up to {latest}, {len(forecasts)} forecasts, "
          f"{n_files} evaluation files, {n_anom} anomaly rows, {n_reg} registry entries, {n_risk} risk rows")


if __name__ == "__main__":
    main()
