"""SQLite schema. ingest.py drops and recreates every table, so running it twice gives the same database."""
import sqlite3

from src.api.config import db_path

TABLES = ["meta", "states", "disease_observations", "forecasts", "fold_results",
          "pair_predictions", "anomaly_annual", "risk_scores", "model_registry"]

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE states (state_id TEXT PRIMARY KEY, state_name TEXT NOT NULL);

CREATE TABLE disease_observations (
    state_id TEXT, disease TEXT, year INTEGER, cases REAL,
    PRIMARY KEY (state_id, disease, year));

CREATE TABLE forecasts (
    state_id TEXT, disease TEXT, year INTEGER, horizon INTEGER,
    forecast INTEGER, persistence REAL, confidence TEXT, notes TEXT, n_years_observed INTEGER,
    PRIMARY KEY (state_id, disease, year, horizon));

CREATE TABLE fold_results (
    tag TEXT, horizon INTEGER, year INTEGER, n INTEGER,
    mae_model REAL, mae_last REAL, mae_avg2 REAL,
    log_model REAL, log_last REAL, log_avg2 REAL,
    PRIMARY KEY (tag, horizon, year));

CREATE TABLE pair_predictions (
    tag TEXT, horizon INTEGER, year INTEGER, state_id TEXT, disease TEXT,
    "true" REAL, model REAL, persistence REAL,
    PRIMARY KEY (tag, horizon, year, state_id, disease));

CREATE TABLE anomaly_annual (
    state_id TEXT, disease TEXT, year INTEGER,
    observed REAL, expected REAL, zscore REAL, pctile REAL,
    PRIMARY KEY (state_id, disease, year));

CREATE TABLE risk_scores (
    state_id TEXT, disease TEXT, year INTEGER, horizon INTEGER,
    risk_level TEXT, percentile REAL, growth_vs_persistence_pct REAL,
    n_prior_years INTEGER, reason TEXT,
    interval_80_low INTEGER, interval_80_high INTEGER,
    PRIMARY KEY (state_id, disease, year, horizon));

CREATE TABLE model_registry (
    name TEXT PRIMARY KEY, version TEXT, file TEXT, trained_years TEXT,
    git_commit TEXT, registered_at TEXT, summary TEXT, details_json TEXT);
"""


def recreate(conn):
    for t in TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {t}")
    conn.executescript(SCHEMA)


def connect(path=None):
    conn = sqlite3.connect(path or db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn
