"""VectorWatch India API (Person 5).

Start:  uvicorn src.api.main:app --port 8000        (after: python -m src.api.ingest)
Docs:   http://localhost:8000/docs
"""
import json
import os
import sqlite3
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query

from src.api import db, performance
from src.api.config import db_path
from src.api.schemas import (Disease, Explanation, ForecastResponse, Health, Historical,
                             ModelInfo, RiskMap, State)

DISCLAIMER = ("Annual forecasts are statistically tied with last year's value (persistence). "
              "A risk level is a rank of the forecast against the state's own history, "
              "not a prediction of an outbreak.")

app = FastAPI(
    title="VectorWatch India API",
    description="Annual dengue, malaria and chikungunya forecasts for 36 states/UTs, always shown next to "
                "persistence. Data is annual, latest year 2022. Horizons 2 and 3 are experimental.",
    version="1.0.0")


def get_db():
    path = db_path()
    if not os.path.exists(path):
        raise HTTPException(503, "Database not found. Run: python -m src.api.ingest")
    conn = db.connect(path)
    try:
        yield conn
    finally:
        conn.close()


def meta(conn):
    out = {}
    for r in conn.execute("SELECT key, value FROM meta"):
        out[r["key"]] = r["value"]
    return out


def split_notes(text):
    return [n for n in (text or "").split(";") if n]


def check_state(conn, state_id):
    if conn.execute("SELECT 1 FROM states WHERE state_id=?", (state_id,)).fetchone() is None:
        raise HTTPException(404, f"Unknown state_id '{state_id}'. See /states.")


def interval(row):
    if row["interval_80_low"] is None:
        return None
    return [row["interval_80_low"], row["interval_80_high"]]


def model_info(conn):
    r = conn.execute("SELECT name, version, summary FROM model_registry ORDER BY name LIMIT 1").fetchone()
    if r is None:
        return ModelInfo(name="annual_final", summary="statistically tied with persistence")
    return ModelInfo(name=r["name"], version=r["version"], summary=r["summary"])


@app.get("/health", response_model=Health, tags=["status"])
def health(conn: sqlite3.Connection = Depends(get_db)):
    m = meta(conn)
    latest = int(m["latest_observed_year"])
    return Health(status="ok", data_version=m["data_version"], latest_observed_year=latest,
                  data_granularity="annual", model=model_info(conn).name,
                  data_note=f"Latest available data: annual, {latest}")


@app.get("/states", response_model=list[State], tags=["data"])
def states(conn: sqlite3.Connection = Depends(get_db)):
    rows = conn.execute("SELECT state_id, state_name FROM states ORDER BY rowid").fetchall()
    return [State(state_id=r["state_id"], state_name=r["state_name"]) for r in rows]


@app.get("/historical/{state_id}", response_model=Historical, tags=["data"])
def historical(state_id: str, conn: sqlite3.Connection = Depends(get_db)):
    check_state(conn, state_id)
    name = conn.execute("SELECT state_name FROM states WHERE state_id=?", (state_id,)).fetchone()[0]
    hist = {}
    for d in Disease:
        rows = conn.execute("SELECT year, cases FROM disease_observations WHERE state_id=? AND disease=? "
                            "ORDER BY year", (state_id, d.value)).fetchall()
        hist[d.value] = [{"year": r["year"], "cases": r["cases"]} for r in rows]
    return Historical(state_id=state_id, state_name=name, data_granularity="annual",
                      latest_observed_year=int(meta(conn)["latest_observed_year"]), history=hist)


@app.get("/forecast/{state_id}/{disease}", response_model=ForecastResponse, tags=["forecast"])
def forecast(state_id: str, disease: Disease,
             include_experimental: bool = Query(False, description="Also return horizons 2 and 3."),
             conn: sqlite3.Connection = Depends(get_db)):
    check_state(conn, state_id)
    hist = conn.execute("SELECT year, cases FROM disease_observations WHERE state_id=? AND disease=? "
                        "ORDER BY year", (state_id, disease.value)).fetchall()
    sql = ("SELECT f.year, f.horizon, f.forecast, f.persistence, f.confidence, f.notes, "
           "r.risk_level, r.percentile, r.growth_vs_persistence_pct, r.reason, "
           "r.interval_80_low, r.interval_80_high "
           "FROM forecasts f JOIN risk_scores r ON r.state_id=f.state_id AND r.disease=f.disease "
           "AND r.year=f.year AND r.horizon=f.horizon "
           "WHERE f.state_id=? AND f.disease=? ")
    if not include_experimental:
        sql += "AND f.horizon=1 "
    rows = conn.execute(sql + "ORDER BY f.horizon", (state_id, disease.value)).fetchall()
    items = []
    for r in rows:
        items.append({
            "year": r["year"], "horizon": r["horizon"], "forecast": r["forecast"],
            "persistence": r["persistence"], "interval_80": interval(r),
            "confidence": r["confidence"], "notes": split_notes(r["notes"]),
            "risk_level": r["risk_level"], "percentile": r["percentile"],
            "growth_vs_persistence_pct": r["growth_vs_persistence_pct"],
            "risk_reason": split_notes(r["reason"]), "experimental": r["horizon"] > 1})
    return ForecastResponse(
        state_id=state_id, disease=disease.value,
        history=[{"year": h["year"], "cases": h["cases"]} for h in hist],
        forecasts=items, model=model_info(conn), disclaimer=DISCLAIMER)


@app.get("/risk-map", response_model=RiskMap, tags=["risk"])
def risk_map(disease: Disease,
             year: Optional[int] = Query(None, description="Forecast year. Default: next year (horizon 1)."),
             conn: sqlite3.Connection = Depends(get_db)):
    latest = int(meta(conn)["latest_observed_year"])
    if year is None:
        year = latest + 1
    horizon = year - latest
    if horizon < 1 or horizon > 3:
        raise HTTPException(404, f"No forecasts for {year}. Available: {latest + 1} to {latest + 3}.")
    rows = conn.execute(
        "SELECT s.state_id, s.state_name, f.forecast, f.persistence, f.confidence, f.notes, "
        "r.risk_level, r.percentile, r.growth_vs_persistence_pct, r.reason, "
        "r.interval_80_low, r.interval_80_high "
        "FROM states s JOIN forecasts f ON f.state_id=s.state_id "
        "JOIN risk_scores r ON r.state_id=f.state_id AND r.disease=f.disease AND r.year=f.year "
        "AND r.horizon=f.horizon WHERE f.disease=? AND f.year=? ORDER BY s.rowid",
        (disease.value, year)).fetchall()
    out = []
    for r in rows:
        out.append({"state_id": r["state_id"], "state_name": r["state_name"], "risk_level": r["risk_level"],
                    "percentile": r["percentile"], "growth_vs_persistence_pct": r["growth_vs_persistence_pct"],
                    "forecast": r["forecast"], "persistence": r["persistence"], "interval_80": interval(r),
                    "confidence": r["confidence"], "notes": split_notes(r["notes"]),
                    "risk_reason": split_notes(r["reason"])})
    return RiskMap(
        disease=disease.value, year=year, horizon=horizon, experimental=horizon > 1,
        method="Percentile of the forecast among the state's previous observed years (same disease).",
        cutoffs=json.loads(meta(conn)["risk_cutoffs"]), disclaimer=DISCLAIMER, states=out)


@app.get("/model-performance", tags=["model"])
def model_performance(tag: str = Query(performance.HEADLINE_TAG, description="Run whose per-fold rows to return."),
                      horizon: int = 1, conn: sqlite3.Connection = Depends(get_db)):
    fold_df = performance.load_fold_df(conn)
    if len(fold_df) == 0:
        raise HTTPException(404, "No evaluation results loaded.")
    summary = performance.summarize(fold_df)
    folds = fold_df[(fold_df["tag"] == tag) & (fold_df["horizon"] == horizon)].sort_values("year")
    if len(folds) == 0:
        raise HTTPException(404, f"No folds for tag '{tag}' horizon {horizon}.")
    fold_rows = folds.astype(object).where(folds.notna(), None).to_dict(orient="records")
    n_variants = len({s["tag"] for s in summary})
    return {
        "metric": "log-MAE (mean |log1p(pred) - log1p(true)|), lower is better",
        "reference": "persistence = last year's value",
        "evaluation": "Rolling origin, test years 2010-2022, trained on earlier years only; "
                      "95% bootstrap interval over folds (10,000 resamples). Interval includes 0 = tie.",
        "variants_run": n_variants,
        "multiple_comparisons_note": f"{n_variants} variants were compared; with that many, one can look "
                                     "better by chance.",
        "headline": next(s for s in summary if s["tag"] == performance.HEADLINE_TAG and s["horizon"] == 1),
        "summary": summary,
        "model_comparisons": performance.comparisons(fold_df),
        "folds_tag": tag, "folds_horizon": horizon,
        "folds": fold_rows,
    }


@app.get("/explanation/{state_id}/{disease}", response_model=Explanation, tags=["explanation"])
def explanation(state_id: str, disease: Disease, horizon: int = Query(1, ge=1, le=3),
                conn: sqlite3.Connection = Depends(get_db)):
    check_state(conn, state_id)
    try:
        from src.explainability import occlusion
    except ImportError:
        raise HTTPException(503, "Explanations need torch and models/stgnn/annual_final.pt in this environment.")
    result = occlusion.explain(state_id, disease.value, horizon)
    row = conn.execute("SELECT confidence, notes FROM forecasts WHERE state_id=? AND disease=? AND horizon=?",
                       (state_id, disease.value, horizon)).fetchone()
    if row is not None and row["confidence"] == "low":
        result["notes"] = result["notes"] + ["low_confidence_forecast"] + split_notes(row["notes"])
    return result


@app.get("/", include_in_schema=False)
def root():
    return {"name": "VectorWatch India API", "docs": "/docs", "health": "/health"}


