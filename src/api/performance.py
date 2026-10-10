"""Summary of the rolling-origin fold files, with 95% bootstrap intervals over folds.

Uses Person 3's bootstrap code (src/evaluation/metrics.py) so numbers match handoff/README.md section 4.
"""
import pandas as pd

from src.evaluation.metrics import summarize_vs_reference

HEADLINE_TAG = "geo_w"

VARIANTS = {
    "geo_w": "GCN, land-border graph, real weather",
    "geo_now": "Without weather",
    "identity": "No graph",
    "knn": "kNN graph",
    "dist": "Distance graph",
    "hybrid": "Hybrid graph",
    "gat": "GAT",
    "single": "Separate model per disease",
    "K2": "Input window of 2 years",
    "K5": "Input window of 5 years",
    "horizons": "Horizon run (1 / 2 / 3 years ahead)",
}


def variant_label(tag, horizon):
    if tag == "horizons":
        return f"{horizon} year(s) ahead"
    return VARIANTS.get(tag, tag)


def summarize(fold_df):
    """fold_df: rows of fold_results (columns tag, horizon, year, log_model, log_last, ...)."""
    rows = []
    groups = fold_df.groupby(["tag", "horizon"])
    for (tag, horizon), g in groups:
        s = summarize_vs_reference(g["log_model"].values, g["log_last"].values)
        better = 0
        for v in (g["log_model"].values - g["log_last"].values):
            if v < 0:
                better += 1
        rows.append({
            "tag": tag, "horizon": int(horizon), "variant": variant_label(tag, int(horizon)),
            "n_folds": int(len(g)),
            "log_mae_model": round(float(g["log_model"].mean()), 3),
            "log_mae_persistence": round(float(g["log_last"].mean()), 3),
            "diff_vs_persistence": round(s["diff"], 3),
            "ci_95": [round(s["ci_lo"], 3), round(s["ci_hi"], 3)],
            "tie": s["tie"], "folds_better_than_persistence": better,
        })
    return rows


def paired_comparison(fold_df, tag_a, tag_b, horizon=1):
    """Model A minus model B on log-MAE per fold (positive = A worse)."""
    a = fold_df[(fold_df.tag == tag_a) & (fold_df.horizon == horizon)].sort_values("year")
    b = fold_df[(fold_df.tag == tag_b) & (fold_df.horizon == horizon)].sort_values("year")
    s = summarize_vs_reference(a["log_model"].values, b["log_model"].values)
    return {"a": VARIANTS.get(tag_a, tag_a), "b": VARIANTS.get(tag_b, tag_b), "metric": "log-MAE",
            "diff_a_minus_b": round(s["diff"], 3), "ci_95": [round(s["ci_lo"], 3), round(s["ci_hi"], 3)],
            "tie": s["tie"]}


def comparisons(fold_df):
    return [paired_comparison(fold_df, "gat", "geo_w"),
            paired_comparison(fold_df, "single", "geo_w")]


def load_fold_df(conn):
    return pd.read_sql_query("SELECT * FROM fold_results", conn)
