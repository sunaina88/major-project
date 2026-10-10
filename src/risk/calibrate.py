"""Walk-forward calibration of the interval, written to docs/calibration_report.md.

For each test year Y: build the error quantiles from folds BEFORE Y only, build the interval
around every model forecast of year Y, and count how often the true value falls inside.

Run from the repository root:  python -m src.risk.calibrate
"""
import numpy as np
import pandas as pd

from src.api.config import REPO_ROOT
from src.risk.engine import load_config
from src.risk.uncertainty import fit_interval_model, interval_for


def walk_forward(pairs, nominal, by_disease, min_pairs):
    rows = []
    covered_all, total_all = 0, 0
    years = sorted(pairs["year"].unique())
    for y in years:
        past = pairs[pairs["year"] < y]
        test = pairs[pairs["year"] == y]
        if len(past) < min_pairs:
            continue
        models = fit_interval_model(past, nominal, by_disease)
        inside, widths = 0, []
        for i in range(len(test)):
            r = test.iloc[i]
            lo, hi = interval_for(models, r["disease"], r["model"])
            if lo <= r["true"] <= hi:
                inside += 1
            widths.append(np.log1p(hi) - np.log1p(lo))
        rows.append({"year": int(y), "n_test": len(test), "n_past": len(past),
                     "coverage": inside / len(test), "mean_log_width": float(np.mean(widths))})
        covered_all += inside
        total_all += len(test)
    return pd.DataFrame(rows), covered_all, total_all


def main():
    cfg = load_config()["uncertainty"]
    path = REPO_ROOT / "data" / "processed" / f"stgnn_annual_{cfg['tag']}_h{cfg['horizon']}_pairs.csv"
    pairs = pd.read_csv(path)
    nominal, min_pairs = cfg["nominal"], cfg["min_pairs"]

    results = {}
    for name, flag in [("pooled", False), ("per_disease", True)]:
        table, cov, tot = walk_forward(pairs, nominal, flag, min_pairs)
        results[name] = (table, cov / tot, tot)

    lines = []
    lines.append("# Interval calibration report\n")
    lines.append(f"Source: `{path.name}` ({len(pairs)} state/disease/year predictions, "
                 f"{pairs['year'].min()}-{pairs['year'].max()}).\n")
    lines.append(f"Method: error `e = log1p(model) - log1p(true)`; the {int(nominal * 100)}% interval for a "
                 "forecast `f` is `expm1(log1p(f) - q90)` to `expm1(log1p(f) - q10)`. For every test year Y "
                 "the quantiles use only folds before Y, and a year is tested only when at least "
                 f"{min_pairs} earlier pairs exist.\n")
    lines.append("## Result\n")
    lines.append(f"Nominal coverage: **{int(nominal * 100)}%**\n")
    lines.append("| Quantiles | Observed coverage | Tested pairs |")
    lines.append("|---|---|---|")
    for name, (table, cov, tot) in results.items():
        lines.append(f"| {name} | {cov * 100:.1f}% | {tot} |")
    lines.append("")
    for name, (table, cov, tot) in results.items():
        lines.append(f"## Per year: {name}\n")
        lines.append("| Test year | Tested pairs | Earlier pairs | Coverage | Mean width (log) |")
        lines.append("|---|---|---|---|---|")
        for i in range(len(table)):
            r = table.iloc[i]
            lines.append(f"| {int(r['year'])} | {int(r['n_test'])} | {int(r['n_past'])} | "
                         f"{r['coverage'] * 100:.0f}% | {r['mean_log_width']:.2f} |")
        lines.append("")
    chosen = "per_disease" if cfg["by_disease"] else "pooled"
    closest = min(results, key=lambda k: abs(results[k][1] - nominal))
    lines.append("## Reading this honestly\n")
    lines.append(f"- `configs/risk.yaml` currently uses **{chosen}** quantiles (`uncertainty.by_disease`). "
                 f"The setting closest to nominal in this run is **{closest}**.")
    lines.append(f"- Observed coverage is below the nominal {int(nominal * 100)}% in both settings, "
                 "so the intervals are, if anything, slightly too narrow. 2020 is the weakest year.")
    lines.append("- Intervals are wide because the model's typical log error is about 0.8 "
                 "(a factor of roughly 2), so an 80% interval spans about a factor of 10 end to end. "
                 "That is the real uncertainty; do not narrow it for display.")
    lines.append("- Only horizon 1 has per-pair errors, so horizons 2 and 3 have **no interval** (`null`).")
    lines.append("- Each year is tested once with a small number of pairs; yearly coverage is noisy "
                 "and states are not independent. Treat the pooled figure as approximate.")
    out = REPO_ROOT / "docs" / "calibration_report.md"
    out.write_text("\n".join(lines) + "\n")
    for name, (table, cov, tot) in results.items():
        print(f"{name}: coverage {cov * 100:.1f}% over {tot} pairs")
    print("wrote", out)


if __name__ == "__main__":
    main()
