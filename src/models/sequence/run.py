import numpy as np, pandas as pd
from src.evaluation.data import get_data
from src.evaluation.backtest import run_backtest, persistence_fn
from src.evaluation.metrics import summarize_vs_reference
from src.models.sequence.seq import make_predict_fn

if __name__ == "__main__":
    states, years, Y, _ = get_data()
    out, summ = {}, []
    for kind in ["gru", "lstm"]:
        folds, pairs = run_backtest(Y, years, states, make_predict_fn(kind))
        folds.to_csv(f"data/processed/seq_{kind}_h1.csv", index=False)
        pairs.to_csv(f"data/processed/seq_{kind}_h1_pairs.csv", index=False)
        s = summarize_vs_reference(folds.log_model, folds.log_last)
        summ.append(dict(model=kind, log_mae=folds.log_model.mean(), log_persistence=folds.log_last.mean(), **s))
    res = pd.DataFrame(summ); res["n_comparisons"] = len(res)
    res.to_csv("data/processed/seq_results.csv", index=False)
    print(res.round(3).to_string())
