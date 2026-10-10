"""Occlusion explanations for the annual ST-GNN. Needs torch and models/stgnn/annual_final.pt.

For one state and disease we remove one group of inputs at a time and measure how the
forecast changes. This is MODEL ATTRIBUTION, NOT A CAUSAL EFFECT.

Input layout of the model (last dimension, 8 features per state and year):
    0,1,2  normalised log1p cases of dengue, malaria, chikungunya
    3,4,5  'observed' flags of the same three diseases
    6,7    annual mean temperature, annual total rainfall
"Zeroing" a feature sets it to the training average (0 after normalisation) and its flag to 0.
"""
import functools
import importlib.util

import numpy as np

from src.api.config import DISEASES, REPO_ROOT, in_repo_root

STGNN = REPO_ROOT / "models" / "stgnn"
CKPT = STGNN / "annual_final.pt"

GROUP_TEXT = {
    "own_history": "The state's own history of this disease",
    "other_diseases": "The state's history of the other two diseases",
    "neighbours": "Disease history of neighbouring states in the model graph",
    "weather": "The state's own temperature and rainfall",
}


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, STGNN / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=1)
def _bundle():
    """Load the checkpoint once and rebuild every model, exactly like models/stgnn/predict.py."""
    import torch
    with in_repo_root():
        model_mod = _load_module("vw_stgnn_model", "model.py")
        data_mod = _load_module("vw_stgnn_annual_data", "annual_data.py")
        ck = torch.load(CKPT, weights_only=False)
        args = ck["args"]
        A_norm, A_raw = data_mod.load_graph(args["graph"])
    models = {}
    for h, store in ck["models"].items():
        groups = []
        for grp in store:
            nets = []
            for sdict in grp["states"]:
                m = model_mod.STModel(A_norm, A_raw, feat_idx=grp["feat"], hidden=args["hidden"],
                                      kind=args["kind"], cap=args["cap"], base="lag1")
                m.load_state_dict(sdict)
                m.eval()
                nets.append(m)
            groups.append((grp["group"], nets))
        models[int(h)] = groups
    return {"ck": ck, "models": models, "to_cases": data_mod.to_cases,
            "A_raw": A_raw.numpy(), "torch": torch}


def available_horizons():
    return sorted(_bundle()["models"].keys())


def _predict(x, horizon):
    """x: [K, N, 8] numpy -> forecast cases [N, 3] (average of the seeds, capped like predict.py)."""
    b = _bundle()
    torch, ck = b["torch"], b["ck"]
    xt = torch.tensor(x[None].astype(np.float32))
    N = x.shape[1]
    pred = np.zeros((N, 3))
    for g, nets in b["models"][horizon]:
        cases = []
        for net in nets:
            with torch.no_grad():
                pn = net(xt)[0].numpy()
            cases.append(b["to_cases"](pn, ck["mu"], ck["sd"], ck["tmax"])[0])
        pred[:, g] = np.mean(cases, 0)[:, g]
    return pred


def neighbours_of(state_id):
    """State ids linked to this state in the graph the model was trained with."""
    ck = _bundle()["ck"]
    return [ck["states"][i] for i in _neighbours(ck["states"].index(state_id))]


def _neighbours(j):
    A = _bundle()["A_raw"]
    out = []
    for i in range(A.shape[0]):
        if i != j and A[j, i] > 0:
            out.append(i)
    return out


def _occlude(x, group, j, k):
    x = x.copy()
    if group == "own_history":
        x[:, j, k] = 0.0
        x[:, j, 3 + k] = 0.0
    elif group == "other_diseases":
        for d in range(3):
            if d != k:
                x[:, j, d] = 0.0
                x[:, j, 3 + d] = 0.0
    elif group == "neighbours":
        for i in _neighbours(j):
            x[:, i, 0:6] = 0.0
    elif group == "weather":
        x[:, j, 6:8] = 0.0
    return x


def base_forecast(state_id, disease, horizon=1):
    """Forecast reproduced from the checkpoint (must equal the forecast file)."""
    b = _bundle()
    j = b["ck"]["states"].index(state_id)
    k = DISEASES.index(disease)
    return float(_predict(b["ck"]["X_last"], horizon)[j, k])


def explain(state_id, disease, horizon=1):
    b = _bundle()
    ck = b["ck"]
    if horizon not in b["models"]:
        raise ValueError(f"horizon {horizon} not in checkpoint")
    j = ck["states"].index(state_id)
    k = DISEASES.index(disease)
    x = ck["X_last"]
    base = float(_predict(x, horizon)[j, k])
    neighbours = _neighbours(j)

    groups = []
    total = 0.0
    for name in ["own_history", "other_diseases", "neighbours", "weather"]:
        without = float(_predict(_occlude(x, name, j, k), horizon)[j, k])
        d_log = float(np.log1p(without) - np.log1p(base))
        total += abs(d_log)
        groups.append({"group": name, "description": GROUP_TEXT[name],
                       "forecast_without_group": round(without, 1),
                       "change_cases": round(without - base, 1),
                       "change_log": round(d_log, 4)})
    for g in groups:
        g["share_of_total_change"] = round(abs(g["change_log"]) / total, 3) if total > 0 else 0.0
        if g["group"] == "neighbours" and not neighbours:
            g["note"] = "no_graph_neighbours"

    notes = []
    last = ck["last"][j, k]
    z = (np.log1p(base) >= np.log1p(ck["tmax"][j, k]) - 1e-9)
    if z:
        notes.append("forecast_is_at_historical_max_cap: changes that push the forecast higher are hidden")
    if total == 0:
        summary = "Removing these inputs did not change the forecast."
    else:
        top = max(groups, key=lambda g: g["share_of_total_change"])
        summary = (f"Largest change when removed: {top['description'].lower()} "
                   f"({int(round(top['share_of_total_change'] * 100))}% of the total change). "
                   "Model attribution, not a causal effect.")
    return {"state_id": state_id, "disease": disease, "year": int(ck["years"][-1]) + horizon,
            "horizon": horizon, "base_forecast": round(base, 1),
            "persistence": None if np.isnan(last) else float(last),
            "label": "Model attribution, not a causal effect",
            "method": "Occlusion: one input group at a time is set to the training average "
                      "(observed flag 0) and the forecast is recomputed.",
            "groups": groups, "summary": summary, "notes": notes}
