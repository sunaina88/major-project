import argparse, numpy as np, pandas as pd, torch, os
from model import STModel
from annual_data import load_graph, to_cases, forecast_table

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="models/stgnn/annual_final.pt")
p.add_argument("--out", default="data/processed/stgnn_forecast_from_checkpoint.csv")
a = p.parse_args()

ck = torch.load(a.ckpt, weights_only=False)
args = ck["args"]; N = len(ck["states"])
A_norm, A_raw = load_graph(args["graph"])
x = torch.tensor(ck["X_last"][None])
frames = []
for h, store in ck["models"].items():
    pred = np.zeros((N, 3)); capped = np.zeros((N, 3), bool)
    for grp in store:
        cs = []
        for sdict in grp["states"]:
            m = STModel(A_norm, A_raw, feat_idx=grp["feat"], hidden=args["hidden"],
                        kind=args["kind"], cap=args["cap"], base="lag1")
            m.load_state_dict(sdict); m.eval()
            with torch.no_grad():
                pn = m(x)[0].numpy()
            cs.append(to_cases(pn, ck["mu"], ck["sd"], ck["tmax"]))
        g = grp["group"]
        pred[:, g] = np.mean([c[0] for c in cs], 0)[:, g]
        capped[:, g] = (np.mean([c[1] for c in cs], 0) > 0.5)[:, g]
    frames.append(forecast_table(ck["states"], ck["years"][-1] + int(h), int(h), pred, capped, ck["last"], ck["nobs"]))
df = pd.concat(frames, ignore_index=True)
df.to_csv(a.out, index=False)
print(f"wrote {a.out} ({len(df)} rows)")

ref = f"data/processed/stgnn_forecast_{args['tag']}.csv"
if os.path.exists(ref):
    r = pd.read_csv(ref)
    same = (r.forecast.values == df.forecast.values).all()
    print("matches the CSV written at training time:", bool(same))