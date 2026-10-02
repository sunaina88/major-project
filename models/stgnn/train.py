import argparse, time, numpy as np, torch
from model import STModel, ALL, NO_WEATHER, masked_mae, masked_huber

p = argparse.ArgumentParser()
p.add_argument("--kind", default="gcn", choices=["gcn", "gat"])
p.add_argument("--features", default="all", choices=["all", "noweather"])
p.add_argument("--graph", default="norm", choices=["norm", "identity"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--hidden", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--patience", type=int, default=30)
p.add_argument("--bs", type=int, default=16)
p.add_argument("--name", default=None)
p.add_argument("--cap", type=float, default=1.5)
p.add_argument("--base", default="lag12", choices=["lag12", "lag1"])
a = p.parse_args()

name = a.name or f"{a.kind}_{a.features}_{a.graph}_s{a.seed}"
torch.manual_seed(a.seed); np.random.seed(a.seed)

G = "data/processed/graph/"
d = np.load(G + "dataset.npz")
xs, ys, ms = (torch.tensor(d[k]) for k in ["xs", "ys", "ms"])
tr, va = d["train_idx"], d["val_idx"]

if a.graph == "norm":
    A_norm = torch.tensor(np.load(G + "adjacency_norm.npy"))
    A_raw = torch.tensor(np.load(G + "adjacency_raw.npy"))
else:
    A_norm = torch.eye(36); A_raw = torch.zeros(36, 36)

feat = ALL if a.features == "all" else NO_WEATHER
model = STModel(A_norm, A_raw, feat_idx=feat, hidden=a.hidden, kind=a.kind, cap=a.cap, base=a.base)
opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)

def evaluate(idx):
    model.eval()
    with torch.no_grad():
        pred = model(xs[idx])
        return masked_mae(pred, ys[idx], ms[idx]).item()

best, best_ep, bad, hist = 1e9, 0, 0, []
t0 = time.time()
for ep in range(1, a.epochs + 1):
    model.train()
    perm = np.random.permutation(tr)
    tot = 0.0
    for i in range(0, len(perm), a.bs):
        b = perm[i:i + a.bs]
        opt.zero_grad()
        loss = masked_huber(model(xs[b]), ys[b], ms[b])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item() * len(b)
    trl = tot / len(perm)
    val = evaluate(va)
    sched.step(val)
    hist.append((ep, trl, val))
    if val < best - 1e-4:
        best, best_ep, bad = val, ep, 0
        torch.save({"state": model.state_dict(), "args": vars(a),
                    "feat_idx": feat, "best_val": best, "epoch": ep},
                   f"models/stgnn/{name}.pt")
    else:
        bad += 1
    if ep % 10 == 0 or ep == 1:
        print(f"ep {ep:3d} train {trl:.4f} val_mae(norm) {val:.4f} best {best:.4f}@{best_ep}")
    if bad >= a.patience:
        print(f"early stop at epoch {ep}")
        break

np.save(f"models/stgnn/{name}_history.npy", np.array(hist))
print(f"DONE {name} | best val {best:.4f} at epoch {best_ep} | {time.time()-t0:.0f}s")