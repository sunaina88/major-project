import numpy as np, torch
from model import STModel, ALL, NO_WEATHER

G = "data/processed/graph/"
A_norm = torch.tensor(np.load(G + "adjacency_norm.npy"))
A_raw = torch.tensor(np.load(G + "adjacency_raw.npy"))
d = np.load(G + "dataset.npz")
x = torch.tensor(d["xs"][:8])

for kind in ["gcn", "gat"]:
    for name, idx in [("all", ALL), ("no_weather", NO_WEATHER)]:
        m = STModel(A_norm, A_raw, feat_idx=idx, kind=kind)
        out = m(x)
        n = sum(p.numel() for p in m.parameters())
        print(f"{kind:4s} {name:10s} out {tuple(out.shape)} params {n} nan={bool(torch.isnan(out).any())}")

# no-graph ablation
I = torch.eye(36)
m = STModel(I, I, kind="gcn")
print("identity graph out:", tuple(m(x).shape))