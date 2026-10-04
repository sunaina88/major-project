import numpy as np, torch, torch.nn as nn

LAG = 3

class Net(nn.Module):
    def __init__(self, kind="gru", hidden=16, in_dim=5):
        super().__init__()
        self.rnn = (nn.GRU if kind == "gru" else nn.LSTM)(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, x):
        h, _ = self.rnn(x)
        return x[:, -1, 0] + self.head(h[:, -1]).squeeze(-1)   # residual on last year's log1p

def windows(L, t_list):
    """L: [T,S,D] log1p with NaN. Returns X [N,LAG,5], last_obs mask, target, index."""
    T, S, D = L.shape
    X, tgt, keep, idx = [], [], [], []
    for t in t_list:
        for s in range(S):
            for d in range(D):
                w = L[t - LAG:t, s, d]
                m = ~np.isnan(w)
                oh = np.zeros((LAG, 3)); oh[:, d] = 1
                X.append(np.column_stack([np.where(m, w, 0.0), m.astype(float), oh]))
                tgt.append(L[t, s, d] if t < T else np.nan)
                keep.append(m[-1]); idx.append((s, d))
    return np.array(X, np.float32), np.array(keep), np.array(tgt, np.float32), idx

def make_predict_fn(kind="gru", epochs=150, lr=5e-3, wd=1e-3, seeds=(0, 1, 2)):
    def fn(Ytr, ytr, yr):
        L = np.log1p(Ytr)
        T, S, D = L.shape
        X, keep, tgt, _ = windows(L, range(LAG, T))
        ok = keep & ~np.isnan(tgt)                      # masked loss: target and last year observed
        Xt, yt = torch.tensor(X[ok]), torch.tensor(tgt[ok])
        Lext = np.concatenate([L, np.full((1, S, D), np.nan)])
        Xq, kq, _, idx = windows(Lext, [T])
        Xq = torch.tensor(Xq)
        preds = []
        for sd in seeds:
            torch.manual_seed(sd)
            net = Net(kind); opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
            for _ in range(epochs):
                opt.zero_grad(); loss = (net(Xt) - yt).abs().mean(); loss.backward(); opt.step()
            with torch.no_grad(): preds.append(net(Xq).numpy())
        p = np.mean(preds, axis=0)
        out = np.full((S, D), np.nan)
        for (s, d), v in zip(idx, p): out[s, d] = np.expm1(v)
        return out
    return fn
