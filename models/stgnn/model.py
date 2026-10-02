import torch, torch.nn as nn, torch.nn.functional as F

ALL = list(range(15))
NO_WEATHER = [0, 1, 2, 3, 4, 5, 13, 14]


class GCNLayer(nn.Module):
    """x: [B,N,Fin] -> relu(A_norm @ (x W))"""
    def __init__(self, fin, fout, A):
        super().__init__()
        self.register_buffer("A", A)          # [N,N] normalized
        self.lin = nn.Linear(fin, fout)

    def forward(self, x):
        return F.relu(torch.einsum("ij,bjf->bif", self.A, self.lin(x)))


class GATLayer(nn.Module):
    """Dense single-head-per-head GAT restricted to graph edges (+ self loops)."""
    def __init__(self, fin, fout, A_raw, heads=2, dropout=0.1):
        super().__init__()
        N = A_raw.shape[0]
        mask = (A_raw + torch.eye(N)) > 0
        self.register_buffer("mask", mask)    # [N,N] bool
        self.h, self.fo = heads, fout // heads
        assert fout % heads == 0
        self.W = nn.Linear(fin, fout, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, self.fo) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(heads, self.fo) * 0.1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.h, self.fo)             # [B,N,H,fo]
        es = (h * self.a_src).sum(-1)                          # [B,N,H]
        ed = (h * self.a_dst).sum(-1)                          # [B,N,H]
        e = F.leaky_relu(ed.unsqueeze(2) + es.unsqueeze(1), 0.2)   # [B,N(i),N(j),H]
        e = e.masked_fill(~self.mask.view(1, N, N, 1), float("-inf"))
        att = self.drop(torch.softmax(e, dim=2))               # over neighbors j
        out = torch.einsum("bijh,bjhf->bihf", att, h).reshape(B, N, -1)
        return F.elu(out)


class STModel(nn.Module):
    def __init__(self, A_norm, A_raw, feat_idx=ALL, hidden=64, kind="gcn",
                 dropout=0.2, cap=1.5, base="lag12"):
        super().__init__()
        self.cap = cap
        self.base = base
        self.register_buffer("feat_idx", torch.tensor(feat_idx, dtype=torch.long))
        fin = len(feat_idx)
        if kind == "gcn":
            self.g1 = GCNLayer(fin, hidden, A_norm)
            self.g2 = GCNLayer(hidden, hidden, A_norm)
        elif kind == "gat":
            self.g1 = GATLayer(fin, hidden, A_raw)
            self.g2 = GATLayer(hidden, hidden, A_raw)
        else:
            raise ValueError(kind)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):                      # x: [B,L,N,15]
        base = x[:, 0 if self.base == "lag12" else -1, :, 0:3]
        x = x[..., self.feat_idx]
        B, L, N, Fd = x.shape
        h = self.g1(x.reshape(B * L, N, Fd)).view(B, L, N, -1)
        h = h.permute(0, 2, 1, 3).reshape(B * N, L, -1)
        _, hn = self.gru(h)
        hn = hn[0].view(B, N, -1)
        hn = self.g2(self.drop(hn))
        return base + self.cap * torch.tanh(self.head(self.drop(hn)))

def masked_mae(pred, y, m):
    return ((pred - y).abs() * m).sum() / m.sum().clamp(min=1)


def masked_huber(pred, y, m, delta=1.0):
    l = F.smooth_l1_loss(pred, y, reduction="none", beta=delta)
    return (l * m).sum() / m.sum().clamp(min=1)