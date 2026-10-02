import ast, numpy as np

G = "data/processed/graph/"
states = open("configs/states.txt").read().split("\n"); N = len(states)

PTS = None
for node in ast.parse(open("models/stgnn/download_weather.py").read()).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "PTS":
        PTS = ast.literal_eval(node.value)
assert PTS is not None and set(PTS) == set(states)

lat = np.radians([PTS[s][0] for s in states]); lon = np.radians([PTS[s][1] for s in states])
a = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
     + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
D = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))          # km
np.save(G + "distance_km.npy", D)

geo = np.load(G + "adjacency_raw.npy")

# kNN (k=4, symmetrised)
knn = np.zeros((N, N), np.float32)
for i in range(N):
    for j in np.argsort(D[i])[1:5]:
        knn[i, j] = 1
knn = np.maximum(knn, knn.T)

# distance kernel: exp(-(d/500km)^2), weights < 0.1 dropped, nearest neighbour always kept
SIG = 500.0
dist = np.exp(-(D / SIG) ** 2).astype(np.float32)
dist[dist < 0.1] = 0
np.fill_diagonal(dist, 0)
for i in range(N):
    j = np.argsort(D[i])[1]
    dist[i, j] = max(dist[i, j], np.exp(-(D[i, j] / SIG) ** 2))
dist = np.maximum(dist, dist.T).astype(np.float32)

# hybrid: land borders (weight 1) plus half-weighted distance kernel
hyb = np.maximum(geo, 0.5 * dist).astype(np.float32)

def normalise(A):
    Ah = A + np.eye(N, dtype=np.float32)
    d = Ah.sum(1)
    Dm = np.diag(1 / np.sqrt(d))
    return (Dm @ Ah @ Dm).astype(np.float32)

def connected(A):
    seen, st = {0}, [0]
    while st:
        u = st.pop()
        for v in np.where(A[u] > 0)[0]:
            if v not in seen: seen.add(v); st.append(v)
    return len(seen) == N

for name, A in [("knn", knn), ("dist", dist), ("hybrid", hyb)]:
    np.save(G + f"adjacency_{name}_raw.npy", A)
    np.save(G + f"adjacency_{name}_norm.npy", normalise(A))
    deg = (A > 0).sum(1)
    print(f"{name:7s} edges {int((A > 0).sum() / 2):4d} | degree min/max/mean {deg.min()}/{deg.max()}/{deg.mean():.1f} "
          f"| connected {connected(A)} | symmetric {np.allclose(A, A.T)}")