import numpy as np

states = open("configs/states.txt").read().split("\n")
idx = {s: i for i, s in enumerate(states)}
N = len(states)

NEIGHBORS = {
 "andaman_and_nicobar_islands": ["west_bengal", "odisha"],
 "andhra_pradesh": ["odisha", "chhattisgarh", "telangana", "karnataka", "tamil_nadu", "puducherry"],
 "arunachal_pradesh": ["assam", "nagaland"],
 "assam": ["arunachal_pradesh", "nagaland", "manipur", "mizoram", "tripura", "meghalaya", "west_bengal"],
 "bihar": ["uttar_pradesh", "jharkhand", "west_bengal"],
 "chandigarh": ["punjab", "haryana"],
 "chhattisgarh": ["madhya_pradesh", "maharashtra", "telangana", "andhra_pradesh", "odisha", "jharkhand", "uttar_pradesh"],
 "dadra_and_nagar_haveli_and_daman_and_diu": ["gujarat", "maharashtra"],
 "delhi": ["haryana", "uttar_pradesh"],
 "goa": ["maharashtra", "karnataka"],
 "gujarat": ["rajasthan", "madhya_pradesh", "maharashtra", "dadra_and_nagar_haveli_and_daman_and_diu"],
 "haryana": ["punjab", "himachal_pradesh", "uttarakhand", "uttar_pradesh", "rajasthan", "delhi", "chandigarh"],
 "himachal_pradesh": ["jammu_and_kashmir", "ladakh", "punjab", "haryana", "uttarakhand", "uttar_pradesh"],
 "jammu_and_kashmir": ["ladakh", "himachal_pradesh", "punjab"],
 "jharkhand": ["bihar", "uttar_pradesh", "chhattisgarh", "odisha", "west_bengal"],
 "karnataka": ["maharashtra", "goa", "telangana", "andhra_pradesh", "tamil_nadu", "kerala"],
 "kerala": ["karnataka", "tamil_nadu", "lakshadweep", "puducherry"],
 "ladakh": ["jammu_and_kashmir", "himachal_pradesh"],
 "lakshadweep": ["kerala"],
 "madhya_pradesh": ["rajasthan", "uttar_pradesh", "chhattisgarh", "maharashtra", "gujarat"],
 "maharashtra": ["gujarat", "madhya_pradesh", "chhattisgarh", "telangana", "karnataka", "goa",
                 "dadra_and_nagar_haveli_and_daman_and_diu"],
 "manipur": ["nagaland", "assam", "mizoram"],
 "meghalaya": ["assam"],
 "mizoram": ["assam", "tripura", "manipur"],
 "nagaland": ["assam", "arunachal_pradesh", "manipur"],
 "odisha": ["west_bengal", "jharkhand", "chhattisgarh", "andhra_pradesh"],
 "puducherry": ["tamil_nadu", "andhra_pradesh", "kerala"],
 "punjab": ["jammu_and_kashmir", "himachal_pradesh", "haryana", "rajasthan", "chandigarh"],
 "rajasthan": ["punjab", "haryana", "uttar_pradesh", "madhya_pradesh", "gujarat"],
 "sikkim": ["west_bengal"],
 "tamil_nadu": ["kerala", "karnataka", "andhra_pradesh", "puducherry"],
 "telangana": ["maharashtra", "chhattisgarh", "karnataka", "andhra_pradesh"],
 "tripura": ["assam", "mizoram"],
 "uttar_pradesh": ["uttarakhand", "himachal_pradesh", "haryana", "delhi", "rajasthan",
                   "madhya_pradesh", "chhattisgarh", "jharkhand", "bihar"],
 "uttarakhand": ["himachal_pradesh", "uttar_pradesh", "haryana"],
 "west_bengal": ["sikkim", "assam", "bihar", "jharkhand", "odisha", "andaman_and_nicobar_islands"],
}

# validate names
assert set(NEIGHBORS) == set(states), set(states) ^ set(NEIGHBORS)
for s, nb in NEIGHBORS.items():
    for n in nb:
        assert n in idx, f"unknown neighbor {n} in {s}"

# symmetric raw adjacency (no self loops)
A = np.zeros((N, N), dtype=np.float32)
for s, nb in NEIGHBORS.items():
    for n in nb:
        A[idx[s], idx[n]] = 1
        A[idx[n], idx[s]] = 1
np.fill_diagonal(A, 0)

# connectivity check (BFS)
seen, stack = {0}, [0]
while stack:
    u = stack.pop()
    for v in np.where(A[u] > 0)[0]:
        if v not in seen:
            seen.add(v); stack.append(v)
print("connected:", len(seen) == N)

# normalized: D^-1/2 (A+I) D^-1/2
A_hat = A + np.eye(N, dtype=np.float32)
d = A_hat.sum(1)
D = np.diag(1.0 / np.sqrt(d))
A_norm = (D @ A_hat @ D).astype(np.float32)

# identity version for the "no graph" ablation
A_id = np.eye(N, dtype=np.float32)

np.save("data/processed/graph/adjacency_raw.npy", A)
np.save("data/processed/graph/adjacency_norm.npy", A_norm)
np.save("data/processed/graph/adjacency_identity.npy", A_id)

deg = A.sum(1).astype(int)
print("nodes:", N, "| undirected edges:", int(A.sum() / 2))
print("degree min/max/mean:", deg.min(), deg.max(), round(deg.mean(), 2))
print("symmetric:", np.allclose(A, A.T))
for s in states:
    print(f"{s:45s} {deg[idx[s]]}")