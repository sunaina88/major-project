import requests, pandas as pd

states = [s.strip() for s in open("configs/states.txt") if s.strip()]
rows = []
for s in states:
    for d in ["dengue", "malaria", "chikungunya"]:
        r = requests.get(f"http://localhost:8000/explanation/{s}/{d}", timeout=120)
        if r.status_code != 200:
            continue
        j = r.json()
        row = {"state": s, "disease": d, "notes": ";".join(j["notes"])}
        row.update({g["group"]: g["share_of_total_change"] for g in j["groups"]})
        rows.append(row)
df = pd.DataFrame(rows)
cols = ["own_history", "other_diseases", "neighbours", "weather"]
print(len(df), "pairs explained")
print(df[cols].describe().round(3))
print((df.neighbours > 0.3).sum(), "pairs with neighbours above 30%")
print(df.sort_values("neighbours", ascending=False).head(8)[["state", "disease", "neighbours", "notes"]].to_string())