import pandas as pd

tensor = pd.read_csv("data/processed/pan_india_state_enriched_tensor.csv", parse_dates=["start_date"])
audit = pd.read_csv("data/processed/zero_audit.csv").fillna({"verdict": ""})
tensor["year"] = tensor.start_date.dt.year
FIRST = {"dengue": 1999, "malaria": 2001, "chikungunya": 2006}

nr = audit[audit.verdict.str.strip() == "not_reported"]
for _, r in nr.iterrows():
    m = (tensor.state_id == r.state_id) & (tensor.year == r.year)
    for col in (f"{r.disease}_cases", f"{r.disease}_deaths"):
        tensor.loc[m, col] = float("nan")
print("blanked", len(nr), "state-years marked not_reported")

for d in FIRST:
    ann = tensor.groupby(["state_id", "year"])[f"{d}_cases"].sum(min_count=1).rename("annual").reset_index()
    ann[f"{d}_reporting_status"] = "reported"
    ann.loc[ann.annual == 0, f"{d}_reporting_status"] = "reported_zero"
    ann.loc[ann.annual.isna(), f"{d}_reporting_status"] = "not_reported"
    ann.loc[ann.year < FIRST[d], f"{d}_reporting_status"] = "not_applicable"
    tensor = tensor.merge(ann[["state_id", "year", f"{d}_reporting_status"]], on=["state_id", "year"], how="left")

tensor = tensor.drop(columns=["year"])
tensor.to_csv("data/processed/pan_india_state_enriched_tensor_clean.csv", index=False)
print(tensor.shape)
for d in FIRST:
    print(d, tensor[f"{d}_reporting_status"].value_counts().to_dict())