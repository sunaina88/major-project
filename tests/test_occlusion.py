import pytest

pytest.importorskip("torch")
import pandas as pd  # noqa: E402

from src.explainability import occlusion  # noqa: E402


def test_checkpoint_reproduces_forecast_file_horizon_1():
    df = pd.read_csv("data/processed/stgnn_forecast_final.csv")
    df = df[df.horizon == 1]
    for i in range(len(df)):
        r = df.iloc[i]
        assert round(occlusion.base_forecast(r.state_id, r.disease, 1)) == r.forecast


def test_shares_sum_to_one_and_label():
    r = occlusion.explain("maharashtra", "malaria", 1)
    total = sum(g["share_of_total_change"] for g in r["groups"])
    assert total == pytest.approx(1.0, abs=0.01) or total == 0
    assert "causal" in r["label"]


def test_neighbour_group_uses_the_model_graph():
    # the graph also links island territories (Lakshadweep -> Kerala), so every state has a neighbour
    assert occlusion.neighbours_of("lakshadweep") == ["kerala"]
    assert len(occlusion.neighbours_of("maharashtra")) >= 5
    r = occlusion.explain("lakshadweep", "dengue", 1)
    assert [g["group"] for g in r["groups"]][2] == "neighbours"
