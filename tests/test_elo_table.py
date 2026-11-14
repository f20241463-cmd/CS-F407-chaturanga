from evaluation.elo_table import ANCHOR, fit, load_results


def test_fit_recovers_known_gap():
    # B scores 0.76 against the anchor, which is about +200 Elo.
    results = [("B", ANCHOR, 76.0, 100), ("C", "B", 50.0, 100)]
    r = fit(results)
    assert r[ANCHOR] == 0.0
    assert 190 < r["B"] < 210
    assert abs(r["C"] - r["B"]) < 5


def test_real_results_order():
    r = fit(load_results())
    final = r["checkpoints/ideas/g_actions32b/iter_1280.pt"]
    assert final > r["checkpoints/ideas/lowlr_chain_1220/iter_1220.pt"] > r[ANCHOR]
