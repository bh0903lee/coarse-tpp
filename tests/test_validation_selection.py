"""Validation selection among the candidates of a rank_change run (coarse_tpp.validation_selection)."""
import pytest

from coarse_tpp import validation_selection as VS

ARMS = ("floor_1e-10", "floor_1e-6", "floor_1e-5", "interval")
LRS = ("lr1e3", "lr3e4")


def res(arm, lr, seed, dev, test_c):
    return dict(cell_id=f"{arm}_{lr}_{seed}", arm=arm, seed=seed, primary_selection="C",
                config=dict(dataset="d", encoder=dict(type="gru"), lr_tag=lr),
                dev=dict(dev), test=dict(C=test_c))


def grid():
    cells = []
    for i, arm in enumerate(ARMS):
        for j, lr in enumerate(LRS):
            for s in range(3):
                # each floor scores its own model highest on dev; dev C prefers the interval model
                dev = {f: (5.0 if arm == f else 1.0) + j for f in VS.FLOORS}
                dev["C"] = (5.0 if arm == "interval" else 1.0) - j
                test_c = (2.0 if arm == "interval" else 1.0 + 0.1 * i) - 0.5 * j + 0.01 * s
                cells.append(res(arm, lr, s, dev, test_c))
    return cells


def test_each_floor_chooses_its_own_model_and_the_deficit_is_seed_paired():
    rows = VS.summarize(VS.candidates(grid()), {"d": {"tie_share": 0.05}}, controls=(), arms=ARMS)
    (r,) = rows
    assert r["tied"] and r["n_candidates"] == 8
    assert r["pick_C"] == ["interval", "lr1e3"]
    for i, f in enumerate(VS.FLOORS):
        x = r[f]
        assert x["self_choice"] and x["pick"] == [f, "lr3e4"]
        assert x["deficit"] == pytest.approx(2.0 - (1.0 + 0.1 * i - 0.5))
        assert x["lo95"] == pytest.approx(x["deficit"]) and x["seeds_positive"] == 3
    # ranks by mean test C: interval lr1e3, interval lr3e4, the floors at lr1e3, then the floors at lr3e4
    assert r["floor_1e-5"]["rank"] == 6 and r["floor_1e-10"]["rank"] == 8


def test_a_result_scored_at_another_checkpoint_is_refused():
    cells = grid()
    cells[0]["primary_selection"] = "own"
    with pytest.raises(ValueError):
        VS.candidates(cells)
