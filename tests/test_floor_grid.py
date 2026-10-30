"""Floor grid (Supplementary S4, Fig. S4): pure functions, cache rules, and a tiny end-to-end run on CPU."""
import json
import math
from pathlib import Path

import numpy as np

from _toy import tied_toy, train_cells
from coarse_tpp import data as D, floor_grid as G


def test_grid_contains_the_training_floors():
    assert len(G.GRID) == 37 and G.GRID[0] == 1e-12 and math.isclose(G.GRID[-1], 1e-3)
    assert np.allclose(np.diff(np.log10(G.GRID)), 0.25)
    assert [G.grid_index(e) for e in (1e-10, 1e-6, 1e-5)] == [8, 24, 28]
    assert G.nearest_trained(1e-12) == "floor_1e-10" and G.nearest_trained(1e-3) == "floor_1e-5"
    assert G.nearest_trained(10 ** -5.5) == "floor_1e-6"          # equidistant: the smaller floor
    assert G.nearest_trained(1e-8) == "floor_1e-10"               # 1e-8 is two decades from both: the smaller floor


def test_top_and_changes():
    assert G.top({"interval": 1.0, "floor_1e-6": 1.0}) == "floor_1e-6"   # exact tie: order of ARMS
    assert G.n_changes(["a", "a", "b", "b", "a"]) == 2 and G.n_changes(["a"]) == 0


def curve(base, bumps):
    """Per-arm seed record whose mean floor score is base[arm] + bumps.get((arm, i), 0) at grid point i."""
    rec = {}
    for a in G.ARMS:
        s = np.full(len(G.GRID), base[a])
        for (b, i), v in bumps.items():
            if b == a:
                s[i] += v
        rec[a] = dict(tie=list(s * 10 / 2), non=list(s * 10 / 2), n=10)
    return rec


def test_unit_curve_and_structure():
    base = {a: -float(k) for k, a in enumerate(G.ARMS)}             # floor_1e-10 first everywhere ...
    bumps = {("floor_1e-6", G.grid_index(1e-6)): 5.0, ("floor_1e-5", G.grid_index(1e-5)): 5.0}   # ... except two floors
    c = {a: float(k) for k, a in enumerate(G.ARMS)}                  # interval first by C
    S, met = G.unit_curve([curve(base, bumps), curve(base, bumps)], c)
    assert met["changes"] == 4 and met["top_c"] == "interval" and met["share_top_eq_c"] == 0
    assert met["top_at_1e6"] == "floor_1e-6" and math.isclose(S["floor_1e-6"][G.grid_index(1e-6)], 4.0)
    st = G.structure(S, met["tops"])
    assert st["all_own_top"] and st["n_other_floor_top"] == 34       # floor_1e-10 is first at the other 34 floors
    assert st["loses_next_to_1e6"] and math.isclose(st["deficit_below_1e6"], 1.0)
    assert math.isclose(st["max_spread"], 11.0)                     # 4 (floor_1e-6 at 1e-6) - (-7)
    assert math.isclose(met["tau_min"], -1.0) and met["tau_max"] > -0.9        # reversed rankings except at the bumps


def test_cache_state(tmp_path):
    rec = dict(version=G.VERSION, grid=list(G.GRID), quarantined=False,
               models={a: dict(cell_id=f"c_{a}", ckpt_sha256="s" * 64, tie=[0.0] * 37, non=[-1.0] * 37, n=3) for a in G.ARMS})
    p = tmp_path / "x.json"
    p.write_text(json.dumps(rec))
    same = {a: (f"c_{a}", "s" * 64) for a in G.ARMS}
    assert G.cache_state(p, same) == "ok" and not G.should_score(p, expected=same) and G.should_score(p, force=True)
    assert G.cache_state(p, dict(same, interval=("c_other", "s" * 64))) == "stale"
    assert G.cache_state(p, dict(same, interval=("c_interval", "t" * 64))) == "stale"
    for change in (dict(grid=list(G.GRID[:-1])), dict(version=G.VERSION + 1), dict(quarantined=True)):
        p.write_text(json.dumps(dict(rec, **change)))
        assert G.cache_state(p, same) == ("quarantined" if "quarantined" in change else "invalid")
    p.write_text(json.dumps(dict(rec, quarantined=True)))
    assert not G.should_score(p, expected=same) and G.should_score(p, retry_quarantined=True, expected=same)
    assert G.cache_state(tmp_path / "missing.json") == "invalid"


def test_end_to_end_on_tiny_cells(tmp_path, monkeypatch):
    old = D.data_dir()
    D.set_data_dir(tmp_path)
    try:
        ds = tied_toy(tmp_path)
        run = tmp_path / "run"
        monkeypatch.setattr(G, "ARMS", ("floor_1e-6", "interval"))
        monkeypatch.setattr(G, "TRAINED_FLOORS", {"floor_1e-6": 1e-6})
        train_cells("rank_change", run, ds, G.ARMS, 1.0, lrs=("lr3e4", "lr1e3"))
        out = run / "floor_grid"
        G.cmd_score(run, out, cpu=True)
        p = out / "cells" / f"{G.item_name(ds, 'gru', 0)}.json"
        rec = json.loads(p.read_text())
        assert not rec["quarantined"] and all(m["abs_diff"] < 1e-8 and m["n_tie"] > 0 for m in rec["models"].values())
        sel = G.selected(run)
        res = sel[(ds, "gru")]["interval"][0]["by_selection"]["C"]["test"]
        m = rec["models"]["interval"]
        for key in ("floor_1e-10", "floor_1e-5"):                     # the stored floors are points of the grid
            i = G.grid_index(float(key.split("_")[1]))
            assert abs((m["tie"][i] + m["non"][i]) / m["n"] - res[key]) < 1e-8
        L = G.cmd_report(run, out)
        S = json.loads((out / "summary.json").read_text())
        assert S["complete"] and len(S["units"]) == 1 and len(S["units"][0]["tops"]) == len(G.GRID)
        assert len((out / "curves.csv").read_text().splitlines()) == 1 + len(G.GRID) * len(G.ARMS)
        assert any("Where the floor-trained models are first" in x for x in L)
        ck = Path(sel[(ds, "gru")]["interval"][0]["config"]["out_dir"]) / "best_C.pt"
        ck.write_bytes(ck.read_bytes() + b"\0")                       # a changed checkpoint makes the cache stale
        assert G.cache_state(p, G.expected_models(sel, ds, "gru", 0)) == "stale"
        G.cmd_report(run, out)
        assert not json.loads((out / "summary.json").read_text())["complete"]
    finally:
        D.set_data_dir(old)
