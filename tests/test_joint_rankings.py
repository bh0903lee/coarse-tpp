"""Joint rankings of time and mark (Supplementary S4, Table S13): rankings from stored results, and a tiny run."""
import json

import numpy as np

from _toy import tied_toy, train_cells
from coarse_tpp import data as D, experiments as X, joint_rankings as JR


def res(seed, f6, C, mark, ties=0.3):
    return dict(seed=seed, by_selection={"J": {"test": {"floor_1e-6": f6, "C": C, "mark": mark, "J": C + mark,
                                                         "tie_share": ties}}})


def test_unit_row_ranks_and_deficit():
    arms = {"floor_1e-6": {0: res(0, 2.0, -3.0, -1.0), 1: res(1, 2.2, -3.2, -1.0)},
            "interval": {0: res(0, -1.0, -1.0, -0.8), 1: res(1, -1.0, -1.0, -0.8)},
            "jitter_gap": {0: res(0, -0.5, -1.5, -0.9), 1: res(1, -0.5, -1.5, -0.9)}}
    r = JR.unit_row(("d", "gru"), arms)
    assert r["joint_top_floor"] == "floor_1e-6" and r["joint_top_interval"] == "interval" and r["joint_changed"]
    assert r["joint_floor_self"] and r["joint_rank_of_floor_top"] == 3
    assert np.isclose(r["joint_deficit_of_floor_top"], (-1.8) - (-4.1))
    assert np.isclose(r["scores"]["joint_floor"]["floor_1e-6"], 1.1) and np.isclose(r["joint_tau"], -1.0)
    assert r["time_top_interval"] == "interval" and r["seeds"] == [0, 1]
    c = JR.summary_counts([r, dict(r, tie_share=0.0, joint_changed=False)])
    assert c["n_tied"] == 1 and c["n_joint_changed"] == 1 and c["joint_rank_of_floor_top"] == [3]


def test_end_to_end_on_tiny_cells(tmp_path, monkeypatch):
    old = D.data_dir()
    D.set_data_dir(tmp_path)
    try:
        ds = tied_toy(tmp_path, n_types=2)
        run = tmp_path / "run"
        monkeypatch.setattr(X, "MARK_ARMS", ("floor_1e-6", "interval"))
        monkeypatch.setattr(X, "REAL_SEEDS", (0,))
        train_cells("marks", run, ds, X.MARK_ARMS + ("mark_only",), 1.0, lrs=("lr3e4", "lr1e3"),
                    extra=dict(mark_head=True), encoder=dict(X.GRU, marks=True))
        L = JR.cmd_report(run, run / "joint_rankings")
        S = json.loads((run / "joint_rankings" / "summary.json").read_text())
        r = S["rows"][0]
        assert S["complete"] and r["n_arms"] == 2 and set(r["scores"]["joint_interval"]) == set(X.MARK_ARMS)
        assert r["joint_rank_of_floor_top"] in (1, 2) and r["joint_deficit_of_floor_top"] >= 0
        assert any("Joint top changes" in x for x in L)
    finally:
        D.set_data_dir(old)
