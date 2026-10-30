"""Ranking bootstrap (Supplementary S4, Table S7): resampling arithmetic, validity checks, and a tiny end-to-end run."""
import json
import math

import numpy as np
import pytest

from _toy import tied_toy, train_cells
from coarse_tpp import data as D, floor_grid as G, rank_bootstrap as RB

ARMS2 = ("floor_1e-6", "interval")


def test_scores_weight_sequences_and_seeds():
    n = np.array([2.0, 3.0])
    X = {"a": np.array([[2.0, 3.0], [4.0, 9.0]])}                  # (seeds, sequences)
    assert math.isclose(RB.scores(n, X, np.ones(2), np.arange(2))["a"], (5 / 5 + 13 / 5) / 2)
    assert math.isclose(RB.scores(n, X, np.array([2.0, 0.0]), np.array([1, 1]))["a"], 8 / 4)
    with pytest.raises(ValueError):
        RB.scores(n, X, np.zeros(2), np.arange(2))


def test_bootstrap_is_deterministic_and_counts(monkeypatch):
    monkeypatch.setattr(RB, "ARMS", ARMS2)
    rng = np.random.default_rng(0)
    n = rng.integers(1, 5, 30).astype(float)
    C = {"floor_1e-6": rng.normal(-1, 0.1, (3, 30)) * n, "interval": rng.normal(-0.5, 0.1, (3, 30)) * n}
    F6 = {"floor_1e-6": rng.normal(1, 0.1, (3, 30)) * n, "interval": rng.normal(-2, 0.1, (3, 30)) * n}
    a, b = RB.bootstrap(n, C, F6, B=50), RB.bootstrap(n, C, F6, B=50)
    assert a == b and a["p_diff"] == 1.0 and a["p_self"] == 1.0
    assert a["tau_q025"] == a["tau_q975"] == -1.0 and a["tau_n_undefined"] == 0
    o = RB.original(n, C, F6)
    assert (o["top_C"], o["top_floor"]) == ("interval", "floor_1e-6")
    with pytest.raises(ValueError):
        RB.bootstrap(n, C, F6, B=0)


def test_summary_counts():
    rows = [dict(tie_share=0.2, top_C="interval", top_floor="floor_1e-6", p_diff=1.0, p_self=1.0),
            dict(tie_share=0.05, top_C="floor_1e-6", top_floor="floor_1e-6", p_diff=0.6, p_self=0.95),
            dict(tie_share=0.0, top_C="interval", top_floor="floor_1e-10", p_diff=0.99, p_self=0.0)]
    c = RB.summary_counts(rows)
    assert (c["n_tied"], c["n_changed"], c["n_changed_p_diff_ge"], c["n_tied_p_self_ge"]) == (2, 1, 1, 2)
    assert c["min_p_self_tied"] == 0.95 and c["n_changed_p_diff_1"] == 1


def test_end_to_end_on_tiny_cells(tmp_path, monkeypatch):
    old = D.data_dir()
    D.set_data_dir(tmp_path)
    try:
        ds = tied_toy(tmp_path)
        run = tmp_path / "run"
        for m in (G, RB):
            monkeypatch.setattr(m, "ARMS", ARMS2)
        train_cells("rank_change", run, ds, ARMS2, 1.0, seeds=(0, 1))
        assert RB.cmd_check(run)
        RB.cmd_run(run, run / "rank_bootstrap", B=20)
        r = json.loads((run / "rank_bootstrap" / "boot.json").read_text())
        assert r["complete"] and len(r["rows"]) == 1 and r["rows"][0]["seeds"] == [0, 1]
        assert 0 <= r["rows"][0]["p_diff"] <= 1 and r["rows"][0]["n_sequences"] == 8
        sel = G.selected(run)                                       # a per-sequence file that no longer matches
        out_dir = sel[(ds, "gru")]["interval"][1]["config"]["out_dir"]
        with np.load(f"{out_dir}/perseq_C.npz") as z:
            arrs = {k: z[k] for k in z.files}
        arrs["C"] = arrs["C"] + 1e-3
        np.savez(f"{out_dir}/perseq_C.npz", **arrs)
        assert not RB.cmd_check(run)
        RB.cmd_run(run, run / "rank_bootstrap", B=20)
        r = json.loads((run / "rank_bootstrap" / "boot.json").read_text())
        assert not r["complete"] and not r["rows"] and "does not reproduce" in r["failed"][0]
    finally:
        D.set_data_dir(old)
