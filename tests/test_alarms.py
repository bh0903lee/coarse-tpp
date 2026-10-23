"""SLA alarms (Supplementary S4): distribution function against the cell probabilities, the h grid, selection rules,
the bootstrap, the rollout path model, and a tiny end-to-end run on CPU."""
import json
import math
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from _toy import tied_toy, train_cells
from coarse_tpp import alarms as K, data as D, experiments as X, lik, readout as Ro


def rand_law(n, k=4, seed=0, mu_lo=-3.0, mu_hi=3.0, s_lo=0.2, s_hi=1.5):
    g = torch.Generator().manual_seed(seed)
    lw = torch.log_softmax(torch.randn(n, k, generator=g, dtype=torch.float64), -1)
    mu = mu_lo + (mu_hi - mu_lo) * torch.rand(n, k, generator=g, dtype=torch.float64)
    ls = torch.log(s_lo + (s_hi - s_lo) * torch.rand(n, k, generator=g, dtype=torch.float64))
    return lik.LogNormMix(lw, mu, ls)


@pytest.mark.parametrize("rule", ["raw", "smooth", "point_mass"])
@pytest.mark.parametrize("delta", [0.05, 0.5, 2.0])
def test_cell_probability_matches_exp_C(rule, delta):
    n = 64
    law = rand_law(n, seed=int(delta * 100) + len(rule))
    g = torch.Generator().manual_seed(7)
    k = torch.randint(0, 12, (n,), generator=g).double()
    k[:8] = 0                                                       # ties
    tau = k * delta
    tl = torch.randn(n, generator=g, dtype=torch.float64)
    pi = torch.sigmoid(tl)
    pC = torch.exp(Ro.C(law, tl, tau, delta, rule))
    for fn in (K.F_B, lambda *a: K.F_B_gl(*a, n=48, panels=16)):
        Fc = fn(law, pi, rule, delta, torch.stack([tau, tau + delta], -1))
        pc = Fc[:, 1] - Fc[:, 0]
        rel = ((pc - pC).abs() / pC.clamp(min=K.CELL_FLOOR)).max()
        assert float(rel) < 1e-6, (rule, delta, float(rel))


@pytest.mark.parametrize("rule", ["raw", "smooth", "point_mass"])
def test_cell_probabilities_sum_to_one(rule):
    delta, n = 0.25, 8
    law = rand_law(n, seed=3, mu_hi=1.0, s_hi=0.8)
    tl = torch.linspace(-2, 2, n, dtype=torch.float64)
    kmax = 4000
    lb = lik._broadcast_law(law, kmax)
    tau = (torch.arange(kmax, dtype=torch.float64) * delta).expand(n, kmax)
    tot = torch.exp(Ro.C(lb, tl.unsqueeze(-1).expand(n, kmax), tau, delta, rule)).sum(-1)
    FB = K.F_B(law, torch.sigmoid(tl), rule, delta, torch.full((n, 1), kmax * delta, dtype=torch.float64))[:, 0]
    assert torch.allclose(tot, FB, atol=1e-9) and float((1 - FB).abs().max()) < 1e-6


def test_closed_form_matches_gauss_legendre_and_sla_probability_is_the_survival():
    law = rand_law(32, seed=11)
    x = torch.rand(32, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(1)) * 10
    pi = torch.full((32,), 0.3, dtype=torch.float64)
    for rule in ("raw", "smooth", "point_mass"):
        a = K.F_B(law, pi, rule, 0.7, x); b = K.F_B_gl(law, pi, rule, 0.7, x, n=48, panels=16)
        assert float((a - b).abs().max()) < 1e-10
    delta, n = 0.5, 6
    law = rand_law(n, seed=5, mu_hi=1.0)
    tl = torch.zeros(n, dtype=torch.float64)
    P = dict(log_w=law.log_w, mu=law.mu, log_sigma=law.log_sigma, pi=torch.sigmoid(tl))
    kh = [0, 2, 5]
    p = K.sla_probs(P, "smooth", delta, kh)
    for j, k in enumerate(kh):      # p_h = P(g > h) = 1 - F(h + delta) = 1 - sum of the cell probabilities up to h
        below = sum(torch.exp(Ro.C(law, tl, torch.full((n,), i * delta, dtype=torch.float64), delta, "smooth")) for i in range(k + 1))
        assert np.allclose(p[:, j], 1 - below.numpy(), atol=1e-12)


def test_h_rounding_down_to_grid():
    delta = 1 / 3600
    gaps = [np.array([0.0, 0.0, 5 * delta, 7.5 * delta, 0.0, 10 * delta, 20 * delta, 30 * delta, 41 * delta])]
    hg = K.h_grid(SimpleNamespace(coarse=SimpleNamespace(iet=gaps)), delta)
    pos = gaps[0][1:][gaps[0][1:] > 0]
    for key, q in K.QUANTS.items():
        kh, h, raw = hg[key]
        assert raw == pytest.approx(float(np.quantile(pos, q)))
        assert kh == math.floor(raw / delta + 1e-6) and h == kh * delta and h <= raw + 1e-12
    assert hg["h0"][0] == 0
    g2 = [np.array([0.0] + [3 * delta * (1 - 1e-15)] * 5)]       # a grid multiple up to float noise stays on it
    assert K.h_grid(SimpleNamespace(coarse=SimpleNamespace(iet=g2)), delta)["q50"][0] == 3
    y, resid = K.outcomes(np.array([0.0, delta, 3 * delta, 4 * delta]), [0, 3], delta)
    assert y.tolist() == [[0, 0], [1, 0], [1, 0], [1, 1]] and resid < 1e-9


def _res(cid, arm, lr, seed, fl, C, ep_c=5, ep_own=5, ds="bpi_x"):
    return dict(cell_id=cid, arm=arm, seed=seed, config=dict(dataset=ds, encoder=dict(type="gru"), lr_tag=lr),
                dev={"floor_1e-6": fl, "C": C}, by_selection=dict(C=dict(selected_epoch=ep_c), own=dict(selected_epoch=ep_own)))


def _table(vals, seeds=K.PLANNED_SEEDS, tri=None, tri_seeds=K.PLANNED_SEEDS):
    """Complete synthetic unit: every planned (arm, lr) with `seeds`; vals {(arm, lr): (dev floor, dev C)} (default low)."""
    cells = []
    arms = list(K.BASE_ARMS) + ([K.TRI] if tri else [])
    for arm in arms:
        for lr in K.LRS:
            e6, c = (tri or {}).get((arm, lr)) or vals.get((arm, lr), (0.0, -5.0))
            for s in (tri_seeds if arm == K.TRI else seeds):
                cells.append(_res(f"{arm}_{lr}_{s}", arm, lr, s, e6 + 0.01 * s, c - 0.01 * s))
    return K.candidates(cells)[("bpi_x", "gru")]


def test_selection_rules():
    vals = {("floor_1e-6", "lr3e4"): (5.0, -3.0), ("floor_1e-6", "lr1e3"): (4.0, -2.9), ("interval", "lr3e4"): (1.0, -2.5),
            ("interval", "lr1e3"): (1.1, -2.6)}
    cands = _table(vals, tri={("triangular", "lr3e4"): (0.5, -2.4)})
    rules = K.make_rules(lambda r: None)
    assert K.choose(cands, rules["F"], False)[:2] == (("floor_1e-6", "lr3e4"), "C")
    assert K.choose(cands, rules["C"], False)[:2] == (("interval", "lr3e4"), "C")
    assert K.choose(cands, rules["C"], True)[:2] == (("triangular", "lr3e4"), "C")
    k, sel, means, bad = K.choose(cands, rules["F_own"], False)      # no own scores: the floor candidates are unrankable
    assert k is None and set(bad) == {("floor_1e-6", "lr3e4"), ("floor_1e-6", "lr1e3")}
    assert K.same_model(cands, (("floor_1e-6", "lr3e4"), "own"), (("floor_1e-6", "lr3e4"), "C"))
    assert not K.same_model(cands, (("floor_1e-6", "lr3e4"), "C"), (("interval", "lr3e4"), "C"))
    # F_own: only the floor candidate moves to its own checkpoint; the others keep best_C and the stored dev floor
    vals = {("floor_1e-6", "lr3e4"): (5.0, -3.0), ("interval", "lr3e4"): (6.0, -2.5)}
    cands = _table(vals)
    own = {"floor_1e-6": 5.5, "interval": 5.0}
    own_lookup = lambda r: own.get(r["arm"], 0.0) if r["config"]["lr_tag"] == "lr3e4" else -1.0
    k, sel, means, bad = K.choose(cands, K.make_rules(own_lookup)["F_own"], False)
    assert (k, sel) == (("interval", "lr3e4"), "C") and not bad
    assert means[("floor_1e-6", "lr3e4")] == pytest.approx(5.5) and means[("interval", "lr3e4")] == pytest.approx(6.02)
    own["floor_1e-6"] = 7.0
    assert K.choose(cands, K.make_rules(own_lookup)["F_own"], False)[:2] == (("floor_1e-6", "lr3e4"), "own")


def test_incomplete_units_are_pending(tmp_path):
    vals = {("floor_1e-6", "lr3e4"): (5.0, -3.0)}
    cands = _table(vals, tri={("triangular", "lr3e4"): (9.0, -1.0), ("triangular", "lr1e3"): (9.0, -1.0)}, tri_seeds=(0,))
    ch = K.all_choices({("bpi_x", "gru"): cands}, tmp_path, own_lookup=lambda r: None)
    wt = ch[(("bpi_x", "gru"), "with_tri")]
    assert not wt["complete"] and wt["rules"] is None and any("triangular" in m and "[1, 2, 3, 4]" in m for m in wt["missing"])
    nt = ch[(("bpi_x", "gru"), "no_tri")]
    assert nt["complete"] and nt["rules"]["C"][0][0] != K.TRI and nt["rules"]["F"][0] == ("floor_1e-6", "lr3e4")
    del cands[("jitter_gap", "lr1e3")][3]
    ok, missing = K.unit_status(cands, False)
    assert not ok and missing == ["jitter_gap lr1e3: seeds [3] missing"]
    assert K.variants_of({k: v for k, v in cands.items() if k[0] != K.TRI}) == ("no_tri",)


def _write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(obj))


def test_rt_requires_a_verified_checkpoint_and_quarantine_is_not_a_cache(tmp_path):
    seeds_res = {s: dict(cell_id=f"c{s}") for s in K.PLANNED_SEEDS}
    rt = dict(version=K.VERSION, complete=True, mae_median=123.0)
    for s in K.PLANNED_SEEDS:
        q = s == 2
        _write(tmp_path / "cells" / f"c{s}.json", dict(version=K.VERSION, quarantined_sels=["C"] if q else [],
                                                        by_sel=dict(C=dict(missing=False, quarantined=q, C_absdiff=1.0 if q else 0.0))))
        _write(tmp_path / "rt" / f"c{s}__C.json", rt)
    rows, pend = K.rt_collect(tmp_path, seeds_res, "C")
    assert len(rows) == 4 and pend == ["s2: quarantined"]
    assert K.verification(tmp_path, "c2", "C") == "quarantined" and K.verification(tmp_path, "c0", "C") == "verified"
    assert K.verification(tmp_path, "c0", "own") == "unverified" and K.verification(tmp_path, "zz", "C") == "unverified"
    _write(tmp_path / "cells" / "c0.json", dict(version=K.VERSION - 1, quarantined_sels=[], by_sel=dict(C=dict(missing=False, quarantined=False))))
    assert K.verification(tmp_path, "c0", "C") == "unverified" and K.sla_cache_state(tmp_path, "c0") == "stale"
    assert K.sla_cache_state(tmp_path, "c2") == "quarantined" and K.sla_cache_state(tmp_path, "c1") == "valid"
    assert K.sla_cache_state(tmp_path, "zz") == "absent"


def test_bootstrap_deterministic_and_correct_point():
    rng = np.random.default_rng(0)
    n = rng.integers(1, 20, 50).astype(float)
    A_ = [rng.random(50) * n for _ in range(5)]
    B_ = [a + 0.1 * n for a in A_]                                   # +0.1 per event in every sequence
    p1, lo1, hi1 = K.boot_diff(A_, B_, n, n_boot=500)
    assert float(p1) == pytest.approx(0.1) and float(lo1) == pytest.approx(0.1) and float(hi1) == pytest.approx(0.1)
    assert (p1, lo1, hi1) == K.boot_diff(A_, B_, n, n_boot=500)
    C_ = [rng.random(50) * n for _ in range(5)]
    q1 = K.boot_diff(A_, C_, n, n_boot=500); q2 = K.boot_diff(A_, C_, n, n_boot=500); q3 = K.boot_diff(A_, C_, n, n_boot=500, seed=1)
    assert all(np.array_equal(a, b) for a, b in zip(q1, q2)) and float(q1[1]) != float(q3[1])
    assert float(q1[1]) < float(q1[0]) < float(q1[2])


def test_metrics_cost_and_auc():
    p = np.array([[0.9], [0.1], [0.3], [0.6]]); y = np.array([[1.0], [0.0], [1.0], [0.0]])
    brier, logs, cost = K.event_terms(p, y)
    assert np.allclose(brier[:, 0], [0.01, 0.01, 0.49, 0.36])
    assert cost[:, 0, 0].tolist() == [0, 0, 1, 1]        # c = 1: alarm iff p > 0.5
    assert cost[:, 1, 0].tolist() == [0, 0, 0, 1]        # c = 4: alarm iff p > 0.2
    assert K.auc(p[:, 0], y[:, 0]) == pytest.approx(0.75) and K.auc(p[:, 0], np.ones(4)) is None


def test_markov_chain_and_paths():
    marks = [np.array([0, 1, 2]), np.array([0, 2])]
    T = K.markov_chain(SimpleNamespace(marks=marks), 4)
    assert T.shape == (4, 5) and np.allclose(T.sum(1), 1) and T[2, 4] == 1.0 and T[0, 1] == 0.5 and T[0, 2] == 0.5
    paths, trunc = K.sample_paths(T, 0, 50, 10, np.random.default_rng(0))
    assert all(len(p) in (1, 2) for p in paths) and not trunc.any() and all(p[-1] == 2 for p in paths)
    T2 = np.zeros((2, 3)); T2[:, 0] = 1.0                           # never ends
    paths, trunc = K.sample_paths(T2, 0, 5, 7, np.random.default_rng(0))
    assert trunc.all() and all(len(p) == 7 for p in paths)


def test_cell_check_passes_the_right_readout_and_fails_the_wrong_one():
    delta, n = 0.01, 200
    law = rand_law(n, seed=21, mu_lo=-1.0, mu_hi=2.0)
    g = torch.Generator().manual_seed(2)
    tau = torch.randint(0, 300, (n,), generator=g).double() * delta
    tl = torch.randn(n, generator=g, dtype=torch.float64)
    P = dict(log_w=law.log_w, mu=law.mu, log_sigma=law.log_sigma, pi=torch.sigmoid(tl), tau32=tau)
    for rule in ("raw", "smooth", "point_mass"):
        P["C"] = Ro.C(law, tl, tau, delta, rule)
        assert K.cell_check(P, rule, delta)["cell_ok"]
        wrong = {"raw": "smooth", "smooth": "raw", "point_mass": "smooth"}[rule]
        assert not K.cell_check(P, wrong, delta)["cell_ok"]


def test_end_to_end_on_a_tiny_process_log(tmp_path, monkeypatch):
    old = D.data_dir()
    D.set_data_dir(tmp_path)
    try:
        ds = tied_toy(tmp_path, name="bpi_toy", n_types=3, bpi=True)
        run = tmp_path / "run"
        monkeypatch.setattr(K, "BASE_ARMS", ("floor_1e-6", "interval"))
        monkeypatch.setattr(K, "LRS", ("lr1e3",))
        monkeypatch.setattr(K, "PLANNED_SEEDS", (0,))
        monkeypatch.setattr(K, "N_BOOT", 50)
        train_cells("process_logs", run, ds, K.BASE_ARMS, 1.0, encoder=dict(X.GRU, marks=True))
        a = SimpleNamespace(run=str(run), base=None, tasks="sla,rt", datasets=None, only=None, limit=None, selected=False,
                            force=False, retry_quarantined=False, cpu=True, threads=2, rt_max_cases=2, rt_all=True, out=None)
        K.cmd_score(a)
        out = run / "alarms"
        recs = [json.loads(p.read_text()) for p in sorted((out / "cells").glob("*.json"))]
        assert len(recs) == 2 and all(not r["quarantined_sels"] for r in recs)
        for r in recs:
            for sel in K.SELS:
                b = r["by_sel"][sel]
                assert b["C_absdiff"] < 1e-8 and b["cell_ok"] and 0 <= b["brier_mean"] <= 1
        L = K.cmd_report(a)
        S = json.loads((out / "summary.json").read_text())
        assert S["n_scored"] == 2 and not S["quarantined"] and len(S["comparisons"]) >= 1
        comp = S["comparisons"][0]
        assert comp["boot95"][0] <= comp["diff"] <= comp["boot95"][1] or comp["same_model"]
        assert any("## 4. Remaining time" in x for x in L) and (out / "rt").glob("*.json")
    finally:
        D.set_data_dir(old)
