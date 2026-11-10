"""Jitter and point-mass scores (Online Resource 1, Section S5): analytic checks of the per-event scores, the pure
analysis functions, and a tiny end-to-end run on CPU."""
import json
import math

import numpy as np
import pytest
import torch

from _toy import tied_toy, train_cells
from coarse_tpp import data as D, jitter_pm as J, lik, readout as Ro


class Flat:
    """Uniform law on [0, L): a density that is flat on every cell inside [0, L)."""

    def __init__(self, L):
        self.L = L

    def log_density(self, t):
        t = t.double()
        return torch.where((t >= 0) & (t < self.L), torch.full_like(t, -math.log(self.L)), torch.full_like(t, -math.inf))

    def log_interval(self, a, b):
        return torch.log((torch.clamp(b, max=self.L) - a) / self.L)


class Expo:
    """Exponential law with rate lam: log q(s) = log lam - lam s is linear in s."""

    def __init__(self, lam):
        self.lam = lam

    def log_density(self, t):
        return math.log(self.lam) - self.lam * t.double()

    def log_interval(self, a, b):
        return -self.lam * a.double() + torch.log1p(-torch.exp(-self.lam * (b - a).double()))


def mixture(n, K=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    lw = torch.log_softmax(torch.randn(n, K, generator=g, dtype=torch.float64), -1)
    return lik.LogNormMix(lw, torch.randn(n, K, generator=g, dtype=torch.float64) * 2,
                          torch.randn(n, K, generator=g, dtype=torch.float64) - 0.5)


def test_jitter_is_exact_for_a_log_linear_density():
    tau, delta, lam = torch.tensor([0.0, 1.0, 3.0, 7.5], dtype=torch.float64), 0.5, 1.7
    got = J.jitter_loglik(Expo(lam), tau, delta)
    assert torch.allclose(got, math.log(lam) - lam * (tau + delta / 2), atol=1e-12)


def test_jitter_bound_and_equality_for_a_flat_density():
    tau, delta = torch.tensor([0.0, 1.0, 2.0, 5.0]), 1.0
    for law in (mixture(4), Expo(0.3)):
        jit = J.jitter_loglik(law, tau, delta)
        ivl = law.log_interval(tau.double(), tau.double() + delta)
        assert torch.all(jit <= ivl - math.log(delta) + 1e-12)          # Jensen: E log q <= log E q
    flat = Flat(10.0)
    jit = J.jitter_loglik(flat, tau, 2.0)
    assert torch.allclose(jit, flat.log_interval(tau.double(), tau.double() + 2.0) - math.log(2.0), atol=1e-12)


def test_jitter_matches_a_fine_quadrature():
    law, tau, delta = mixture(5, seed=1), torch.tensor([0.0, 0.5, 1.0, 2.0, 4.0]), 1.0
    x, w = np.polynomial.legendre.leggauss(2000)
    u = torch.tensor((x + 1) / 2); w = torch.tensor(w / 2)
    ref = sum(wk * law.log_density(tau.double() + delta * uk) for uk, wk in zip(u, w))
    got = J.jitter_loglik(law, tau, delta)
    assert torch.allclose(got[1:], ref[1:], atol=1e-6)
    # at a tie the log density of a log-normal component is unbounded near 0: the 32-node rule is less accurate there
    assert abs(float(got[0] - ref[0])) < 1e-2


def test_point_mass_score_is_the_interval_tie_term_on_ties_and_a_density_elsewhere():
    law, delta = mixture(6, seed=2), 1.0
    tau = torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0, 0.0], dtype=torch.float64)
    tie = Ro.tie_mask(tau, delta)
    tl = torch.linspace(-2, 2, 6, dtype=torch.float64)
    dens = law.log_density(torch.where(tie, torch.ones_like(tau), tau))
    for rule in ("raw", "smooth", "point_mass"):
        pm = J.point_mass_loglik(law, tl, tau, delta, rule)
        c = Ro.C(law, tl, tau, delta, rule)
        assert torch.allclose(pm[tie], c[tie])
        d = dens + torch.nn.functional.logsigmoid(-tl) if rule == "point_mass" else dens
        assert torch.allclose(pm[~tie], d[~tie])
    # raw readout: the tie term is the probability of the first cell, log Q(delta)
    pm = J.point_mass_loglik(law, None, tau, delta, "raw")
    assert torch.allclose(pm[tie], law.log_cdf(torch.full_like(tau, delta))[tie])
    # point-mass readout: log(pi + (1 - pi) P(first cell)) with the triangular cell of the smooth readout
    p = torch.sigmoid(tl[tie])
    first = torch.exp(lik.event_loglik(law, tau, "triangular", delta=delta))[tie]
    assert torch.allclose(J.point_mass_loglik(law, tl, tau, delta, "point_mass")[tie], torch.log(p + (1 - p) * first))


def fake_cells(means, lrs=("lr1e3",), seeds=(0, 1, 2), ds="d", enc="gru"):
    """Result-like dicts; means[arm][score] = (dev, test) values, seed s adds 1e-3 s to the test value."""
    out = []
    for arm, sc in means.items():
        for lr in lrs:
            for s in seeds:
                v = {k: (d, t + 1e-3 * s) for k, (d, t) in sc.items()}
                out.append(dict(cell_id=f"{arm}_{lr}_{s}", arm=arm, seed=s, _v=v,
                                config=dict(dataset=ds, encoder=dict(type=enc), head="mix", K=8, lr_tag=lr)))
    return out


def fake_value(r, score, split):
    v = r["_v"].get(score)
    return None if v is None else v[0 if split == "dev" else 1]


def test_rankings_tops_deficit_and_self_preference():
    arms = ("floor_1e-6", "jitter_gap", "point_mass", "interval")
    C = {"floor_1e-6": -2.0, "jitter_gap": -1.1, "point_mass": -1.5, "interval": -1.0}
    jit = {"floor_1e-6": -3.0, "jitter_gap": -2.0, "point_mass": -2.6, "interval": -2.1}
    means = {a: {"C": (C[a], C[a]), "jitter": (jit[a], jit[a])} for a in arms}
    cells = fake_cells(means)
    iu = J.units(cells, fake_value, "C", "C", arms)
    su = J.units(cells, fake_value, "jitter", "jitter", arms)
    rows = J.compare(su, iu, arms, lambda g: 0.2, "jitter_gap")
    r = rows[0]
    assert r["top"] == "jitter_gap" and r["top_interval"] == "interval" and r["changed"] and r["self_top"]
    assert math.isclose(r["deficit"], 0.1) and r["rank_by_interval"] == 2 and r["seeds_positive"] == 3
    assert math.isclose(r["tau"], 2 / 3)                         # one discordant pair of six
    s = J.summarize(rows)
    assert s["changed_tied"] == 1 and s["self_top_tied"] == 1 and s["ranks_changed_tied"] == [2]


def test_learning_rate_is_chosen_by_the_validation_value():
    means = {"interval": {"C": (-1.0, -1.0)}, "floor_1e-6": {"C": (-2.0, -2.0)}}
    cells = fake_cells(means, lrs=("lr1e3",))
    worse = fake_cells({"interval": {"C": (-0.5, -9.0)}}, lrs=("lr3e4",))            # better dev, worse test
    u = J.units(cells + worse, fake_value, "C", "C", ("interval", "floor_1e-6"))
    assert u[("d", "gru", "mix", 8)]["interval"]["lr"] == "lr3e4"
    assert math.isclose(u[("d", "gru", "mix", 8)]["interval"]["mean"], -9.0 + 1e-3)


def test_bootstrap_counts():
    rng = np.random.default_rng(0)
    n = rng.integers(1, 20, 30).astype(float)
    arms = ["a", "b"]
    C = {"a": np.stack([n * -1.0, n * -1.0]), "b": np.stack([n * -2.0, n * -2.0])}
    same = J.bootstrap_score(n, C, C, "a", arms, B=50)
    assert same == dict(p_diff=0.0, p_self=1.0)
    rev = {"a": C["b"], "b": C["a"]}
    assert J.bootstrap_score(n, rev, C, "a", arms, B=50) == dict(p_diff=1.0, p_self=0.0)


def test_reference_rows_against_B():
    arms = ("floor_1e-6", "jitter_gap", "interval")
    scores = ("floor_1e-6", "C", "jitter")
    d = {"floor_1e-6": dict(B=-3.0, **{"floor_1e-6": 5.0, "C": -2.0, "jitter": -3.0}),
         "jitter_gap": dict(B=-1.005, **{"floor_1e-6": 1.0, "C": -1.1, "jitter": -1.0}),
         "interval": dict(B=-1.0, **{"floor_1e-6": 0.0, "C": -1.0, "jitter": -1.2})}
    rows = J.reference_rows({("P", 0.1): d}, arms, scores)
    sm = J.reference_summary(rows, scores, arms)
    assert rows[0]["top_B"] == "interval"
    assert sm["C"]["top_as_B"] == 1 and sm["C"]["within"] == 1 and math.isclose(sm["C"]["median_tau"], 1.0)
    assert rows[0]["jitter"]["top"] == "jitter_gap" and sm["jitter"]["within"] == 1 and sm["jitter"]["self_top"] == 1
    assert math.isclose(sm["floor_1e-6"]["mean_shortfall"], 2.0) and sm["floor_1e-6"]["self_top"] == 1


def test_choose_and_brier_comparison():
    cands = {("x", "lr1e3"): {0: 1.0, 1: 3.0}, ("y", "lr1e3"): {0: 2.5, 1: 2.0}}
    key, means = J.choose(cands, lambda v: v, list(cands), (0, 1))
    assert key == ("y", "lr1e3") and means[("x", "lr1e3")] == 2.0
    assert J.choose(cands, lambda v: None, list(cands), (0, 1))[0] is None
    n = np.array([2.0, 2.0])
    data = {("x", "lr1e3"): (n, [np.array([0.2, 0.2]), np.array([0.2, 0.2])]),
            ("y", "lr1e3"): (n, [np.array([0.4, 0.4]), np.array([0.4, 0.4])])}
    boot = lambda a, b, nn: (np.mean([y.sum() - x.sum() for x, y in zip(a, b)]) / nn.sum(), -1.0, 1.0)
    c = J.brier_comparison(data.get, ("x", "lr1e3"), ("y", "lr1e3"), boot)
    assert math.isclose(c["diff"], 0.1) and math.isclose(c["rel"], -0.5)           # score selection: half the loss
    assert J.brier_comparison(data.get, ("x", "lr1e3"), ("x", "lr1e3"), boot) == dict(same=True)


def test_predictive_rows(monkeypatch):
    monkeypatch.setattr(J, "RANK_ARMS", ("floor_1e-6", "jitter_gap", "point_mass", "interval"))
    vals = {"floor_1e-6": (5.0, -2.0, -3.0, -2.0, 1.2), "jitter_gap": (1.0, -1.1, -1.0, -1.1, 1.05),
            "point_mass": (0.0, -1.5, -2.0, 0.5, 1.1), "interval": (0.0, -1.0, -1.2, -1.0, 1.0)}
    G = {a: {s: {"floor_1e-6": f6, "C": c, "jitter": j, "point_mass": pm, "twcrps": tw + 0.01 * s, "mae_median": tw}
                 for s in range(3)} for a, (f6, c, j, pm, tw) in vals.items()}
    r = J.predictive_rows({("d", "gru"): G})[("d", "gru")]
    assert r["top"] == {"floor_1e-6": "floor_1e-6", "C": "interval", "jitter": "jitter_gap", "point_mass": "point_mass"}
    assert math.isclose(r["jitter"]["twcrps"]["pct"], 100 * (-0.05) / 1.06) and r["jitter"]["changed"]
    assert math.isclose(r["C"]["tau_twcrps"], 1.0)


def test_end_to_end_on_tiny_cells(tmp_path, monkeypatch):
    old = D.data_dir()
    D.set_data_dir(tmp_path)
    try:
        ds = tied_toy(tmp_path)
        run = tmp_path / "run"
        arms = ("floor_1e-6", "jitter_gap", "point_mass", "interval")
        monkeypatch.setattr(J, "RANK_ARMS", arms)
        train_cells("rank_change", run, ds, arms, 1.0, seeds=(0, 1))
        J.cmd_score(run, cpu=True)
        files = sorted((run / "jitter_pm" / "cells").glob("*.json"))
        assert len(files) == 8
        for p in files:
            r = json.loads(p.read_text())
            assert not r["quarantined"] and r["test"]["n_ties"] > 0
            assert abs(r["test"]["d_C"]) < 1e-6 and abs(r["dev"]["d_floor_1e-6"]) < 1e-6
            t = r["test"]
            n, nt = t["n_events"], t["n_ties"]
            assert math.isclose(t["point_mass"], (nt * t["point_mass_tie"] + (n - nt) * t["point_mass_nontie"]) / n, rel_tol=1e-9)
            assert math.isclose(t["point_mass_tie"], t["C_tie"], rel_tol=1e-12)          # ties: the interval tie term
            if r["treatment"] in ("floor", "interval", "jitter_gap"):                    # raw readout: Jensen bound
                assert t["jitter"] <= t["C"] - t["log_delta"] + 1e-12
            with np.load(p.with_suffix(".npz")) as z:
                assert math.isclose(z["jitter"].sum() / z["n"].sum(), t["jitter"], rel_tol=1e-12)
        assert J.cache_state(files[0], run / "cells" / files[0].stem / "best_C.pt") == "ok"
        ns = type("A", (), dict(out=str(tmp_path / "rep"), audit=str(tmp_path / "none.json"), B=20, rank_change=str(run),
                                intensity_scores=None, intensity_jitter=None, intensity_repro=None, exceedance=None,
                                **{k: None for k in ("rank_change_seeds", "wider_encoders", "mixture_components", "density_heads",
                                                     "resplit", "new_datasets", "process_logs", "synthetic", "coarsened_real",
                                                     "coarsened_more")}))
        L = J.cmd_report(ns)
        S = json.loads((tmp_path / "rep" / "summary.json").read_text())
        sm = S["rank_change"]["summary"]
        assert sm["C"]["n"] == 1 and sm["C"]["changed"] == 0 and sm["jitter"]["n"] == 1
        assert set(S["rank_change"]["bootstrap"]) and not S["rank_change"]["bootstrap_left_out"]
        assert S["decomposition"]["rows"][0]["KL_C_smallest"] in arms
        assert any("Bootstrap" in x for x in L)
        ck = run / "cells" / files[0].stem / "best_C.pt"
        ck.write_bytes(ck.read_bytes() + b"\0")                                   # a changed checkpoint makes the cache stale
        assert J.cache_state(files[0], ck) == "stale"
    finally:
        D.set_data_dir(old)


def test_model_agnostic_wrappers():
    from coarse_tpp import score as S
    law = mixture(4, seed=3)
    tau = np.array([1e-6, 1.0, 2.0, 1e-6])                    # stored ties (1e-6) are read as recorded zero gaps
    ll, tie = S.jitter_loglik(law, tau, 1.0)
    assert tie.tolist() == [True, False, False, True]
    ivl, _ = S.interval_loglik(law, tau, 1.0)
    assert np.all(ll <= ivl + 1e-12)
    assert np.allclose(ll, J.jitter_loglik(law, torch.tensor([0.0, 1.0, 2.0, 0.0]), 1.0).numpy())
    pm = S.point_mass_loglik(ivl, np.array([0.0, -1.0, -2.0, 0.0]), tie, log1m_pi=np.full(4, -0.1))
    assert np.allclose(pm, [ivl[0], -1.1, -2.1, ivl[3]])
    with pytest.raises(ValueError):
        S.jitter_loglik(law, tau, 0.0)
