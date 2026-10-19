"""Predictive law q_B of the waiting time, cell CRPS, and the re-evaluation of saved models (coarse_tpp.predictive)."""
import json
import math
import pickle

import numpy as np
import pytest
import torch

from coarse_tpp import data as D, experiments as X, lik, predictive as PR, readout as Ro, train as TR

T = lambda *a: torch.tensor(a, dtype=torch.float64)
Z = lambda n: torch.zeros(n, dtype=torch.float64)


def mix(n, w, mu, sig):
    return lik.LogNormMix(torch.log(T(*w)).expand(n, len(w)), T(*mu).expand(n, len(w)), torch.log(T(*sig)).expand(n, len(w)))


@pytest.mark.parametrize("k", [0, 3, 40, 5000])
def test_crps_of_a_point_law_is_zero_at_its_own_cell(k):
    d = 0.5
    law = mix(1, [1.0], [math.log((k + 0.5) * d)], [1e-9])
    crps, _, lc = PR.cell_crps(law, Z(1), "raw", d, T(k * d), n0=64)
    assert float(crps) < 1e-10
    assert abs(float(lc)) < 1e-8                              # all mass in the recorded cell
    if k + 3 < 64:            # inside the exact region (beyond it the midpoint-rule remainder does not resolve a step)
        other, _, _ = PR.cell_crps(law, Z(1), "raw", d, T((k + 2) * d), n0=64)
        assert abs(float(other) - 2 * d) < 1e-8               # two cells away: CRPS = 2 delta


def test_crps_matches_the_brute_force_sum_and_truncation():
    d = 1.0
    law = mix(2, [0.4, 0.6], [math.log(30.0), math.log(400.0)], [0.6, 0.3])
    tau = T(12.0, 700.0)
    approx, tw, _ = PR.cell_crps(law, Z(2), "raw", d, tau, n0=16, panels=256, cap=500.0)
    j = torch.arange(200000, dtype=torch.float64)
    G = PR.cdf(law, Z(2), "raw", d, j.unsqueeze(0).expand(2, -1) * d + d)     # u_j = (j + 1) delta here (r = 0)
    terms = (G - (j.unsqueeze(0) >= (tau / d).unsqueeze(-1)).double()) ** 2
    assert torch.allclose(approx, d * terms.sum(-1), rtol=1e-4)
    assert torch.allclose(tw, d * (terms * ((j + 1) * d <= 500.0).double()).sum(-1), rtol=1e-4)


def test_mean_of_a_known_mixture_under_each_readout():
    law = mix(1, [0.3, 0.7], [0.0, 1.0], [0.5, 0.2])
    m = 0.3 * math.exp(0.125) + 0.7 * math.exp(1.0 + 0.02)
    pi = T(0.25)
    assert abs(float(PR.mean(law, pi, "raw", 0.1)) - m) < 1e-12
    assert abs(float(PR.mean(law, pi, "smooth", 0.1)) - (m + 0.05)) < 1e-12
    assert abs(float(PR.mean(law, pi, "point_mass", 0.1)) - (0.25 * 0.05 + 0.75 * (m + 0.05))) < 1e-12


def test_median_by_bisection():
    law = mix(1, [1.0], [1.3], [0.7])
    assert abs(float(PR.median(law, Z(1), "raw", 0.01)) - math.exp(1.3)) < 1e-9


def test_smooth_cdf_is_the_average_of_Q_over_the_last_cell():
    d = 0.3
    law = mix(1, [0.3, 0.7], [-1.0, 0.5], [0.4, 1.2])
    for x in (0.05, 0.3, 0.31, 2.0, 40.0):
        u = torch.linspace(max(x - d, 0.0), x, 200001, dtype=torch.float64)
        num = float(torch.trapezoid(PR.q_cdf(law, u.unsqueeze(0)).squeeze(0), u)) / d
        assert abs(float(PR.smooth_cdf(law, torch.tensor([[x]], dtype=torch.float64), d)) - num) < 1e-8


@pytest.mark.parametrize("rule", ["raw", "smooth", "point_mass"])
def test_cell_probability_from_F_equals_the_score_C(rule):
    d = 0.2
    law = mix(5, [0.3, 0.7], [-1.0, 0.5], [0.4, 1.2])
    tau = T(0.0, 0.2, 0.4, 1.0, 3.0)
    tl = T(-1.0, 0.3, 0.0, -2.0, 1.0)
    _, _, lc = PR.cell_crps(law, torch.sigmoid(tl), rule, d, tau, n0=32)
    assert torch.allclose(lc, Ro.C(law, tl, tau, d, rule), atol=1e-8)


@pytest.fixture()
def toy_run(tmp_path):
    """Two treatments x two seeds of a tiny tied data set, trained for two epochs on CPU."""
    old = D.data_dir()
    D.set_data_dir(tmp_path / "data")
    rng = np.random.default_rng(0)
    for sp, n in (("train", 40), ("dev", 10), ("test", 10)):
        seqs = []
        for _ in range(n):
            g = np.round(rng.exponential(2.0, 30)); g[0] = 0.0
            seqs.append([dict(time_since_start=float(t), time_since_last_event=float(max(x, 1e-6)), type_event=0)
                         for t, x in zip(np.cumsum(g), g)])
        (tmp_path / "data" / "toy").mkdir(parents=True, exist_ok=True)
        pickle.dump({sp: seqs, "dim_process": 1}, open(tmp_path / "data" / "toy" / f"{sp}.pkl", "wb"))
    run = tmp_path / "run"
    cs = []
    for arm in ("floor_1e-6", "interval"):
        for s in (0, 1):
            c = X.cell("rank_change", run, "toy", arm, s, "lr1e3", 1.0)
            c.update(optim=dict(batch_size=32, max_epochs=2, patience=20, grad_clip=1.0, lr=1e-3), cpu=True)
            TR.run(c)
            cs.append(c)
    X.write("rank_change", run, cells=cs)
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps([dict(dataset="toy", tie_share=0.2, delta=1.0)]))
    yield run, audit
    D.set_data_dir(old)


def test_score_and_report_of_a_small_run(toy_run, tmp_path):
    run, audit = toy_run
    PR.main(["score", str(run)])
    rows = [json.loads(p.read_text()) for p in sorted((run / "predictive" / "cells").glob("*.json"))]
    assert len(rows) == 4 and not any(r["quarantined"] for r in rows)
    assert all(r["C_absdiff"] < 1e-6 and r["n_events"] > 0 for r in rows)
    assert all(0 < r["twcrps"] <= r["crps"] and math.isfinite(r["mae_median"]) for r in rows)
    PR.report(run, audit, arms=("floor_1e-6", "interval"), n_seeds=2)
    (summary,) = json.loads((run / "predictive" / "summary.json").read_text())
    assert summary["tied"] and summary["floor_top"] in ("floor_1e-6", "interval")
    if summary["changed"]:
        ref = [summary["by_treatment"][summary["C_top"]]["twcrps"] - summary["by_treatment"][summary["floor_top"]]["twcrps"]]
        assert abs(summary["d_twcrps"] - ref[0]) < 1e-9
    assert abs(summary["tau_floor_C"]) == 1.0                     # two treatments: tau is +-1
