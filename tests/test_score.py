"""The audit rule and the model-agnostic interval score."""
import csv
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from coarse_tpp import lik, score as S

ROOT = Path(__file__).resolve().parents[1]


def test_detect_resolution_finds_a_lattice_and_ties():
    rng = np.random.default_rng(0)
    g = np.round(rng.exponential(30.0, 20000))                     # integer seconds with ties
    out = S.detect_resolution(np.where(g == 0, 1e-6, g))
    assert out["delta"] == 1.0 and out["delta_rule"] == "(ii) lattice" and out["tie_share"] > 0.01
    c = S.detect_resolution(rng.exponential(1.0, 5000))            # continuous
    assert c["delta"] is None and c["delta_rule"] == "(iv) undefined"
    k = S.detect_resolution(rng.exponential(1.0, 5000), constructed=0.1)
    assert k["delta"] == 0.1 and k["delta_rule"] == "(i) constructed"
    h = np.concatenate([np.full(400, 0.01), 0.1 + rng.exponential(1.0, 1000)])   # spike at the floor value, then a gap
    assert S.detect_resolution(h)["hidden_flag"]


def test_ties_rule_when_no_lattice():
    rng = np.random.default_rng(1)
    g = np.concatenate([np.zeros(300), 0.37 + rng.exponential(1.0, 9700) * np.pi])
    out = S.detect_resolution(g)
    if out["delta_hat"] is None or out["delta_hat_rel_median"] < 1e-6:
        assert out["delta_rule"] == "(iii) ties, min positive gap" and out["delta"] == out["min_pos_gap"]


def test_interval_score_from_cdf_agrees_with_the_mixture_law():
    torch.manual_seed(0)
    K, n = 3, 400
    lw = torch.log_softmax(torch.randn(n, K), -1); mu = torch.randn(n, K); ls = torch.randn(n, K) * 0.3
    law = lik.LogNormMix(lw, mu, ls)
    tau = torch.rand(n).double() * 3
    tau[:50] = 1e-6                                                    # stored ties
    delta = 0.25
    ref, tie = S.interval_loglik(law, tau, delta)
    g = torch.where(tau <= S.TIE_STORED, torch.zeros_like(tau), tau)
    F_lo = torch.where(g > 0, law.log_cdf(g).exp(), torch.zeros_like(g)).numpy()
    F_hi = law.log_cdf(g + delta).exp().numpy()
    ll, tie2 = S.interval_loglik_from_cdf(tau.numpy(), F_lo, F_hi)
    assert np.allclose(ll, ref, atol=1e-6) and (tie == tie2).all() and tie.sum() == 50
    ll_small, _ = S.interval_loglik(law, tau, 1e-6)                    # tiny delta: density + log delta
    dens, _ = S.floor_loglik(law, tau, 1e-6)
    assert np.allclose(ll_small[~tie], dens[~tie] + np.log(1e-6), atol=1e-3)
    s = S.summarize(ll, tie)
    assert s["n"] == n and s["n_tie"] == 50 and s["n_nonfinite"] == 0 and s["mean_tie"] <= 0


def test_cli_audit_and_score(tmp_path):
    rng = np.random.default_rng(1)
    seqs = []
    for _ in range(30):
        dt = np.round(rng.exponential(2.0, 20)); dt[0] = 0.0
        dt = np.where(dt == 0.0, 1e-6, dt)
        seqs.append([dict(time_since_start=float(t), time_since_last_event=float(g), type_event=0) for t, g in zip(np.cumsum(dt), dt)])
    p = tmp_path / "toy.pkl"
    pickle.dump({"train": seqs, "dim_process": 1}, open(p, "wb"))
    run = lambda *a: json.loads(subprocess.run([sys.executable, str(ROOT / "scripts" / "score.py"), *a], capture_output=True,
                                               text=True, check=True).stdout)
    out = run("audit", "--pkl", str(p))
    assert out["delta"] == 1.0 and out["tie_share"] > 0
    c = tmp_path / "pred.csv"
    with open(c, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["tau", "F_lo", "F_hi"])
        w.writerow([1e-6, 0.0, 0.2]); w.writerow([3.0, 0.6, 0.7]); w.writerow([5.0, 0.9, 0.95])
    out = run("score", "--csv", str(c), "--delta", "1")
    assert out["n"] == 3 and out["n_tie"] == 1 and abs(out["mean"] - np.mean(np.log([0.2, 0.1, 0.05]))) < 1e-9
