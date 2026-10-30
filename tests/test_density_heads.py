"""Weibull-mixture and spline-flow heads (Supplementary Table S8): laws, heads, grid, training and report."""
import json
import math

import numpy as np
import pytest
import torch
from scipy import stats as st

from coarse_tpp import analysis as A, data as D, experiments as X, lik
from coarse_tpp.data import Stats
from coarse_tpp.models import FLOW_LO, Model, GRUEncoder, build

from _toy import tied_toy, train_cells


def weib_law(B=5, K=3):
    g = torch.Generator().manual_seed(1)
    return lik.WeibullMix(torch.log_softmax(torch.randn(B, K, generator=g), -1), torch.randn(B, K, generator=g),
                          0.8 * torch.randn(B, K, generator=g))


def flow_law(B=5, K=8, bound=6.0, m=0.3, s=1.7):
    g = torch.Generator().manual_seed(2)
    w, h, d = lik.rqs_params(torch.randn(B, 3 * K - 1, generator=g), K)
    return lik.SplineFlow(w, h, d, m, s, bound)


def test_weibull_mixture_matches_scipy():
    law = weib_law()
    t = torch.tensor([1e-8, 0.03, 0.5, 2.0, 9.0], dtype=torch.float64)
    w, lam, k = law.log_w.exp().numpy(), law.log_lam.exp().numpy(), law.k.numpy()
    x = t.numpy()[:, None]
    pdf = (w * st.weibull_min.pdf(x, k, scale=lam)).sum(-1)
    cdf = (w * st.weibull_min.cdf(x, k, scale=lam)).sum(-1)
    assert np.allclose(law.log_density(t).numpy(), np.log(pdf), rtol=0, atol=1e-10)
    assert np.allclose(law.log_cdf(t).numpy(), np.log(cdf), rtol=0, atol=1e-10)
    a, b = t, t + 0.25
    ref = (w * (st.weibull_min.cdf(b.numpy()[:, None], k, scale=lam) - st.weibull_min.cdf(x, k, scale=lam))).sum(-1)
    assert np.allclose(law.log_interval(a, b).numpy(), np.log(ref), rtol=0, atol=1e-9)
    z = torch.zeros(5, dtype=torch.float64)
    assert torch.allclose(law.log_interval(z, b), law.log_cdf(b))             # a = 0 is the distribution function


def test_spline_flow_is_a_monotone_law():
    law = flow_law()
    t = torch.tensor([1e-9, 0.02, 0.4, 3.0, 50.0], dtype=torch.float64)
    # the density is the derivative of the distribution function, the interval is the difference
    h = 1e-6 * t
    num = (law.log_cdf(t + h).exp() - law.log_cdf(t - h).exp()) / (2 * h)
    assert torch.allclose(law.log_density(t), num.log(), atol=1e-5)
    iv = law.log_interval(t, 2 * t).exp()
    assert torch.allclose(iv, law.log_cdf(2 * t).exp() - law.log_cdf(t).exp(), atol=1e-12, rtol=1e-9)
    y = torch.linspace(-8, 8, 2001, dtype=torch.float64)
    z, _ = lik.rqs_forward(y, law.w[:1], law.h[:1], law.d[:1], law.bound)
    assert bool((z[1:] > z[:-1]).all())
    out = y.abs() >= law.bound                                                    # identity tails
    assert torch.equal(z[out], y[out])


def test_flow_with_uniform_knots_is_the_log_normal():
    K, m, s = 8, 0.3, 1.7
    raw = torch.zeros(1, 3 * K - 1)
    raw[..., 2 * K:] = math.log(math.expm1(1.0 - lik.RQS_MIN_D))                # all derivatives 1
    w, h, d = lik.rqs_params(raw, K)
    law = lik.SplineFlow(w, h, d, m, s, 6.0)
    t = torch.tensor([1e-10, 1e-6, 0.5, 4.0], dtype=torch.float64)
    ref = st.lognorm.logpdf(t.numpy(), s, scale=math.exp(m))
    assert np.allclose(law.log_density(t).numpy(), ref, atol=1e-10)


@pytest.mark.parametrize("law", [weib_law(), flow_law()])
def test_triangular_quadrature_accepts_the_new_laws(law):
    tau = torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0], dtype=torch.float64)
    v = lik.event_loglik(law, tau, "triangular", delta=1.0)
    assert v.shape == tau.shape and bool(torch.isfinite(v).all())


def stats_of(max_iet=40.0):
    return Stats(log_mean=0.2, log_std=1.3, scale=math.exp(0.2), max_train_iet=max_iet, median=1.0, q99=20.0,
                 q001=0.01, q999=35.0)


def test_head_initialization():
    S = stats_of()
    torch.manual_seed(0)
    m = Model(GRUEncoder(), "weib", 64, S)
    law, tl = m._laws(torch.zeros(2, 3, 64))
    assert tl is None and torch.allclose(law.k, torch.ones_like(law.k))       # exponential components at the start
    m = Model(GRUEncoder(), "flow", 64, S)
    assert m.head.out_features == 3 * 8 - 1
    lo, hi = math.log(FLOW_LO), math.log(40.0 * 100)
    assert m.flow_bound == pytest.approx(max(abs(lo - 0.2), abs(hi - 0.2)) / 1.3)
    law, _ = m._laws(torch.zeros(2, 3, 64))                                     # zero state: exactly the log-normal
    t = torch.tensor([[1e-10, 1e-6, 1.0]] * 2, dtype=torch.float64)
    ref = torch.tensor(st.lognorm.logpdf(t.numpy(), 1.3, scale=math.exp(0.2)))
    assert torch.allclose(law.log_density(t), ref, atol=1e-6)
    with pytest.raises(ValueError):
        Model(GRUEncoder(), "flow", 64, S, tie=True)
    cfg = dict(encoder=dict(X.GRU), head="weib", K=8, treatment="floor", sigma_min=2.0)
    with pytest.raises(ValueError):
        build(cfg, S)


def test_grid(tmp_path):
    p = tmp_path / "audit.json"
    p.write_text(json.dumps([dict(dataset=d, delta=1.0, tie_share=0.05) for d in X.HEADS_DATASETS]))
    cs = X.BUILDERS["density_heads"](tmp_path, X.audit_table(p))
    assert len(cs) == 400 and len({c["cell_id"] for c in cs}) == 400
    assert {c["head"] for c in cs} == {"weib", "flow"} and {c["arm"] for c in cs} == set(X.HEADS_ARMS)
    assert {c["dataset"] for c in cs} == set(X.HEADS_DATASETS) and {c["encoder"]["type"] for c in cs} == {"gru"}
    assert all(c["K"] == 8 and c["seed"] in X.REAL_SEEDS for c in cs)
    assert {(c["treatment"], c["eps"]) for c in cs} == {("floor", 1e-10), ("floor", 1e-6), ("jitter_gap", None),
                                                        ("interval", None)}


@pytest.fixture(scope="module")
def datadir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    old = D.data_dir()
    D.set_data_dir(tmp)
    tied_toy(tmp, name="volcano")                     # a tied toy under the name of one of the grid's data sets
    yield tmp
    D.set_data_dir(old)


def test_training_and_report(datadir, tmp_path):
    heads = tmp_path / "density_heads"
    cs = []
    for h in X.DENSITY_HEADS:
        cs += train_cells("density_heads", heads, "volcano", X.HEADS_ARMS, 1.0, lrs=("lr3e4", "lr1e3"), seeds=(0, 1),
                          head=h)
    X.write("density_heads", heads, cells=cs)
    for c in cs:
        res = json.loads((tmp_path / "density_heads" / "cells" / c["cell_id"] / "result.json").read_text())
        assert c["out_dir"] == str((heads / "cells" / c["cell_id"]).resolve())
        assert res["status"] == "ok" and all(math.isfinite(res["test"][k]) for k in ("C",) + A.HEAD_FLOORS)
    base = tmp_path / "rank_change"
    train_cells("rank_change", base, "volcano", X.HEADS_ARMS, 1.0, lrs=("lr3e4", "lr1e3"), seeds=(0, 1))
    audit = {"volcano": dict(tie_share=0.35)}
    L, s = A.report_density_heads(heads, [base], audit)
    rows = {(r["dataset"], r["head"]): r for r in s["rows"]}
    for h in X.DENSITY_HEADS + ("mix",):
        r = rows[("volcano", h)]
        assert r["complete"] and r["spread"] >= 0 and r["margin"] >= 0
        assert r["self"] == (r["top_floor6"] == "floor_1e-6") and r["changed"] == (r["top_floor6"] != r["top_C"])
        f6 = {a: m["floor_1e-6"] for a, m in r["means"].items()}
        assert max(f6.values()) == f6[r["top_floor6"]]
    assert not rows[("email_eu", "weib")]["complete"]
    assert s["heads"]["weib"]["cells"]["ok"] == 16 and s["heads"]["mix"]["n_complete"] == 1


def test_unit_metrics_definitions():
    def res(**t):
        return {"by_selection": {"C": {"test": t}}}
    def arm(c, f10, f6, f5):
        return {0: res(**{"C": c, "floor_1e-10": f10, "floor_1e-6": f6, "floor_1e-5": f5}),
                1: res(**{"C": c, "floor_1e-10": f10 + 0.2, "floor_1e-6": f6, "floor_1e-5": f5})}
    arms = {"floor_1e-10": arm(-3.0, -1.0, -2.5, -2.9), "floor_1e-6": arm(-3.2, -2.0, -1.0, -1.5),
            "jitter_gap": arm(-2.9, -2.6, -2.6, -2.6), "interval": arm(-2.8, -2.7, -2.7, -2.7)}
    m = A.head_unit_metrics(arms)
    assert m["spread"] == pytest.approx(-1.0 - (-1.9))                       # seed mean of floor 1e-10 is -1.9
    assert m["top_floor6"] == "floor_1e-6" and m["margin"] == pytest.approx(1.5) and m["self"]
    assert m["top_C"] == "interval" and m["changed"]
    assert A.top_arm({"b": 1.0, "a": 1.0}) == "a"
