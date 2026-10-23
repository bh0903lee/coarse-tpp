"""Component edits (Supplementary S4): pure functions, cache rules, and a tiny end-to-end run on CPU."""
import json
import math
import numpy as np
import pytest
import torch

from _toy import tied_toy, train_cells
from coarse_tpp import component_edits as K, data as D, lik

T = lambda *a: torch.tensor(a, dtype=torch.float64)
EPS = 1e-6
LE = math.log(EPS)


def narrow_mix(n=3, s0=0.01, w0=0.2):
    """Two components: a narrow one exactly at log eps (weight w0, sigma s0) and a broad one at mu = 1."""
    lw = torch.log(T(w0, 1 - w0)).expand(n, 2).clone()
    mu = T(LE, 1.0).expand(n, 2).clone()
    ls = torch.log(T(s0, 1.5)).expand(n, 2).clone()
    return lw, mu, ls


def test_comp_terms_match_lik():
    lw, mu, ls = narrow_mix()
    x = T(1e-6, 0.3, 4.0)
    ref = lik.LogNormMix(lw, mu, ls).log_density(x)
    assert torch.allclose(torch.logsumexp(K.comp_terms(lw, mu, ls, x), -1), ref, atol=1e-12)


def test_kstar_identifies_the_component_at_the_floor():
    lw, mu, ls = narrow_mix()
    k, share, z = K.kstar(lw, mu, ls, EPS)
    assert (k == 0).all() and torch.allclose(z, torch.zeros(3, dtype=torch.float64)) and (share.exp() > 0.999).all()
    mu2 = mu.clone(); mu2[:, 0] = 5.0                     # moved away: the broad component carries the density at eps
    assert (K.kstar(lw, mu2, ls, EPS)[0] == 1).all()
    lw3 = torch.log(T(0.5, 0.5)).expand(2, 2).clone()
    mu3 = T(LE, 0.0).expand(2, 2).clone(); mu3[1] = T(0.0, LE)
    assert K.kstar(lw3, mu3, torch.log(T(0.1, 0.1)).expand(2, 2).clone(), EPS)[0].tolist() == [0, 1]


def test_scale_edit_analytic():
    """Widening the narrow component at log eps from s0 to s1 changes log q(eps) by log(s0 / s1) up to the broad
    component's (tiny) contribution; exact check against the closed form of the two-component mixture."""
    s0, s1, w0 = 0.01, 0.8, 0.2
    lw, mu, ls = narrow_mix(s0=s0, w0=w0)
    tau = torch.zeros(3, dtype=torch.float64)
    k, _, _ = K.kstar(lw, mu, ls, EPS)
    mu_e, ls_e = K.edit(mu, ls, k, sigma_new=s1)
    before, after = K.floor_score(lw, mu, ls, tau, EPS), K.floor_score(lw, mu_e, ls_e, tau, EPS)
    broad = (1 - w0) * math.exp(-0.5 * ((LE - 1.0) / 1.5) ** 2) / 1.5
    exp_b = math.log(w0 / s0 + broad) - 0.5 * math.log(2 * math.pi) - LE
    exp_a = math.log(w0 / s1 + broad) - 0.5 * math.log(2 * math.pi) - LE
    assert torch.allclose(before, torch.full_like(before, exp_b), atol=1e-10)
    assert torch.allclose(after, torch.full_like(after, exp_a), atol=1e-10)
    assert abs((exp_a - exp_b) - math.log(s0 / s1)) < 1e-3
    assert torch.equal(mu_e, mu) and torch.equal(ls_e[:, 1], ls[:, 1])          # only component k*, only sigma


def test_location_edit_analytic():
    """Moving the narrow component to log(delta) = 0 leaves the broad component's density at eps; with a wide sigma
    at mu = 0 the moved component contributes w0 N(log eps; 0, s1) / eps."""
    lw, mu, ls = narrow_mix()
    tau = torch.zeros(3, dtype=torch.float64)
    k, _, _ = K.kstar(lw, mu, ls, EPS)
    mu_e, ls_e = K.edit(mu, ls, k, mu_new=0.0)
    after = K.floor_score(lw, mu_e, ls_e, tau, EPS)
    broad = math.log(0.8) - 0.5 * math.log(2 * math.pi) - math.log(1.5) - 0.5 * ((LE - 1.0) / 1.5) ** 2 - LE
    assert torch.allclose(after, torch.full_like(after, broad), atol=1e-10)
    assert torch.equal(ls_e, ls) and (mu_e[:, 0] == 0.0).all() and (mu_e[:, 1] == 1.0).all()
    mu_w, ls_w = K.edit(mu, ls, k, sigma_new=0.8, mu_new=0.0)
    both = K.floor_score(lw, mu_w, ls_w, tau, EPS)
    c0 = math.log(0.2) - 0.5 * math.log(2 * math.pi) - math.log(0.8) - 0.5 * (LE / 0.8) ** 2 - LE
    assert torch.allclose(both, torch.full_like(both, float(np.logaddexp(c0, broad))), atol=1e-10)
    mu_r, ls_r = K.edit(mu[:2], ls[:2], torch.tensor([0, 1]), sigma_new=T(0.5, 0.7), mu_new=T(2.0, 3.0))
    assert math.isclose(float(ls_r[0, 0].exp()), 0.5) and math.isclose(float(ls_r[1, 1].exp()), 0.7)
    assert float(mu_r[0, 0]) == 2.0 and float(mu_r[1, 1]) == 3.0 and float(mu_r[0, 1]) == 1.0 and float(mu_r[1, 0]) == LE


def test_weighted_median():
    v = T(1.0, 2.0, 3.0, 4.0).unsqueeze(0)
    assert float(K.weighted_median_rows(v, T(0.1, 0.1, 0.1, 0.7).unsqueeze(0))[0]) == 4.0
    assert float(K.weighted_median_rows(v, T(0.6, 0.2, 0.1, 0.1).unsqueeze(0))[0]) == 1.0
    assert float(K.weighted_median_rows(v, T(0.25, 0.25, 0.25, 0.25).unsqueeze(0))[0]) == 2.0     # lower median
    assert math.isnan(float(K.weighted_median_rows(v, torch.zeros(1, 4, dtype=torch.float64))[0]))
    lw = torch.log(T(0.9, 0.05, 0.05)).unsqueeze(0)
    ls = torch.log(T(0.01, 1.0, 2.0)).unsqueeze(0)
    med, wo = K.other_sigma_median(lw, ls, torch.tensor([0]))
    assert math.isclose(float(med[0]), 1.0) and math.isclose(float(wo[0]), 0.1)
    assert K.pooled_weighted_median(T(1.0, 5.0, 2.0, 6.0).reshape(2, 2), T(0.5, 0.5, 0.9, 0.1).reshape(2, 2)) == 2.0


def test_margin_decomposition_adds_up():
    rng = np.random.default_rng(0)
    for _ in range(20):
        sums = {a: (float(rng.normal(-100, 50)), float(rng.normal(-1000, 100))) for a in K.ARMS}
        n = 123
        for self_arm in K.SELF:
            m = K.margin(sums, self_arm, n)
            assert math.isclose(m["tie"] + m["nontie"], m["total"], rel_tol=0, abs_tol=1e-12)
            S = {a: sum(v) / n for a, v in sums.items()}
            best_other = max(S[a] for a in K.ARMS if a != self_arm)
            assert math.isclose(m["total"], S[self_arm] - best_other, abs_tol=1e-12)
            assert m["runner_up"] != self_arm and math.isclose(S[m["runner_up"]], best_other)


def record():
    """Model record: two tie rows with three narrow components, two non-tie rows with known sigmas."""
    lw = torch.log(T(0.2, 0.4, 0.4)).expand(4, 3).clone()
    mu = T(LE, LE + 0.01, LE + 0.02).expand(4, 3).clone()
    ls = torch.log(T(0.01, 0.02, 0.02)).expand(4, 3).clone()
    ls[2] = torch.log(T(0.5, 1.5, 3.0)); ls[3] = torch.log(T(1.0, 2.0, 4.0))
    lw[2] = torch.log(T(0.1, 0.6, 0.3)); lw[3] = torch.log(T(0.5, 0.25, 0.25))
    tau = T(0.0, 0.0, 0.7, 3.0)
    return dict(lw=lw, mu=mu, ls=ls, tau=tau, tie=tau <= 0)


def test_scale_is_the_pooled_nontie_weighted_median_and_edits_touch_only_ties():
    p = record()
    rows = p["tie"]
    k, _, _ = K.kstar(p["lw"][rows], p["mu"][rows], p["ls"][rows], EPS)
    sig_nt, sig_evt, fb = K.width_values(p, k)
    # pooled non-tie entries by sigma: 0.5(.1) 1.0(.5) 1.5(.6) 2.0(.25) 3.0(.3) 4.0(.25); total 2, half 1; cumulative
    # .1 .6 1.2 -> lower weighted median 1.5
    assert sig_nt == 1.5
    assert not fb.any() and torch.allclose(sig_evt, torch.full_like(sig_evt, 0.02))     # scale_event stays narrow
    V = K.edit_variants(p["mu"][rows], p["ls"][rows], k, sig_nt, sig_evt, math.log(1.0))
    assert set(V) == set(K.EDITS)
    ki = k.unsqueeze(-1)
    assert torch.allclose(V["scale"][1].gather(-1, ki).exp(), torch.full((2, 1), 1.5, dtype=torch.float64))
    assert (V["both"][0].gather(-1, ki) == 0.0).all() and (V["location"][0].gather(-1, ki) == 0.0).all()
    t = lambda v: K.floor_sums(p, EPS, v[0], v[1], rows)
    t0, u0 = K.floor_sums(p, EPS)
    assert t(V["scale"])[1] == u0 and t(V["scale"])[0] < t(V["scale_event"])[0] < t0
    p["lw"][0] = torch.log(T(1.0, 1e-300, 1e-300))               # no weight on the other components: fallback
    sig_nt2, sig_evt2, fb2 = K.width_values(p, k)
    assert fb2.tolist() == [True, False] and float(sig_evt2[0]) == sig_nt2


def _cell(tmp_path, **kw):
    rec = dict(version=K.VERSION, quarantined=False, floors={f: {} for f in K.FLOORS},
               edits={f: {e: {} for e in ("none",) + K.EDITS} for f in K.FLOORS})
    rec.update(kw)
    path = tmp_path / "c.json"
    path.write_text(json.dumps(rec))
    return path


def test_quarantined_cache_is_not_valid(tmp_path):
    ok = _cell(tmp_path)
    assert K.cache_state(ok) == "ok" and K.valid(ok) and not K.should_score(ok) and K.should_score(ok, force=True)
    q = _cell(tmp_path, quarantined=True)
    assert K.cache_state(q) == "quarantined" and not K.valid(q)
    assert not K.should_score(q) and K.should_score(q, retry_quarantined=True) and K.should_score(q, force=True)
    assert K.cache_state(_cell(tmp_path, version=K.VERSION - 1)) == "invalid"
    assert K.cache_state(_cell(tmp_path, edits={f: {"none": {}, "scale": {}} for f in K.FLOORS})) == "invalid"
    assert K.should_score(tmp_path / "missing.json")


def test_end_to_end_on_tiny_cells(tmp_path, monkeypatch):
    old = D.data_dir()
    D.set_data_dir(tmp_path)
    try:
        ds = tied_toy(tmp_path)
        run = tmp_path / "run"
        monkeypatch.setattr(K, "ARMS", ("floor_1e-6", "interval"))
        monkeypatch.setattr(K, "FLOORS", {"floor_1e-6": 1e-6})
        train_cells("rank_change", run, ds, K.ARMS, 1.0, lrs=("lr3e4", "lr1e3"))
        out = run / "component_edits"
        K.cmd_score(run, out, cpu=True)
        rec = json.loads((out / "cells" / f"{K.item_name(ds, 'gru', 0)}.json").read_text())
        assert not rec["quarantined"] and rec["n_tie"] > 0 and all(m["abs_diff"] < 1e-8 for m in rec["models"].values())
        f = "floor_1e-6"
        assert all(rec["edits"][f][e][f] != rec["edits"][f]["none"][f] for e in K.PRIMARY)   # the tie sum of the floor model moves
        L = K.cmd_report(run, out)
        S = json.loads((out / "summary.json").read_text())
        assert S["n_scored"] == S["n_items"] == 1 and not S["quarantined"]
        m = S["margins"][f"{ds}/gru"][f]["base"]
        assert math.isclose(m["tie"] + m["nontie"], m["total"], abs_tol=1e-12)
        assert any("## 2. Summary" in x for x in L)
        K.cmd_score(run, out, cpu=True)                     # a valid cache is not re-scored
    finally:
        D.set_data_dir(old)
