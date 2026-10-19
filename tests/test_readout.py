"""Readouts: every q_B is a density of s, cell probabilities sum to one and equal the integral of q_B."""
import math

import numpy as np
import pytest
import torch
from scipy import integrate

from coarse_tpp import lik, readout as Ro

T = lambda *a: torch.tensor(a, dtype=torch.float64)


def law1():
    return lik.LogNormMix(torch.log(T(0.3, 0.7)), T(-2.0, 0.3), torch.log(T(0.4, 1.2)))


@pytest.mark.parametrize("rule", ["raw", "smooth", "point_mass"])
def test_B_densities_integrate_to_one(rule):
    law, tl, d = law1(), T(0.4), 0.3
    f = lambda s: float(Ro.B(law, tl, T(s), d, rule).exp())
    pts = [d * k for k in range(1, 6)]
    tot = integrate.quad(f, 0, 50, points=pts, limit=500)[0] + integrate.quad(f, 50, np.inf, limit=200)[0]
    assert abs(tot - 1.0) < 2e-3, (rule, tot)


@pytest.mark.parametrize("rule", ["raw", "smooth", "point_mass"])
def test_C_cells_sum_to_one_and_match_B(rule):
    law, tl, d = law1(), T(0.4), 0.3
    ks = torch.arange(0, 200000, dtype=torch.float64) * d
    lawb = lik.LogNormMix(law.log_w.expand(len(ks), 2), law.mu.expand(len(ks), 2), law.log_sigma.expand(len(ks), 2))
    c = Ro.C(lawb, tl.expand(len(ks)), ks, d, rule).exp()
    assert abs(float(c.sum()) - 1) < 1e-3, rule
    for k in (0, 2):
        ref = integrate.quad(lambda s: float(Ro.B(law, tl, T(s), d, rule).exp()), k * d, (k + 1) * d, limit=200)[0]
        assert abs(float(c[k]) - ref) < 1e-6, (rule, k)


def test_delta0_scores():
    law, tl = law1(), T(0.4)
    for rule in ("raw", "smooth"):
        assert abs(float(Ro.C(law, tl, T(0.7), 0.0, rule)) - float(law.log_density(T(0.7)))) < 1e-12
    # mixed measure: tie mass Q(TIE0) (+ pi) and density elsewhere; total mass one
    p_tie = float(Ro.C(law, tl, T(0.0), 0.0, "point_mass").exp())
    f = lambda s: float(Ro.C(law, tl, T(s), 0.0, "point_mass").exp())
    tot = p_tie + integrate.quad(f, Ro.TIE0, 50, limit=500)[0] + integrate.quad(f, 50, np.inf)[0]
    assert abs(tot - 1) < 2e-3


def test_rules_of_the_treatments():
    assert Ro.RULE["interval"] == Ro.RULE["jitter_gap"] == Ro.RULE["iw_jitter"] == Ro.RULE["interval_jitter_mix"] == "raw"
    assert Ro.RULE["jitter_time"] == Ro.RULE["jitter_time_gauss"] == Ro.RULE["triangular"] == "smooth"
    assert Ro.RULE["point_mass"] == "point_mass"
