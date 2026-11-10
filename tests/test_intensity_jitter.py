"""Jitter and point-mass scores of intensity-based models: quadrature, closed forms and the pure report helpers."""
import math

import numpy as np
import pytest
import torch

from coarse_tpp import intensity_eval as IE
from coarse_tpp import intensity_jitter as J


class FakeIntensity:
    """Intensity object with a fixed total intensity lam(s) (no model)."""

    def __init__(self, fn, mid="NHP", gaps=((0.0, 0.0, 2.5, 0.7), (0.0, 1.2, 0.0, 3.0))):
        g = torch.tensor(gaps, dtype=torch.float32)
        self.model, self.mid, self.fn = None, mid, fn
        non_pad = torch.ones_like(g, dtype=torch.bool)
        self.batch = (torch.cumsum(g, 1), g, torch.zeros_like(g, dtype=torch.long), non_pad, None)

    def total(self, dtimes):
        return self.fn(dtimes.double())


def run(fn, delta=1.0, mid="NHP", rule=IE.Rule()):
    ev = FakeIntensity(fn, mid)
    return J.scores(ev, delta, rule, "email_eu", 1.000001e-6, "cpu")


@pytest.mark.parametrize("mid", ["NHP", "S2P2"])
def test_constant_intensity_gives_the_closed_form(mid):
    lam, delta = 0.8, 0.5
    rule = IE.Rule(strata=64, cell_panels=50)
    o = run(lambda x: torch.full_like(x, lam), delta, mid, rule)
    tau = IE._targets(FakeIntensity(None).batch, 1.000001e-6)
    assert torch.allclose(o["jitter"], math.log(lam) - lam * (tau + delta / 2), atol=1e-12)
    assert torch.allclose(o["interval"], -lam * tau + math.log(1 - math.exp(-lam * delta)), atol=1e-12)
    assert torch.allclose(o["floor_1e-6"], math.log(lam) - lam * tau.clamp(min=1e-6), atol=1e-12)


def test_jitter_is_at_most_interval_minus_log_delta():
    """Jensen: E_u log q(tau + delta u) <= log((1/delta) int_cell q) = interval - log delta."""
    for delta in (0.1, 1.0, 3.0):
        o = run(lambda x: 0.3 + 2.0 * torch.exp(-3.0 * x) + 0.5 * torch.sin(x) ** 2, delta)
        assert torch.all(o["jitter"] <= o["interval"] - math.log(delta) + 1e-10)
        # and the gap closes as the cell shrinks (the density is smooth)
    small = run(lambda x: 0.3 + 2.0 * torch.exp(-3.0 * x), 1e-3)
    assert torch.allclose(small["jitter"], small["interval"] - math.log(1e-3), atol=1e-5)


def test_merged_rule_integrates_exactly_up_to_every_jitter_node():
    uj, wj = J.jitter_nodes()
    assert abs(wj.sum() - 1) < 1e-14 and np.all((uj > 0) & (uj < 1))
    for breaks, n in ((IE.GRADED_BREAKS, 16), (tuple(np.arange(11) / 10), 2)):
        u, W, A = J.merged_rule(breaks, n)
        assert abs(W.sum() - 1) < 1e-13
        f = 1 + 3 * u ** 2                      # degree 2 < 2n: exact on every panel
        assert np.allclose(A @ f, uj + uj ** 3, atol=1e-13)


def test_jitter_rule_follows_the_scoring_rule():
    r = IE.Rule()
    assert len(J.jitter_rule(r, "email_eu", "NHP")[0]) == 16 * (len(IE.GRADED_BREAKS) - 1 + J.N_NODES)
    assert len(J.jitter_rule(r, "email_eu", "AttNHP")[0]) == 64 * (len(IE.GRADED_BREAKS) - 1 + J.N_NODES)
    assert len(J.jitter_rule(IE.Rule(cell_panels=100), "email_eu", "S2P2")[0]) == 2 * (100 + J.N_NODES)


def test_point_mass_from_stored_scores():
    s = {"n": 10, "n_tie": 4, "interval_tie": -1.5, "floor_1e-10_nontie": -3.0}
    assert J.point_mass(s) == pytest.approx((4 * -1.5 + 6 * -3.0) / 10)
    # on ties alone it is the interval tie term, without ties the density at the recorded gap
    assert J.point_mass(dict(s, n_tie=10)) == -1.5
    assert J.point_mass(dict(s, n_tie=0, interval_tie=None)) == -3.0
    assert J.point_mass(dict(s, interval_tie=None)) is None


def test_gate_flags_differences_and_counts():
    ref = {"n": 5, "n_tie": 1, "interval": -2.0, "floor_1e-6": -1.0}
    assert J.gate(dict(ref), ref)[0]
    assert J.gate(dict(ref, **{"floor_1e-6": -1.0 + 5e-5}), ref)[0]
    assert not J.gate(dict(ref, interval=-2.0 + 5e-5), ref)[0]
    assert not J.gate(dict(ref, n_tie=2), ref)[0]


def cell(ds, model, seed, dev, test, exp="time_only"):
    return dict(experiment=exp, dataset=ds, model=model, seed=seed, status="ok", quarantined=False,
                dev={"floor_1e-6": dev, "jitter": dev}, test={"floor_1e-6": test, "jitter": test, "n": 10, "n_tie": 1})


def test_selection_and_paired_difference():
    cs = [cell("email_eu", "S2P2", s, 1.0, 2.0 + s) for s in range(5)]
    cs += [cell("email_eu", "NHP", s, 0.5, 2.5) for s in range(5)]
    cs += [cell("email_eu", "THP", s, 9.0, 0.0) for s in range(4)]        # four seeds only
    assert J.select(cs, "email_eu", "floor_1e-6", True)[0] == "S2P2"
    assert J.select(cs, "email_eu", "floor_1e-6", False)[0] == "THP"
    d = J.s2p2_minus_nhp(cs, "email_eu", "jitter")
    assert d["mean"] == pytest.approx(1.5) and d["n_pos"] == 4 and d["n"] == 5
