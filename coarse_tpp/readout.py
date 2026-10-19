"""Conversion rules (readouts) and the two scores of the paper.

Every trained model gives, from the recorded history, a law q (+ a tie probability pi for the point-mass model).
It is turned into a law of s_i = t_i - t~_{i-1} (latent time since the last recorded time) by a rule fixed per
treatment:
  raw         floor, jitter_gap, iw_jitter, interval, interval_jitter_mix, latent     q_B = q
  smooth      jitter_time, jitter_time_gauss, triangular       q_B(s) = (Q(s) - Q(s - delta)) / delta
              (latent-gap models convolved with the U(0, delta) position of the previous record)
  point_mass  point_mass    q_B(s) = pi 1[s < delta] / delta + (1 - pi)(Q(s) - Q(s - delta)) / delta
Scores:
  C (recorded cell)   log of the probability of the recorded cell [tau~, tau~ + delta) under q_B
                      = interval (raw), triangular (smooth),
                        log(pi + (1 - pi) tri(0)) / log(1 - pi) + tri(k) (point_mass)
  B (latent time)     log q_B(s_i)   (synthetic data and coarsened data with known latent times)
delta = 0: C is the log score with respect to (point mass at 0) + (Lebesgue on (TIE0, inf)); every recorded gap
<= TIE0 is a tie. A continuous model puts Q(TIE0) on the tie and q on (TIE0, inf):
  continuous models   tie: log Q(TIE0)                       gap > TIE0: log q(gap)
  point_mass          tie: log(pi + (1 - pi) Q(TIE0))         gap > TIE0: log(1 - pi) + log q(gap)
B of the point-mass model at delta = 0 is log(1 - pi) + log q.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import lik

RULE = {"latent": "raw", "floor": "raw", "interval": "raw", "jitter_gap": "raw", "iw_jitter": "raw",
        "interval_jitter_mix": "raw", "jitter_time": "smooth", "jitter_time_gauss": "smooth", "triangular": "smooth",
        "point_mass": "point_mass"}
TIE0 = 1e-6


def _smooth_B(law, s, delta):
    s = s.double()
    return law.log_interval((s - delta).clamp(min=0.0), s.clamp(min=1e-300)) - math.log(delta)


def C(law, tl, tau, delta, rule):
    """Per-event recorded-cell score under the readout `rule`."""
    tau = tau.double()
    if delta == 0:
        tie = tau <= TIE0
        dens = law.log_density(torch.where(tie, torch.ones_like(tau), tau))
        q0 = law.log_cdf(torch.full_like(tau, TIE0))
        if rule == "point_mass":
            lp, l1p = F.logsigmoid(tl.double()), F.logsigmoid(-tl.double())
            return torch.where(tie, torch.logaddexp(lp, l1p + q0), l1p + dens)
        return torch.where(tie, q0, dens)
    tie = tau <= 0
    gap = lik.event_loglik(law, tau, "interval", delta=delta)
    if rule == "raw":
        return gap
    tri = lik.event_loglik(law, tau, "triangular", delta=delta)
    if rule == "smooth":
        return tri
    lp, l1p = F.logsigmoid(tl.double()), F.logsigmoid(-tl.double())
    tie_mass = torch.logaddexp(lp, l1p + tri)
    return torch.where(tie, tie_mass, l1p + tri)


def B(law, tl, s, delta, rule):
    """Per-event latent-time score log q_B(s) under the readout `rule`."""
    s = s.double()
    if delta == 0 and rule == "point_mass":          # the density part carries mass 1 - pi
        return F.logsigmoid(-tl.double()) + law.log_density(s.clamp(min=1e-10))
    if delta == 0 or rule == "raw":
        return law.log_density(s.clamp(min=1e-10))
    if rule == "smooth":
        return _smooth_B(law, s, delta)
    l1p = F.logsigmoid(-tl.double())
    pt = F.logsigmoid(tl.double()) - math.log(delta)
    cont = l1p + _smooth_B(law, s, delta)
    return torch.where(s < delta, torch.logaddexp(pt, cont), cont)
