"""Evaluation of event-sequence models on coarsely recorded data: the resolution audit and the interval score.

  detect_resolution(gaps, constructed=None)
      The audit rule of the paper (Section 2, Appendix B) applied to an array of recorded inter-event gaps: tie
      share, lattice detection, hidden-tie flag and the assigned resolution delta (or None).

  interval_loglik_from_cdf(tau, F_lo, F_hi) / interval_loglik(law, tau, delta) / floor_loglik(law, tau, eps)
      Per-event scores. For a recorded gap tau~ (0 for a tie) and resolution delta the interval score is
      log P(tau~ <= s < tau~ + delta) = log(F(tau~ + delta) - F(tau~)), where F is the model's distribution function
      of the time since the last recorded event. A model only needs to provide F at the two cell edges (any head:
      density, mixture or intensity via F = 1 - exp(-Lambda)). The floor score is the density at max(tau~, eps),
      the convention the interval score replaces; it is kept for comparison.

  summarize(ll, tie)
      Mean score with the tie / non-tie split.

Recorded zero gaps are often stored as a small positive floor (the NHP / EasyTPP files store 1e-6); pass
tie_stored=1.000001e-6 (the default) to read such values as ties, or 0.0 for files that store exact zeros.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

TIE_STORED = 1.000001e-6
_CLOCK = (1.0, 60.0, 3600.0, 86400.0)
CANDIDATES = sorted({round(k * 10.0 ** e, 12) for e in range(-6, 6) for k in (1, 2, 5)}
                    | {2.0 ** -e for e in range(1, 21)}
                    | {u / v for u in (1e-3,) + _CLOCK for v in _CLOCK} | {1 / 24})


# ------------------------------------------------------------------ resolution audit
def lattice_share(g: np.ndarray, d: float) -> float:
    """Share of positive gaps that are integer multiples k >= 1 of d within |g/d - k| <= 1e-4 + 1e-9 g/d."""
    q = g / d
    k = np.rint(q)
    return float(np.mean((k >= 1) & (np.abs(q - k) <= 1e-4 + 1e-9 * q)))


def read_gaps(gaps, tie_stored: float = TIE_STORED) -> np.ndarray:
    """Recorded gaps as float64 with stored ties mapped to exactly 0 (any gap <= tie_stored, including a genuinely
    positive one below the threshold)."""
    g = np.asarray(gaps, dtype=np.float64)
    return np.where(g <= tie_stored, 0.0, g)


def detect_resolution(gaps, constructed: float | None = None, tie_stored: float = TIE_STORED) -> dict[str, Any]:
    """The audit rule on an array of recorded gaps (all splits pooled, first events excluded by the caller).

    delta = (i) `constructed` when the data set documents a resolution; (ii) else the largest candidate grid that
    at least 99% of the positive gaps lie on (lattice), if it is >= 1e-6 of the median positive gap; (iii) else the
    smallest positive gap when at least 1% of the gaps are ties; (iv) else None (continuous at storage precision)."""
    g = read_gaps(gaps, tie_stored)
    if g.size == 0:
        raise ValueError("no gaps")
    pos = g[g > 0]
    out: dict[str, Any] = dict(n=int(g.size), tie_share=float(np.mean(g <= 0)))
    if pos.size == 0:
        out.update(delta=constructed, delta_rule="(i) constructed" if constructed else "(iv) undefined", delta_hat=None)
        return out
    m0 = float(pos.min())
    out.update(median_gap=float(np.median(g)), median_pos_gap=float(np.median(pos)), min_pos_gap=m0, max_gap=float(g.max()))
    cands = sorted(set(CANDIDATES) | {m0 / k for k in range(1, 11)})
    shares = {d: lattice_share(pos, d) for d in cands if d <= pos.max()}
    ok = [d for d, s in shares.items() if s >= 0.99]
    dh = max(ok) if ok else None
    out["lattice_shares"] = {repr(d): s for d, s in shares.items() if s >= 0.5}
    out["delta_hat"] = dh
    out["delta_hat_rel_median"] = (dh / out["median_pos_gap"]) if dh else None
    s1 = float(np.mean((pos >= m0) & (pos < 2 * m0))); s2 = float(np.mean((pos >= 2 * m0) & (pos < 10 * m0)))
    out.update(hidden_s1=s1, hidden_s2=s2, hidden_flag=bool(s1 >= 0.05 and s2 < s1 / 10))
    if constructed:
        out["delta"], out["delta_rule"] = float(constructed), "(i) constructed"
    elif dh is not None and out["delta_hat_rel_median"] >= 1e-6:
        out["delta"], out["delta_rule"] = dh, "(ii) lattice"
    elif out["tie_share"] >= 0.01:
        out["delta"], out["delta_rule"] = m0, "(iii) ties, min positive gap"
    else:
        out["delta"], out["delta_rule"] = None, "(iv) undefined"
    return out


# ------------------------------------------------------------------ scores
def _log_diff(F_hi, F_lo):
    p = np.asarray(F_hi, np.float64) - np.asarray(F_lo, np.float64)
    if np.any(p < -1e-9):
        raise ValueError("F(tau + delta) < F(tau): the distribution function must be non-decreasing")
    return np.log(np.clip(p, 1e-300, None))


def interval_loglik_from_cdf(tau, F_lo, F_hi, tie_stored: float = TIE_STORED):
    """Per-event interval score log(F_hi - F_lo) from the model's distribution function at the cell edges
    F_lo = F(tau~) and F_hi = F(tau~ + delta) (F_lo must be 0 for ties, where tau~ = 0). Returns (ll, tie)."""
    g = read_gaps(tau, tie_stored)
    F_lo = np.where(g <= 0, 0.0, np.asarray(F_lo, np.float64))
    return _log_diff(F_hi, F_lo), g <= 0


def interval_loglik(law, tau, delta: float, tie_stored: float = TIE_STORED):
    """Per-event interval score for a law object with `log_interval(a, b)` (coarse_tpp.lik laws) or `cdf(x)`
    (any callable returning F); tau are stored gaps (torch or numpy). Returns (ll, tie) as numpy arrays."""
    import torch
    t = torch.as_tensor(np.asarray(tau, np.float64))
    g = torch.where(t <= tie_stored, torch.zeros_like(t), t)
    if hasattr(law, "log_interval"):
        ll = law.log_interval(g, g + delta)
    else:
        ll = torch.log((law.cdf(g + delta) - law.cdf(g)).clamp(min=1e-300))
    return ll.detach().cpu().numpy().astype(np.float64), (g <= 0).cpu().numpy()


def floor_loglik(law, tau, eps: float, tie_stored: float = TIE_STORED):
    """Per-event floor score log f(max(tau~, eps)) for comparison with the convention; `law` needs `log_density`."""
    import torch
    t = torch.as_tensor(np.asarray(tau, np.float64))
    g = torch.where(t <= tie_stored, torch.zeros_like(t), t)
    ll = law.log_density(g.clamp(min=eps))
    return ll.detach().cpu().numpy().astype(np.float64), (g <= 0).cpu().numpy()


def summarize(ll, tie) -> dict[str, Any]:
    """Mean per-event score with the tie / non-tie split; non-finite values are counted, not averaged."""
    ll = np.asarray(ll, np.float64); tie = np.asarray(tie, bool)
    fin = np.isfinite(ll)
    out = dict(n=int(ll.size), n_tie=int(tie.sum()), n_nonfinite=int((~fin).sum()),
               mean=float(ll[fin].mean()) if fin.any() else math.nan)
    out["mean_tie"] = float(ll[fin & tie].mean()) if (fin & tie).any() else None
    out["mean_nontie"] = float(ll[fin & ~tie].mean()) if (fin & ~tie).any() else None
    return out
