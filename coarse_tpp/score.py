"""Evaluation of event-sequence models on coarsely recorded data: the resolution audit and the interval score.

  detect_resolution(gaps, constructed=None)
      The audit rule of the paper (Section 2, Appendix B) applied to an array of recorded inter-event gaps: tie
      share, lattice detection, hidden-tie flag and the assigned resolution delta (or None).

  interval_loglik_from_cdf(tau, F_lo, F_hi) / interval_loglik(law, tau, delta) / floor_loglik(law, tau, eps)
      Per-event scores. For a recorded gap tau~ (0 for a tie) and resolution delta the interval score is
      log P(tau~ <= s < tau~ + delta) = log(F(tau~ + delta) - F(tau~)), where F is the model's distribution function
      of the time since the last recorded event. A model only needs to provide F at the two cell edges (any head:
      density, mixture or intensity via F = 1 - exp(-Lambda)). The difference of F is the probability of the
      half-open cell when the law has no atom at a positive time; an atom at 0 is allowed (at a tie the lower edge
      is taken as F(0-) = 0). For a law with atoms at positive times pass the left limits F(x-) at both edges, or
      give the law a `log_interval(a, b)` method. A cell of exact zero probability scores -inf. The floor score is
      the density at max(tau~, eps), the convention the interval score replaces; it is kept for comparison.

  summarize(ll, tie)
      Mean score over all events with the tie / non-tie split, the number of non-finite scores and the mean over
      the finite scores only (a diagnostic, not the score over all events).

Recorded zero gaps are often stored as a small positive floor (the NHP / EasyTPP files store 1e-6); pass
tie_stored=1.000001e-6 (the default) to read such values as ties, or 0.0 for files that store exact zeros.
Input checks: gaps must be finite and non-negative, delta and eps finite and positive (ValueError otherwise).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

TIE_STORED = 1.000001e-6
CDF_TOL = 1e-9          # tolerated roundoff of distribution function values outside [0, 1] or decreasing
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
    positive one below the threshold). Gaps must be finite and non-negative (ValueError otherwise)."""
    g = np.asarray(gaps, dtype=np.float64)
    if not np.isfinite(g).all():
        raise ValueError("recorded gaps must be finite")
    if np.any(g < 0):
        raise ValueError("recorded gaps must be non-negative")
    return np.where(g <= tie_stored, 0.0, g)


def _check_positive(name: str, x: float):
    if not (isinstance(x, (int, float)) and math.isfinite(x) and x > 0):
        raise ValueError(f"{name} must be a finite positive number, got {x!r}")


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
    """log(F_hi - F_lo) for distribution function values; -inf for a cell of zero probability. Values must be finite
    with 0 <= F_lo <= F_hi <= 1 up to CDF_TOL (differences within the tolerance are clipped to [0, 1])."""
    F_hi, F_lo = np.asarray(F_hi, np.float64), np.asarray(F_lo, np.float64)
    if not (np.isfinite(F_hi).all() and np.isfinite(F_lo).all()):
        raise ValueError("distribution function values must be finite")
    if np.any(F_lo < -CDF_TOL) or np.any(F_hi > 1 + CDF_TOL):
        raise ValueError("distribution function values must lie in [0, 1]")
    p = F_hi - F_lo
    if np.any(p < -CDF_TOL):
        raise ValueError("F(tau + delta) < F(tau): the distribution function must be non-decreasing")
    with np.errstate(divide="ignore"):
        return np.log(np.clip(p, 0.0, 1.0))


def interval_loglik_from_cdf(tau, F_lo, F_hi, tie_stored: float = TIE_STORED):
    """Per-event interval score log(F_hi - F_lo) from the model's distribution function at the cell edges
    F_lo = F(tau~) and F_hi = F(tau~ + delta); for ties (tau~ = 0) F_lo is set to 0. For a law with atoms at positive
    times F_lo and F_hi must be the left limits F(tau~-) and F((tau~ + delta)-). tau, F_lo and F_hi must have the same
    shape; invalid values raise ValueError (see _log_diff). Returns (ll, tie)."""
    g = read_gaps(tau, tie_stored)
    F_lo, F_hi = np.asarray(F_lo, np.float64), np.asarray(F_hi, np.float64)
    if not (g.shape == F_lo.shape == F_hi.shape):
        raise ValueError(f"tau, F_lo and F_hi must have the same shape, got {g.shape}, {F_lo.shape}, {F_hi.shape}")
    F_lo = np.where(g <= 0, 0.0, F_lo)
    return _log_diff(F_hi, F_lo), g <= 0


def interval_loglik(law, tau, delta: float, tie_stored: float = TIE_STORED):
    """Per-event interval score for a law object with `log_interval(a, b)` = log P(a <= S < b) (coarse_tpp.lik laws)
    or `cdf(x)` (any callable returning F); tau are stored gaps (torch or numpy). The `cdf` path is exact for laws
    without atoms at positive times (an atom at 0 counts toward the tie cell); other laws must provide
    `log_interval`. Returns (ll, tie) as numpy arrays."""
    import torch
    _check_positive("delta", delta)
    g = torch.as_tensor(read_gaps(tau, tie_stored))
    if hasattr(law, "log_interval"):
        ll = law.log_interval(g, g + delta)
    else:
        F_lo = torch.where(g <= 0, torch.zeros_like(g), torch.as_tensor(law.cdf(g), dtype=torch.float64))
        F_hi = torch.as_tensor(law.cdf(g + delta), dtype=torch.float64)
        return _log_diff(F_hi.detach().cpu().numpy(), F_lo.detach().cpu().numpy()), (g <= 0).cpu().numpy()
    return ll.detach().cpu().numpy().astype(np.float64), (g <= 0).cpu().numpy()


def floor_loglik(law, tau, eps: float, tie_stored: float = TIE_STORED):
    """Per-event floor score log f(max(tau~, eps)) for comparison with the convention; `law` needs `log_density`."""
    import torch
    _check_positive("eps", eps)
    g = torch.as_tensor(read_gaps(tau, tie_stored))
    ll = law.log_density(g.clamp(min=eps))
    return ll.detach().cpu().numpy().astype(np.float64), (g <= 0).cpu().numpy()


def summarize(ll, tie) -> dict[str, Any]:
    """Mean per-event score over all events, with the tie / non-tie split.

    mean, mean_tie, mean_nontie are over all events of the group: -inf when an event has a cell of zero probability,
    nan when a score is nan (None for an empty group). all_finite tells whether every score is finite; n_neg_inf and
    n_nonfinite count the -inf and all non-finite scores. mean_finite is the mean over the finite scores only (n_finite
    of them), a diagnostic that must not be read as the score over all events."""
    ll = np.asarray(ll, np.float64); tie = np.asarray(tie, bool)
    if ll.shape != tie.shape:
        raise ValueError(f"ll and tie must have the same shape, got {ll.shape}, {tie.shape}")
    fin = np.isfinite(ll)

    def mean(x):
        with np.errstate(invalid="ignore"):
            return float(np.mean(x)) if x.size else None

    out = dict(n=int(ll.size), n_tie=int(tie.sum()), all_finite=bool(fin.all()), n_finite=int(fin.sum()),
               n_neg_inf=int(np.isneginf(ll).sum()), n_nonfinite=int((~fin).sum()),
               mean=mean(ll), mean_tie=mean(ll[tie]), mean_nontie=mean(ll[~tie]),
               mean_finite=float(ll[fin].mean()) if fin.any() else math.nan)
    return out
