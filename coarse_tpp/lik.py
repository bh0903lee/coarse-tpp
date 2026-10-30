"""Likelihoods of one inter-event gap under a predictive law, for exact and coarsely recorded gaps.

Every predictive law (log-normal mixture, RMTPP intensity, THP intensity, Weibull mixture, spline flow) exposes
  log_density(tau)      log p(tau), tau > 0
  log_interval(a, b)    log P(a <= tau < b), 0 <= a < b (a = 0 allowed)
  log_cdf(b)            log P(tau < b)
All computations are float64. Recording model: absolute times are floored to a grid of width delta,
t~ = delta floor(t / delta); the recorded gap is tau~ = k delta.

Scores of a recorded gap (`event_loglik`):
  floor        log p(max(tau~, eps))                    (the eps-floor convention)
  interval     log P(tau~ <= tau < tau~ + delta)        (interval likelihood of the recorded cell)
  triangular   log E_{v ~ U(0, delta)} P(tau~ + v - delta < tau <= tau~ + v): the probability of the recorded
               cell index when both end points carry independent uniform phases
  point_mass   tie: log pi; positive gap: log(1 - pi) + log p(tau~)   (zero-inflated model, MoTPP eq. 4)
Jitter treatments change the data, not the likelihood (coarse_tpp.data), and are scored as floor 1e-10.
"""
from __future__ import annotations

import math

import numpy as np
import torch

LOG_2PI = math.log(2 * math.pi)
LN2 = math.log(2.0)
_NARROW = 1e-5          # below this width in z units an interval probability uses the midpoint rule
_GL = {}


def gauss_legendre(n: int, dtype=torch.float64, device="cpu"):
    """Nodes in (0, 1) and weights summing to 1."""
    key = (n, dtype, str(device))
    if key not in _GL:
        x, w = np.polynomial.legendre.leggauss(n)
        _GL[key] = (torch.tensor((x + 1) / 2, dtype=dtype, device=device),
                    torch.tensor(w / 2, dtype=dtype, device=device))
    return _GL[key]


def log1mexp(x):
    """log(1 - exp(x)) for x <= 0, finite for every finite input (x is clamped just below 0)."""
    x = x.clamp(max=-1e-300)
    near = x > -LN2
    x1 = torch.where(near, x, torch.full_like(x, -1.0))          # each branch gets a safe input, so the
    x2 = torch.where(near, torch.full_like(x, -1.0), x)          # unused branch has a finite gradient
    return torch.where(near, torch.log(-torch.expm1(x1)), torch.log1p(-torch.exp(x2)))


def log_ndtr_diff(za, zb):
    """log(Phi(zb) - Phi(za)) for za < zb, za may be -inf. Stable in both tails; finite gradients."""
    lndtr = torch.special.log_ndtr
    upper = za > 0                                   # both in the upper tail: use survival functions
    hi = torch.where(upper, -za, zb)
    lo = torch.where(upper, -zb, za)
    lhi, llo = lndtr(hi), lndtr(lo)
    exact = lhi + log1mexp(llo - lhi)
    w = (zb - za)
    zm = torch.where(torch.isfinite(za), 0.5 * (za + zb), zb)
    narrow = -0.5 * LOG_2PI - 0.5 * zm * zm + torch.log(w.clamp(min=1e-300))
    return torch.where(torch.isfinite(za) & (w < _NARROW), narrow, exact)


class LogNormMix:
    """sum_k w_k LogNormal(mu_k, sigma_k) on raw gaps; parameters (..., K)."""

    def __init__(self, log_w, mu, log_sigma):
        self.log_w, self.mu, self.log_sigma = log_w.double(), mu.double(), log_sigma.double()
        self.sigma = self.log_sigma.exp()

    def _z(self, t):
        return (torch.log(t).unsqueeze(-1) - self.mu) / self.sigma

    def log_density(self, tau):
        tau = tau.double()
        lt = torch.log(tau).unsqueeze(-1)
        z = (lt - self.mu) / self.sigma
        comp = -0.5 * LOG_2PI - self.log_sigma - 0.5 * z * z - lt
        return torch.logsumexp(self.log_w + comp, dim=-1)

    def log_cdf(self, b):
        return torch.logsumexp(self.log_w + torch.special.log_ndtr(self._z(b.double())), dim=-1)

    def log_interval(self, a, b):
        a, b = a.double(), b.double()
        pos = a > 0
        a_safe = torch.where(pos, a, b * 0.5)           # any positive value; the a = 0 branch is log_cdf
        za, zb = self._z(a_safe), self._z(b)
        inner = torch.logsumexp(self.log_w + log_ndtr_diff(za, zb), dim=-1)
        return torch.where(pos, inner, self.log_cdf(b))


LAM_CAP = 1e200          # compensators above this are capped (probability exp(-1e200) = 0 anyway)


class _Intensity:
    """Next-event law from a conditional intensity lambda(s), s = time since the last event:
    p(s) = lambda(s) exp(-Lambda(s)), P(a <= tau < b) = exp(-Lambda(a)) - exp(-Lambda(b)).
    Compensators are capped at LAM_CAP and log intensities at +-700 so that heavy-tailed gaps cannot overflow
    float64; the cap only binds where the probability is 0 to float precision."""

    def log_density(self, tau):
        tau = tau.double()
        return self.log_lambda(tau) - self.Lambda(tau)

    def log_cdf(self, b):
        return log1mexp(-self.Lambda(b.double()))

    def log_interval(self, a, b):
        a, b = a.double(), b.double()
        La = self.Lambda(a)
        dL = self.Lambda_between(a, b)
        return -La + log1mexp(-dL)


class RMTPP(_Intensity):
    """lambda(s) = exp(c + w s / scale) (Du et al. 2016); closed-form compensator.
    c (...,) and w (...,) are free (the model bounds w); |x| < 1e-6 uses the series of the expm1 ratio."""

    def __init__(self, c, w, scale: float):
        self.c, self.w, self.scale = c.double(), w.double(), float(scale)

    def log_lambda(self, s):
        return (self.c + self.w * (s / self.scale)).clamp(-700.0, 700.0)

    @staticmethod
    def _log_ratio(x):
        """log((e^x - 1) / x), stable for every x and with finite gradients in every branch."""
        small = x.abs() < 1e-6
        xs = torch.where(small, torch.ones_like(x), x)
        pos = torch.where(xs > 0, xs, torch.ones_like(xs))
        neg = torch.where(xs < 0, xs, -torch.ones_like(xs))
        big = pos + log1mexp(-pos) - torch.log(pos)                   # x > 0
        low = torch.log(-torch.expm1(neg)) - torch.log(-neg)          # x < 0
        return torch.where(small, 0.5 * x, torch.where(xs > 0, big, low))

    def log_Lambda(self, s):
        """log Lambda(s) = c + log s + log((e^x - 1) / x), x = w s / scale (-inf at s = 0)."""
        x = self.w * s / self.scale
        return self.c + torch.log(s) + self._log_ratio(x)

    def Lambda(self, s):
        return torch.exp(self.log_Lambda(s).clamp(max=math.log(LAM_CAP)))

    def log_Lambda_between(self, a, b):
        """log(Lambda(b) - Lambda(a)) = c + w a / scale + log(b - a) + log((e^x - 1) / x), x = w (b - a) / scale,
        without subtracting two compensators (that would lose every digit in a decaying tail)."""
        d = (b - a).clamp(min=1e-300)
        return self.c + self.w * a / self.scale + torch.log(d) + self._log_ratio(self.w * d / self.scale)

    def Lambda_between(self, a, b):
        return torch.exp(self.log_Lambda_between(a, b).clamp(max=math.log(LAM_CAP)))

    def log_interval(self, a, b):
        a, b = a.double(), b.double()
        return -self.Lambda(a) + _log1mexp_neg_exp(self.log_Lambda_between(a, b))


def _log1mexp_neg_exp(l):
    """log(1 - exp(-exp(l))) for any finite l: log(1 - e^{-y}) with y = e^l; = l - y/2 + O(y^2) for small y
    (kept in log space, so an interval mass far below 1e-300 is still exact)."""
    small = l < -20.0
    ls = torch.where(small, torch.zeros_like(l), l)
    big = log1mexp(-torch.exp(ls.clamp(max=math.log(LAM_CAP))))
    lt = torch.where(small, l, torch.full_like(l, -30.0))
    return torch.where(small, lt - 0.5 * torch.exp(lt), big)


class THPIntensity(_Intensity):
    """lambda(s) = softplus(c + beta s / scale) (Zuo et al. 2020, time-only form with the elapsed time scaled by a
    fixed train time scale instead of the current absolute time). For beta <= 0 the law may be defective
    (probability of no further event), exactly as in THP.

    Compensator: with x = c + k s, k = beta / scale, the s-range is split where |x| = XL = 40.
      x < -XL   softplus(x) = e^x (1 + O(e^-40)):        exact,  e^{x(a)} (b - a) exprel(k (b - a))
      x > XL    softplus(x) = x  (+ e^-x <= 4e-18):      exact,  (b - a) (c + k (a + b) / 2)
      |x| <= XL Gauss-Legendre, `n_quad` (16) nodes on each of P equal panels, P chosen per call so that a
                panel spans at most 4 units of x (relative error below 1e-13; softplus has its nearest complex
                singularity at distance pi).
    The pieces are formed in s, so k -> 0 is exact as well. log lambda is log-softplus without a floor."""

    XL = 40.0

    def __init__(self, c, beta, scale: float, n_quad: int = 16):
        self.c, self.beta, self.scale, self.n = c.double(), beta.double(), float(scale), n_quad

    @property
    def k(self):
        return self.beta / self.scale

    def lam(self, s):
        return torch.nn.functional.softplus(self.c + self.k * s)

    def log_lambda(self, s):
        x = self.c + self.k * s
        low = x < -self.XL
        xl = torch.where(low, x, torch.full_like(x, -self.XL))
        xh = torch.where(low, torch.zeros_like(x), x)
        return torch.where(low, xl - 0.5 * torch.exp(xl),          # log log1p(e^x) = x - e^x / 2 + ...
                           torch.log(torch.nn.functional.softplus(xh)))

    def _bounds(self):
        """s where x = -XL and x = XL, ordered by s (k = 0 is evaluated separately; a tiny stand-in slope keeps
        every intermediate finite)."""
        k = self.k
        ks = torch.where(k.abs() < 1e-300, torch.full_like(k, 1e-300), k)
        s1, s2 = (-self.XL - self.c) / ks, (self.XL - self.c) / ks
        return torch.minimum(s1, s2), torch.maximum(s1, s2)

    def Lambda_between(self, a, b):
        a, b = a.double(), b.double()
        k = self.k
        lo, hi = self._bounds()
        clip = lambda u, l, h: torch.minimum(torch.maximum(u, l), h)
        # middle regime: [m1, m2] = [a, b] intersected with [lo, hi]
        m1, m2 = clip(a, lo, hi), clip(b, lo, hi)
        width_x = (k.abs() * (m2 - m1)).detach()
        P = int(max(1, min(64, math.ceil(float(width_x.max()) / 4.0)))) if width_x.numel() else 1
        x, w = gauss_legendre(self.n, device=self.c.device)
        node = ((torch.arange(P, dtype=torch.float64, device=self.c.device).unsqueeze(-1) + x) / P).reshape(-1)
        wt = (w.repeat(P) / P)
        sn = m1.unsqueeze(-1) + (m2 - m1).unsqueeze(-1) * node
        mid = (m2 - m1) * (torch.nn.functional.softplus(self.c.unsqueeze(-1) + k.unsqueeze(-1) * sn) * wt).sum(-1)
        # the two outer regimes: below-left / above-right of [lo, hi] in s; which is "low x" depends on sign k
        L1, L2 = torch.minimum(a, lo), torch.minimum(b, lo)                 # s < lo
        R1, R2 = torch.maximum(a, hi), torch.maximum(b, hi)                 # s > hi
        def expo(u1, u2):        # integral of e^{c + k s} over [u1, u2], cancellation-free
            d = (u2 - u1)
            dd = torch.where(d > 0, d, torch.ones_like(d))
            lv = self.c + k * u1 + torch.log(dd) + RMTPP._log_ratio(k * dd)
            return torch.where(d > 0, torch.exp(lv.clamp(max=math.log(LAM_CAP))), torch.zeros_like(d))
        def lin(u1, u2):         # integral of (c + k s) over [u1, u2]
            d = (u2 - u1)
            return torch.where(d > 0, d * (self.c + 0.5 * k * (u1 + u2)), torch.zeros_like(d))
        pos = k > 0                                   # x increasing: s < lo is the e^x regime, s > hi linear
        left = torch.where(pos, expo(L1, L2), lin(L1, L2))
        right = torch.where(pos, lin(R1, R2), expo(R1, R2))
        kz = k.abs() < 1e-300
        # k = 0 exactly: (b - a) softplus(c), written with the midpoint c + k (a + b) / 2 so that the value is the
        # same and d/dk at k = 0 equals the exact derivative sigmoid(c) (b^2 - a^2) / 2 of the integral. Without it
        # a slope initialised at 0 would receive no gradient from an objective made of compensators alone.
        flat = (b - a) * torch.nn.functional.softplus(self.c + k * 0.5 * (a + b))
        out = torch.where(kz, flat, left + mid + right)
        return out.clamp(min=1e-300, max=LAM_CAP)

    def Lambda(self, s):
        return self.Lambda_between(torch.zeros_like(s), s)


# ------------------------------------------------------------------ further density heads
# Two further flexible laws of the gap (density-heads experiment, Supplementary Table S8), both with closed-form
# distribution functions so that the interval score is exact.
LOG_K_LO, LOG_K_HI = math.log(0.05), math.log(20.0)     # Weibull shape bounds


class WeibullMix:
    """sum_k w_k Weibull(scale lambda_k, shape k_k) on raw gaps; parameters (..., K): log_w, log_lam, log_k.
    x_k(t) = (t / lambda_k)^k_k; density k/lambda (t/lambda)^(k-1) exp(-x); P(tau < b) = 1 - exp(-x(b)).
    A shape below 1 makes the density diverge at zero, as a narrow log-normal component does near the floor."""

    def __init__(self, log_w, log_lam, log_k):
        self.log_w, self.log_lam, self.log_k = log_w.double(), log_lam.double(), log_k.double()
        self.k = self.log_k.exp()

    def _log_x(self, t):
        return (self.k * (torch.log(t).unsqueeze(-1) - self.log_lam)).clamp(max=700.0)

    def log_density(self, tau):
        tau = tau.double()
        lx = self._log_x(tau)
        comp = self.log_k - torch.log(tau).unsqueeze(-1) + lx - lx.exp()
        return torch.logsumexp(self.log_w + comp, dim=-1)

    def log_cdf(self, b):
        return torch.logsumexp(self.log_w + _log1mexp_neg_exp(self._log_x(b.double())), dim=-1)

    def log_interval(self, a, b):
        a, b = a.double(), b.double()
        pos = a > 0
        a_safe = torch.where(pos, a, b * 0.5)
        xa = self._log_x(a_safe).exp()
        # x(b) - x(a) = x(b) (1 - exp(-k log(b / a))), with the log ratio formed from the interval width rather than
        # from two rounded logarithms
        ratio = self.k * torch.log1p((b - a_safe) / a_safe).unsqueeze(-1)
        log_dx = self._log_x(b) + log1mexp(-ratio)
        inner = torch.logsumexp(self.log_w - xa + _log1mexp_neg_exp(log_dx), dim=-1)
        return torch.where(pos, inner, self.log_cdf(b))


RQS_MIN_W, RQS_MIN_H, RQS_MIN_D = 1e-3, 1e-3, 1e-3


def rqs_params(raw, K: int):
    """raw (..., 3K - 1) -> knot widths and heights (..., K) as fractions of the domain and internal derivatives
    (..., K - 1). Normalized in float64, the precision of the spline (float32 fractions can sum to a value other than
    1 after promotion, which would make the spline jump at the identity tails)."""
    raw = raw.double()
    w = torch.softmax(raw[..., :K], -1)
    h = torch.softmax(raw[..., K:2 * K], -1)
    w = RQS_MIN_W + (1 - RQS_MIN_W * K) * w
    h = RQS_MIN_H + (1 - RQS_MIN_H * K) * h
    d = RQS_MIN_D + torch.nn.functional.softplus(raw[..., 2 * K:])
    return w, h, d


def rqs_forward(y, w, h, d, bound: float):
    """Monotone rational-quadratic spline (Durkan et al. 2019) on [-bound, bound], identity outside, boundary
    derivatives 1. y (...,); w, h (..., K) fractions; d (..., K - 1). Returns (T(y), log T'(y)), float64."""
    y = y.double(); w, h, d = w.double(), h.double(), d.double()
    shape = torch.broadcast_shapes(y.shape, w.shape[:-1])
    y = y.expand(shape)
    w, h, d = (t.expand(*shape, t.shape[-1]) for t in (w, h, d))
    one = torch.ones_like(d[..., :1])
    dd = torch.cat([one, d, one], -1)                                     # (..., K + 1)
    xk = torch.cat([torch.zeros_like(w[..., :1]), w.cumsum(-1)], -1) * 2 * bound - bound
    yk = torch.cat([torch.zeros_like(h[..., :1]), h.cumsum(-1)], -1) * 2 * bound - bound
    inside = (y > -bound) & (y < bound)
    yi = torch.where(inside, y, torch.zeros_like(y))
    K = w.shape[-1]
    idx = (torch.searchsorted(xk[..., 1:-1].contiguous(), yi.unsqueeze(-1).contiguous())).clamp(0, K - 1)
    g = lambda t: t.gather(-1, idx).squeeze(-1)
    g1 = lambda t: t.gather(-1, idx + 1).squeeze(-1)
    x0, x1, y0, y1, d0, d1 = g(xk), g1(xk), g(yk), g1(yk), g(dd), g1(dd)
    wb, hb = x1 - x0, y1 - y0
    s = hb / wb
    xi = ((yi - x0) / wb).clamp(0.0, 1.0)
    t = xi * (1 - xi)
    den = s + (d1 + d0 - 2 * s) * t
    out = y0 + hb * (s * xi * xi + d0 * t) / den
    logd = 2 * torch.log(s) + torch.log(d1 * xi * xi + 2 * s * t + d0 * (1 - xi) ** 2) - 2 * torch.log(den)
    return torch.where(inside, out, y), torch.where(inside, logd, torch.zeros_like(y))


class SplineFlow:
    """Flow on the standardized log gap: y = (log tau - m) / s, z = T(y) ~ N(0, 1), T a monotone rational-quadratic
    spline on [-bound, bound] (identity outside). log p(tau) = log phi(T(y)) + log T'(y) - log s - log tau;
    P(tau < b) = Phi(T(y_b)). `bound` is chosen by the head so that the floors lie inside the spline domain."""

    def __init__(self, w, h, d, m: float, s: float, bound: float):
        self.w, self.h, self.d = w, h, d
        self.m, self.s, self.bound = float(m), float(s), float(bound)

    def _T(self, t):
        y = (torch.log(t.double()) - self.m) / self.s
        return rqs_forward(y, self.w, self.h, self.d, self.bound)

    def log_density(self, tau):
        tau = tau.double()
        z, ld = self._T(tau)
        return -0.5 * LOG_2PI - 0.5 * z * z + ld - math.log(self.s) - torch.log(tau)

    def log_cdf(self, b):
        z, _ = self._T(b.double())
        return torch.special.log_ndtr(z)

    def log_interval(self, a, b):
        a, b = a.double(), b.double()
        pos = a > 0
        a_safe = torch.where(pos, a, b * 0.5)
        za, _ = self._T(a_safe)
        zb, _ = self._T(b)
        return torch.where(pos, log_ndtr_diff(za, zb), self.log_cdf(b))


# ------------------------------------------------------------------ scores
SCORES = ("floor", "interval", "triangular", "point_mass")


def _mix_log_piece(law, a, b, left: bool, delta: float, base=None):
    """Per component, log of (1/delta) * integral_a^b (tau - base) p_k(tau) dtau (left; base <= a, default a)
    or (1/delta) * integral_a^b (b - tau) p_k(tau) dtau (right), a >= 0, via the log-normal partial
    expectation: integral_a^b tau p_k = exp(mu + s^2/2) [Phi(zb - s) - Phi(za - s)]."""
    pos = a > 0
    a_safe = torch.where(pos, a, b * 0.5)
    za = (torch.log(a_safe).unsqueeze(-1) - law.mu) / law.sigma
    zb = (torch.log(b).unsqueeze(-1) - law.mu) / law.sigma
    lnd = torch.special.log_ndtr
    lp = torch.where(pos.unsqueeze(-1), log_ndtr_diff(za, zb), lnd(zb))                       # log P_k
    lq = torch.where(pos.unsqueeze(-1), log_ndtr_diff(za - law.sigma, zb - law.sigma), lnd(zb - law.sigma))
    m = torch.exp((law.mu + 0.5 * law.sigma ** 2 + lq - lp).clamp(max=700.0))                 # E[tau | a <= tau < b]
    aa, bb = a.unsqueeze(-1), b.unsqueeze(-1)
    m = torch.minimum(torch.maximum(m, aa), bb)
    lever = (m - aa) if left else (bb - m)
    # narrow interval (in z units): the conditional mean is the geometric midpoint to O(w); form the
    # lever without cancellation (subtracting two shifted z values would lose ~eps/w relative precision)
    half = 0.5 * torch.log1p((b - a_safe) / a_safe).unsqueeze(-1)
    lev_n = (aa * torch.expm1(half)) if left else (bb * -torch.expm1(-half))
    narrow = pos.unsqueeze(-1) & ((zb - za) < _NARROW)
    lever = torch.where(narrow, lev_n, lever)
    if left and base is not None:                      # kernel origin below the clipped lower limit (0 < tau~ < delta)
        lever = lever + (a - base).unsqueeze(-1)
    return lp + torch.log(lever.clamp(min=1e-300)) - math.log(delta)


def tri_loglik(law, tau_obs, delta, n_quad: int = 64):
    """log E_{v~U(0,delta)} P(tau_obs + v - delta < tau <= tau_obs + v) (triangular kernel).
    Log-normal mixture: exact, as the two linear pieces of the kernel,
      L = (1/delta)[ int_{(k-1)delta}^{k delta} (tau - (k-1)delta) p + int_{k delta}^{(k+1)delta} ((k+1)delta - tau) p ].
    Intensity laws: Gauss-Legendre quadrature in v with `n_quad` nodes."""
    if isinstance(law, LogNormMix):
        t = tau_obs.double()
        right = _mix_log_piece(law, t, t + delta, left=False, delta=delta)
        lo = (t - delta).clamp(min=0.0)
        has_left = t > 0
        a_l = torch.where(has_left, lo, t)
        left = _mix_log_piece(law, a_l, torch.where(has_left, t, t + delta), left=True, delta=delta,
                              base=torch.where(has_left, t - delta, a_l))
        both = torch.logaddexp(left, right)
        comp = torch.where(has_left.unsqueeze(-1), both, right)
        return torch.logsumexp(law.log_w + comp, dim=-1)
    x, w = gauss_legendre(n_quad, device=tau_obs.device)
    v = delta * x                                                    # (Q,)
    b = tau_obs.double().unsqueeze(-1) + v                            # (..., Q)
    a = (b - delta).clamp(min=0.0)
    lp = _broadcast_law(law, n_quad).log_interval(a, b)               # (..., Q)
    return torch.logsumexp(lp + torch.log(w), dim=-1)


def _broadcast_law(law, q):
    """The same law repeated along a new last batch axis of size q (for quadrature)."""
    if isinstance(law, LogNormMix):
        f = lambda t: t.unsqueeze(-2).expand(*t.shape[:-1], q, t.shape[-1])
        return LogNormMix(f(law.log_w), f(law.mu), f(law.log_sigma))
    g = lambda t: t.unsqueeze(-1).expand(*t.shape, q)
    if isinstance(law, RMTPP):
        return RMTPP(g(law.c), g(law.w), law.scale)
    if isinstance(law, THPIntensity):
        return THPIntensity(g(law.c), g(law.beta), law.scale, law.n)
    f = lambda t: t.unsqueeze(-2).expand(*t.shape[:-1], q, t.shape[-1])
    if isinstance(law, WeibullMix):
        return WeibullMix(f(law.log_w), f(law.log_lam), f(law.log_k))
    if isinstance(law, SplineFlow):
        return SplineFlow(f(law.w), f(law.h), f(law.d), law.m, law.s, law.bound)
    raise TypeError(type(law))


def event_loglik(law, tau_obs, score: str, delta: float = 0.0, eps: float = 1e-6,
                 tie_logit=None, tie_tol: float = 1e-9):
    """Per-event log-likelihood of recorded gaps under `score` (shape of tau_obs)."""
    tau = tau_obs.double()
    if score == "floor":
        return law.log_density(tau.clamp(min=eps))
    if score == "point_mass":
        if tie_logit is None:
            raise ValueError("point_mass needs the tie logit")
        tl = tie_logit.double()
        tie = tau <= tie_tol
        dens = law.log_density(torch.where(tie, torch.ones_like(tau), tau))
        return torch.where(tie, torch.nn.functional.logsigmoid(tl), torch.nn.functional.logsigmoid(-tl) + dens)
    if delta <= 0:
        raise ValueError(f"{score} needs delta > 0")
    if score == "interval":
        return law.log_interval(tau, tau + delta)
    if score == "triangular":
        return tri_loglik(law, tau, delta)
    raise ValueError(score)
