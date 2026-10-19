"""Likelihoods of coarse_tpp.lik against direct numerical integration (float64)."""
import math

import numpy as np
import torch
from scipy import integrate, stats

from coarse_tpp import lik

T = lambda *a: torch.tensor(a, dtype=torch.float64)


def mix():
    lw = torch.log(T(0.2, 0.5, 0.3)); mu = T(-3.0, 0.5, 2.0); ls = torch.log(T(0.05, 0.8, 2.5))
    return lik.LogNormMix(lw, mu, ls), (lw.exp().numpy(), mu.numpy(), ls.exp().numpy())


def mix_cdf(x, p):
    w, mu, s = p
    return sum(wk * stats.lognorm.cdf(x, sk, scale=math.exp(mk)) if x > 0 else 0.0 for wk, mk, sk in zip(w, mu, s))


def mix_pdf(x, p):
    w, mu, s = p
    return sum(wk * stats.lognorm.pdf(x, sk, scale=math.exp(mk)) for wk, mk, sk in zip(w, mu, s))


def test_mixture_density_interval_cdf():
    law, p = mix()
    for t in (1e-8, 0.05, 1.0, 30.0, 1e4):
        assert abs(float(law.log_density(T(t))) - math.log(mix_pdf(t, p))) < 1e-9
    for a, b in ((0.0, 0.01), (0.0, 1.0), (0.04, 0.05), (1.0, 1.0 + 1e-7), (100.0, 1e3), (1e5, 1e6)):
        ref = mix_cdf(b, p) - mix_cdf(a, p)
        got = float(law.log_interval(T(a), T(b)).exp())
        assert abs(got - ref) <= 1e-9 * max(ref, 1e-300) + 1e-14, (a, b, got, ref)


def test_triangular_identity_and_limits():
    law, p = mix()
    d = 0.1
    for k in (0, 0.3, 0.7, 1, 1.37, 7, 10**6):
        ref = integrate.quad(lambda t: mix_pdf(t, p) * max(0.0, 1 - abs(t - k * d) / d), max(0, (k - 1) * d), (k + 1) * d,
                             points=[k * d], limit=400, epsabs=1e-13)[0]
        got = float(lik.event_loglik(law, T(k * d), "triangular", delta=d).exp())
        assert abs(got - ref) < 1e-8 * max(ref, 1e-12) + 1e-13, (k, got, ref)
    ks = torch.arange(0, 2_000_000, dtype=torch.float64) * d       # probabilities over the grid sum to one
    lawb = lik.LogNormMix(law.log_w.expand(len(ks), 3), law.mu.expand(len(ks), 3), law.log_sigma.expand(len(ks), 3))
    tot = float(lik.event_loglik(lawb, ks, "triangular", delta=d).exp().sum())
    assert abs(tot - (mix_cdf(2e5, p))) < 1e-8, tot
    nl = lik.LogNormMix(torch.log(T(1.0)), T(math.log(0.137)), torch.log(T(1e-3)))   # a narrow component in one cell
    got = [float(lik.event_loglik(nl, T(k * d), "triangular", delta=d).exp()) for k in (0, 1, 2)]
    assert abs(got[1] - (1 - 0.37)) < 2e-3 and abs(got[2] - 0.37) < 2e-3 and got[0] < 1e-12


def test_small_delta_limit_matches_density():
    law, _ = mix()
    t, d = 2.0, 1e-7
    for s in ("interval", "triangular"):
        v = float(lik.event_loglik(law, T(t), s, delta=d)) - math.log(d)
        assert abs(v - float(law.log_density(T(t)))) < 1e-4, s


def test_jensen_uniform_jitter_bound():
    law, _ = mix()
    d, k = 0.2, 3
    u = torch.rand(200000, dtype=torch.float64) * d
    mc = float(law.log_density(k * d + u).mean()) + math.log(d)
    assert mc <= float(lik.event_loglik(law, T(k * d), "interval", delta=d)) + 1e-9


def test_point_mass_and_floor():
    law, _ = mix()
    tl = T(0.3)
    assert abs(float(lik.event_loglik(law, T(0.0), "point_mass", tie_logit=tl)) - math.log(1 / (1 + math.exp(-0.3)))) < 1e-12
    v = float(lik.event_loglik(law, T(2.0), "point_mass", tie_logit=tl))
    assert abs(v - (math.log(1 / (1 + math.exp(0.3))) + float(law.log_density(T(2.0))))) < 1e-12
    assert float(lik.event_loglik(law, T(0.0), "floor", eps=1e-6)) == float(law.log_density(T(1e-6)))


def test_intensity_laws():
    s = 3.0
    for law, lam in ((lik.RMTPP(T(-1.0), T(0.4), s), lambda x: math.exp(-1.0 + 0.4 * x / s)),
                     (lik.RMTPP(T(-1.0), T(-0.2), s), lambda x: math.exp(-1.0 - 0.2 * x / s)),
                     (lik.THPIntensity(T(0.2), T(0.5), s, n_quad=64), lambda x: math.log1p(math.exp(0.2 + 0.5 * x / s)))):
        Lam = lambda x: integrate.quad(lam, 0, x, limit=200)[0]
        for x in (1e-6, 0.3, 4.0):
            assert abs(float(law.log_density(T(x))) - (math.log(lam(x)) - Lam(x))) < 1e-7
        a, b = 0.5, 2.5
        ref = math.exp(-Lam(a)) - math.exp(-Lam(b))
        assert abs(float(law.log_interval(T(a), T(b)).exp()) - ref) < 1e-8
        assert abs(float(law.log_interval(T(0.0), T(b)).exp()) - (1 - math.exp(-Lam(b)))) < 1e-8
        tri = float(lik.event_loglik(law, T(1.0), "triangular", delta=0.5).exp())
        reft = integrate.quad(lambda t: lam(t) * math.exp(-Lam(t)) * max(0, 1 - abs(t - 1.0) / 0.5), 0.5, 1.5, points=[1.0])[0]
        assert abs(tri - reft) < 1e-6


def test_rmtpp_interval_in_a_decaying_tail():
    c, w = T(0.0).requires_grad_(), T(-0.99).requires_grad_()
    v = lik.RMTPP(c, w, 1.0).log_interval(T(29.0), T(29.0001))
    gc, gw = torch.autograd.grad(v, (c, w))
    assert abs(float(v) - (-38.930490882)) < 1e-6
    assert abs(float(gw) - 27.979745948) < 1e-5 and abs(float(gc) + 0.0101010101) < 1e-8


def test_thp_compensator_monotone_and_exact():
    law = lik.THPIntensity(T(0.0), T(-1.0), 1.0)
    vals = [float(law.Lambda(T(s))) for s in (10.0, 1000.0, 6000.0)]
    assert vals[0] <= vals[1] <= vals[2] and abs(vals[2] - math.pi ** 2 / 12) < 1e-12
    for c0, b0, sc in [(0.5, 0.7, 2.0), (-45.0, 1.0, 1.0), (60.0, -0.5, 1.0), (-3.0, 1e-9, 1.0)]:
        law = lik.THPIntensity(T(c0), T(b0), sc)
        f = lambda u: np.logaddexp(0.0, c0 + b0 * u / sc)
        for a, b in [(0.0, 1e-4), (0.0, 7.0), (3.0, 3.0001), (0.0, 1000.0)]:
            ref = integrate.quad(f, a, b, epsabs=0, epsrel=1e-12, limit=500,
                                 points=[x for x in (1.0, 10.0, 100.0) if a < x < b] or None)[0]
            got = float(law.Lambda_between(T(a), T(b)))
            assert abs(got - ref) <= 1e-10 * ref, (c0, b0, a, b, got, ref)


def test_thp_gradients_finite_on_a_batch():
    torch.manual_seed(0)
    c = (torch.randn(64, dtype=torch.float64) * 5).requires_grad_()
    be = (torch.randn(64, dtype=torch.float64) * 3).requires_grad_()
    tau = torch.rand(64, dtype=torch.float64) * 50
    law = lik.THPIntensity(c, be, 1.3)
    v = (lik.event_loglik(law, tau, "interval", delta=0.1) + law.log_density(tau + 1e-3)).sum()
    gc, gb = torch.autograd.grad(v, (c, be))
    assert torch.isfinite(v) and torch.isfinite(gc).all() and torch.isfinite(gb).all()


def test_thp_slope_gets_a_gradient_at_zero_from_compensators_alone():
    c = T(0.3).requires_grad_(); beta = T(0.0).requires_grad_()
    law = lik.THPIntensity(c, beta, 1.0)
    v = lik.event_loglik(law, T(1.0), "interval", delta=0.5)
    g = torch.autograd.grad(v, beta)[0]
    eps = 1e-6
    num = (float(lik.event_loglik(lik.THPIntensity(T(0.3), T(eps), 1.0), T(1.0), "interval", delta=0.5))
           - float(lik.event_loglik(lik.THPIntensity(T(0.3), T(-eps), 1.0), T(1.0), "interval", delta=0.5))) / (2 * eps)
    assert abs(float(g) - num) < 1e-5 and abs(float(g)) > 1e-3


def test_gradients_finite_at_ties_and_extremes():
    lw = torch.log(T(0.5, 0.5)).requires_grad_(); mu = T(-12.0, 3.0).requires_grad_(); ls = T(-3.0, 1.0).requires_grad_()
    tau = T(0.0, 0.0, 1e-3, 50.0, 1e7)
    for s in ("interval", "triangular"):
        lawb = lik.LogNormMix(lw.expand(5, 2), mu.expand(5, 2), ls.expand(5, 2))
        v = lik.event_loglik(lawb, tau, s, delta=0.01).sum()
        g = torch.autograd.grad(v, (lw, mu, ls))
        assert torch.isfinite(v) and all(torch.isfinite(x).all() for x in g), s
