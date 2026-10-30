"""Per-event scores of trained EasyTPP models (nine architectures) under floors, the interval score and, for marked
data, the joint score of time and mark (Section 5.2, Supplementary Tables S3-S5). Used by coarse_tpp.rescore_retrained
and coarse_tpp.intensity_exceedance; no training.

Conventions (as each model's own log-likelihood): the sequences enter in EasyTPP's batch layout (float32 times and
gaps, right padding with the pad index); the per-mark intensity is evaluated through the code path each model's
`loglike_loss` uses for its Monte Carlo samples, at times after events 0..N-2 cast to float32, plus EasyTPP's float32
eps per mark, and summed over marks. Integrals are formed in float64. A stored gap <= the data set's tie threshold is a
tie (tau = 0).

Scores per target event (every event but the first of a sequence):
  floor_<eps>   log lambda(x) - Lambda(0, x), x = max(tau, eps), eps = 1e-10, 1e-6, 1e-5
  interval      log P(tau <= s < tau + delta) = -Lambda(0, tau) + log(1 - exp(-Lambda(tau, tau + delta)))
Integration (Rule):
  * every integral by a composite Gauss-Legendre rule graded toward both ends of the range: ten panels with breaks 0,
    1e-6, 1e-4, 0.01, 0.1, 0.5, 0.9, 0.99, 0.9999, 0.999999, 1 of the range, `per_panel` nodes each (32; 128 for AttNHP
    on Email-EU, whose intensity is computed from absolute times in single precision);
  * S2P2 (whose intensity can form peaks a few ms wide at whole seconds and at fixed fractions of a second, which fixed
    rules over the whole gap miss): Lambda(0, tau) by stratified Monte Carlo, one uniform point in each of `strata`
    equal strata of [0, tau] (unbiased); the generator is seeded per batch from `strat_seed` and the batch's first time
    stamps, so a re-run on the same device gives the same value (CPU and CUDA generators give different draws); the
    ranges [tau, x] of the floors (ties only) by the graded rule and the recorded cell by `cell_panels` uniform panels
    with two Gauss-Legendre nodes each.
  * IntensityFree (log-normal mixture of the gap, no intensity): closed form in float64, with the affine of the log gap
    that EasyTPP's runner sets from the train split; no eps is added. Its training clamps gaps at 1e-5.
  * FullyNN (network output Phi = cumulative hazard, intensity dPhi/dtau by autograd): as in EasyTPP's objective the
    compensator is Phi(tau) itself (Phi(0) is not subtracted, recorded as `phi0`); floor log(sum_k dPhi_k/dtau + K eps)
    - Phi(x); interval -Phi(tau) + log(1 - exp(-y)), y the graded-rule integral of the same eps-carrying intensity
    over the cell.
Joint scores (marked data; lambda_k + eps per mark):
  mark_rec = log(lambda_k / lambda) at the recorded gap (0 for a tie); mark_<floor> the same at max(tau, eps);
  joint_<floor> = <floor> + mark_<floor>;
  joint_interval = log int_cell lambda_k(s) exp(-Lambda(s)) ds = interval + log P(k | cell), with
    P(k | cell) = sum_i W_i lambda_k(s_i) exp(-L_i) / sum_i W_i lambda(s_i) exp(-L_i) on the nodes and weights of the
    interval's cell rule and L_i = int_tau^{s_i} lambda by the panel-wise Gauss-Legendre integration matrix on the same
    nodes; so sum_k exp(joint_interval_k) = exp(interval). `cellcheck` = log sum_i W_i lambda(s_i) exp(-L_i) -
    log(1 - exp(-sum_i W_i lambda(s_i))) is the quadrature consistency of the cell (0 for an exact rule);
  joint_interval2 = interval + mark_rec (an approximation).
  IntensityFree: the mark is a categorical head independent of the gap (softmax over K + 1 slots including the pad
  slot, EasyTPP's definition), so every mark term is log p(k). FullyNN (proper_marked_intensities = False): lambda_k =
  (d sum_j Phi_j / dtau) / K, so its mark term is the constant -log K.
"""
from __future__ import annotations

import contextlib
import io
import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import torch

from . import easytpp as E
from .rescore_intensity import batches, make_batch

FLOORS = {"floor_1e-10": 1e-10, "floor_1e-6": 1e-6, "floor_1e-5": 1e-5}
G_CHUNK = {"RMTPP": 256, "NHP": 128, "SAHP": 128, "AttNHP": 16, "THP": 128, "S2P2": 32, "ODETPP": 64}
GRADED_BREAKS = (0.0, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 0.9, 0.99, 0.9999, 0.999999, 1.0)
VERSION = 1


# ------------------------------------------------------------------ integration rules
@dataclass(frozen=True)
class Rule:
    """Integration settings of one scoring call; immutable and passed explicitly (no module state)."""
    per_panel: int = 32
    strata: int = 16384                     # S2P2 compensator; 0 = graded rule for S2P2 too
    strat_seed: int = 0
    cell_panels: int = 1500
    overrides: tuple = field(default=(("email_eu", "AttNHP", 128),))   # (data set, model, per_panel)

    def nodes_per_panel(self, ds: str, model_id: str) -> int:
        return next((n for d, m, n in self.overrides if d == ds and m == model_id), self.per_panel)

    def stratified(self, model_id: str) -> bool:
        return bool(self.strata) and model_id == "S2P2"

    def identity(self, ds: str, model_id: str) -> dict:
        """The settings that determine this model's scores (part of every cache identity)."""
        out = dict(per_panel=self.nodes_per_panel(ds, model_id))
        if self.stratified(model_id):
            out.update(strata=self.strata, strat_seed=self.strat_seed, cell_panels=self.cell_panels)
        return out


@lru_cache(maxsize=None)
def _graded(nodes_per_panel: int):
    x, w = np.polynomial.legendre.leggauss(nodes_per_panel)
    us, ws = [], []
    for a, b in zip(GRADED_BREAKS[:-1], GRADED_BREAKS[1:]):
        us.append(a + (b - a) * (x + 1) / 2); ws.append((b - a) * w / 2)
    return np.concatenate(us), np.concatenate(ws)


def graded_rule(nodes_per_panel: int):
    """Composite Gauss-Legendre rule on [0, 1], panels graded toward both ends (GRADED_BREAKS)."""
    u, w = _graded(int(nodes_per_panel))
    return torch.as_tensor(u, dtype=torch.float64), torch.as_tensor(w, dtype=torch.float64)


@lru_cache(maxsize=None)
def _dense(npanels: int):
    x2, w2 = np.polynomial.legendre.leggauss(2)
    a = np.arange(npanels)[:, None] / npanels
    return (a + (x2[None, :] + 1) / 2 / npanels).ravel(), np.tile(w2 / 2 / npanels, npanels)


def dense_cell_rule(npanels: int):
    """`npanels` uniform panels on [0, 1] with two Gauss-Legendre nodes each."""
    u, w = _dense(int(npanels))
    return torch.as_tensor(u, dtype=torch.float64), torch.as_tensor(w, dtype=torch.float64)


def panels_graded(n: int):
    b = GRADED_BREAKS
    return tuple((b[i], b[i + 1], n) for i in range(len(b) - 1))


def panels_dense(npanels: int):
    return tuple((i / npanels, (i + 1) / npanels, 2) for i in range(npanels))


def _gl_integration_matrix(n):
    """S[i, j] = int_{-1}^{x_i} l_j(x) dx for the n-point Gauss-Legendre nodes x and Lagrange basis l_j (exact for
    polynomials of degree < n), via l_j = sum_m (2m+1)/2 w_j P_m(x_j) P_m and int_{-1}^x P_m = (P_{m+1} - P_{m-1})/(2m+1)."""
    from numpy.polynomial import legendre as L
    x, w = L.leggauss(n)
    P = np.stack([L.legval(x, np.eye(n + 1)[m]) for m in range(n + 1)])
    S = np.tile((x + 1) / 2, (n, 1)).T
    for m in range(1, n):
        S = S + 0.5 * np.outer(P[m + 1] - P[m - 1], P[m])
    return S * w[None, :]


@lru_cache(maxsize=None)
def rule_from_panels(panels: tuple):
    """Nodes u, weights W on [0, 1] and the cumulative matrix C with int_0^{u_i} f ~ sum_j C[i, j] f(u_j)."""
    us, ws, blocks = [], [], []
    for a, b, n in panels:
        x, w = np.polynomial.legendre.leggauss(n)
        us.append(a + (b - a) * (x + 1) / 2); ws.append((b - a) * w / 2)
        blocks.append((b - a) / 2 * _gl_integration_matrix(n))
    u, W = np.concatenate(us), np.concatenate(ws)
    G = len(u); C = np.zeros((G, G))
    off = 0
    for (a, b, n), blk in zip(panels, blocks):
        C[off:off + n, :off] = W[None, :off]
        C[off:off + n, off:off + n] = blk
        off += n
    return u, W, C


def cell_rule(rule: Rule, ds: str, model_id: str):
    """(u, W, C) tensors of the recorded cell under `rule`: the rule of the interval score for the same cell."""
    panels = panels_dense(rule.cell_panels) if rule.stratified(model_id) else panels_graded(rule.nodes_per_panel(ds, model_id))
    return tuple(torch.as_tensor(v, dtype=torch.float64) for v in rule_from_panels(panels))


# ------------------------------------------------------------------ models
def model_class(model_id):
    import importlib
    mod = {"RMTPP": "torch_rmtpp", "NHP": "torch_nhp", "SAHP": "torch_sahp", "AttNHP": "torch_attnhp", "THP": "torch_thp",
           "S2P2": "torch_s2p2", "ODETPP": "torch_ode_tpp", "FullyNN": "torch_fullynn",
           "IntensityFree": "torch_intensity_free"}[model_id]
    return getattr(importlib.import_module(f"easy_tpp.model.torch_model.{mod}"), model_id)


def dt_stats(train_pkl):
    """EasyTPP's TPPDataset.get_dt_stats on the train split (the mean and SD of the raw gaps that the runner sets as
    the affine of IntensityFree's log gap; they are not written to the run configuration)."""
    E.activate("train")
    from easy_tpp.preprocess.dataset import TPPDataset
    d = E.read_file(train_pkl)["train"]
    data = dict(time_seqs=[[x["time_since_start"] for x in q] for q in d], type_seqs=[[x["type_event"] for x in q] for q in d],
                time_delta_seqs=[[x["time_since_last_event"] for x in q] for q in d])
    with contextlib.redirect_stdout(io.StringIO()):
        m, s, _, _ = TPPDataset(data).get_dt_stats()
    return float(m), float(s)


def build_model(model_id: str, cfg: dict, device, train_pkl=None):
    """Rebuild a model from the model_config block of its EasyTPP output configuration through EasyTPP's ModelConfig."""
    E.activate("train")
    from easy_tpp.config_factory import ModelConfig
    m = dict(cfg["model_config"])
    m["gpu"] = 0 if device.type == "cuda" else -1
    m.setdefault("use_mc_samples", True)
    mc = ModelConfig(**m)
    mc.thinning = None
    if model_id == "IntensityFree":
        mc.mean_log_inter_time, mc.std_log_inter_time = dt_stats(train_pkl)
    return model_class(model_id)(mc).to(device).eval()


# ------------------------------------------------------------------ intensities
class Intensity:
    """Per-mark and total intensities of one model on one batch (the event-level pass of S2P2, which does not depend on
    the evaluation times, is computed once per batch)."""

    def __init__(self, model, model_id, batch):
        self.model, self.mid, self.batch = model, model_id, batch
        self._s2p2 = None

    def raw(self, dtimes):
        """lambda_k + eps, [B, N-1, G, K] in the model's precision, at dtimes [B, N-1, G] (float32) after events
        0..N-2."""
        model, mid = self.model, self.mid
        ts, dts, ks, non_pad, attn = self.batch
        if mid == "RMTPP":
            _, rh = model.forward((ts, dts, ks, None, None))
            lam = model.evolve_and_get_intentsity(rh[:, :-1, :], dtimes)
        elif mid == "NHP":
            _, rh = model.forward((ts, dts, ks, None, None))
            lam = model.layer_intensity(model.get_states(rh[:, :-1, :], dtimes))
        elif mid == "SAHP":
            enc = model.forward(ts[:, :-1], dts[:, :-1], ks[:, :-1], attn[:, :-1, :-1])
            lam = model.softplus(model.state_decay(enc[:, :, None, :], dtimes[..., None]))
        elif mid == "AttNHP":
            lam = model.compute_intensities_at_sample_times(ts[:, :-1], dts[:, :-1], ks[:, :-1],
                                                            dtimes + ts[:, :-1, None], attention_mask=attn[:, :-1, :-1])
        elif mid == "THP":
            enc = model.forward(ts[:, :-1], ks[:, :-1], attn[:, :-1, :-1])
            lam = model.softplus(model.compute_states_at_sample_times(enc, dtimes))
        elif mid == "S2P2":
            if self._s2p2 is None:
                fr = model.forward((ts, dts, ks, non_pad, None))
                self._s2p2 = (fr["right_xs_BNLP"][:, :-1], [None if u is None else u[:, :-1, :] for u in fr["right_us_BNH"]])
            lam = model._evolve_and_get_intensity_at_sampled_dts(self._s2p2[0], dtimes, self._s2p2[1])
        elif mid == "ODETPP":
            _, right = model.forward(dts, ks)
            lam = model.layer_intensity(model.compute_states_at_sample_times(right[:, :-1], dtimes))
        elif mid == "FullyNN":
            with torch.enable_grad():
                hs = model.forward(ts[:, :-1], dts[:, :-1], ks[:, :-1]).detach()
                B, N1, P = dtimes.shape
                h = hs[:, :, None, :].expand(B, N1, P, hs.shape[-1])
                x = dtimes.float().detach().clone().requires_grad_(True)
                _, lam = model.layer_intensity(h, x)
            lam = lam.detach()
        else:
            raise ValueError(mid)
        return lam + model.eps

    def marked(self, dtimes):
        return self.raw(dtimes).double()

    def total(self, dtimes):
        """Total intensity [B, N-1, G] at float64 dtimes, evaluated in float32 chunks of G_CHUNK points (summed over
        marks in the model's precision, then float64)."""
        g = G_CHUNK[self.mid]
        return torch.cat([self.raw(dtimes[..., i:i + g].float()).sum(-1).double() for i in range(0, dtimes.shape[-1], g)], -1)

    def marked_chunked(self, dtimes):
        g = G_CHUNK.get(self.mid, 64)
        return torch.cat([self.marked(dtimes[..., i:i + g].float()) for i in range(0, dtimes.shape[-1], g)], dim=-2)


# ------------------------------------------------------------------ closed forms
def _log_diff_ndtr(za, zb):
    """log(Phi(zb) - Phi(za)) for za <= zb (float64, stable in both tails; za may be -inf)."""
    from torch.special import log_ndtr
    lo = log_ndtr(zb) + torch.log1p(-torch.exp(log_ndtr(za) - log_ndtr(zb)))
    hi = log_ndtr(-za) + torch.log1p(-torch.exp(log_ndtr(-zb) - log_ndtr(-za)))
    return torch.where(za > 0, hi, lo)


def ifl_params(model, batch):
    """IntensityFree mixture parameters of targets 1..N-1 (float64): locs, scales, log weights [B, N-1, C]."""
    from easy_tpp.model.torch_model.torch_intensity_free import clamp_preserve_gradients
    ts, dts, ks, non_pad, attn = batch
    ctx = model.forward(dts[:, :-1], ks[:, :-1])
    raw = model.linear(ctx).double()
    C = model.num_mix_components
    locs, log_scales, lw = raw[..., :C], raw[..., C:2 * C], raw[..., 2 * C:]
    log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
    return locs, log_scales.exp(), torch.log_softmax(lw, -1)


def ifl_logpdf(params, x, m=0.0, s=1.0):
    locs, sc, lw = params
    y = (torch.log(x)[..., None] - m) / s
    comp = -0.5 * ((y - locs) / sc) ** 2 - torch.log(sc) - 0.5 * math.log(2 * math.pi)
    return torch.logsumexp(lw + comp, -1) - math.log(s) - torch.log(x)


def ifl_log_interval(params, a, b, m=0.0, s=1.0):
    """log(F(b) - F(a)), a may be 0."""
    locs, sc, lw = params
    la = torch.where(a > 0, torch.log(a.clamp(min=1e-300)), torch.full_like(a, -math.inf))
    za = ((la[..., None] - m) / s - locs) / sc
    zb = ((torch.log(b)[..., None] - m) / s - locs) / sc
    return torch.logsumexp(lw + _log_diff_ndtr(za, zb), -1)


def ifl_log_survival(params, x, m=0.0, s=1.0):
    """log P(gap >= x) at x [..., H]."""
    locs, sc, lw = params
    z = ((torch.log(x)[..., None] - m) / s - locs[..., None, :]) / sc[..., None, :]
    return torch.logsumexp(lw[..., None, :] + torch.special.log_ndtr(-z), -1)


def fullynn_phi(model, batch, xs):
    """FullyNN cumulative hazard Phi (summed over marks) and sum_k dPhi_k/dx + K eps at xs [B, N-1, P] (float64)."""
    ts, dts, ks, non_pad, attn = batch
    with torch.enable_grad():
        hs = model.forward(ts[:, :-1], dts[:, :-1], ks[:, :-1]).detach()
        B, N1, P = xs.shape
        h = hs[:, :, None, :].expand(B, N1, P, hs.shape[-1])
        x = xs.float().detach().clone().requires_grad_(True)
        Phi, dPhi = model.layer_intensity(h, x)
    K = Phi.shape[-1]
    return Phi.sum(-1).detach().double(), (dPhi.sum(-1).detach().double() + K * model.eps)


# ------------------------------------------------------------------ time scores
def _targets(batch, tie_thr):
    ts, dts, ks, non_pad, attn = batch
    tgt = dts[:, 1:].double()
    return torch.where(tgt <= tie_thr, torch.zeros_like(tgt), tgt)


def _finish(out, tau, batch):
    out["tie"] = tau <= 0
    out["mask"] = batch[3][:, 1:]
    return out


def time_scores(ev: Intensity, delta, rule: Rule, ds: str, tie_thr: float, device):
    """Floor and interval scores of every target of the batch (module docstring)."""
    model, mid, batch = ev.model, ev.mid, ev.batch
    tau = _targets(batch, tie_thr)
    xs = {k: torch.clamp(tau, min=e) for k, e in FLOORS.items()}
    ug, wg = graded_rule(rule.nodes_per_panel(ds, mid)); ug, wg = ug.to(device), wg.to(device)
    out = {}
    if mid == "IntensityFree":
        p = ifl_params(model, batch)
        m, s = float(model.mean_log_inter_time), float(model.std_log_inter_time)
        for k in FLOORS:
            out[k] = ifl_logpdf(p, xs[k], m, s)
        if delta:
            out["interval"] = ifl_log_interval(p, tau, tau + delta, m, s)
        return _finish(out, tau, batch)
    if mid == "FullyNN":
        pts = [xs[k] for k in FLOORS] + [tau, torch.zeros_like(tau)]
        if delta:
            pts += list(torch.unbind(tau[..., None] + delta * ug, -1))
        Phi, lam = fullynn_phi(model, batch, torch.stack(pts, -1))
        for j, k in enumerate(FLOORS):
            out[k] = torch.log(lam[..., j]) - Phi[..., j]
        if delta:
            y = ((lam[..., 5:] * wg).sum(-1) * delta).clamp(min=1e-300)
            out["interval"] = -Phi[..., 3] + torch.log(-torch.expm1(-y))
        out["phi0"] = Phi[..., 4]
        return _finish(out, tau, batch)
    if rule.stratified(mid) and delta:
        return _finish(_strat_scores(ev, tau, xs, delta, rule, (ug, wg), device), tau, batch)
    ranges = [(torch.zeros_like(tau), xs[k]) for k in FLOORS] + [(torch.zeros_like(tau), tau)]
    if delta:
        ranges.append((tau, tau + delta))
    n = len(ug)
    pts = [a[..., None] + (b - a)[..., None] * ug for a, b in ranges]
    pts.append(torch.stack([xs[k] for k in FLOORS], -1))
    lam = ev.total(torch.cat(pts, -1))
    Lam = [((lam[..., r * n:(r + 1) * n] * wg).sum(-1)) * (b - a) for r, (a, b) in enumerate(ranges)]
    lam_pt = lam[..., len(ranges) * n:]
    for j, k in enumerate(FLOORS):
        out[k] = torch.log(lam_pt[..., j]) - Lam[j]
    if delta:
        y = Lam[4].clamp(min=1e-300)
        out["interval"] = -Lam[3] + torch.log(-torch.expm1(-y))
    return _finish(out, tau, batch)


def strat_generator(rule: Rule, batch, device):
    """Generator of the stratified draws of one batch: seeded from rule.strat_seed and the batch's first time stamps."""
    ts = batch[0]
    gen = torch.Generator(device=device)
    gen.manual_seed(int(rule.strat_seed) * 1_000_003 + int(float(ts[0, :8].double().sum().item() * 1000) % 1_000_000_007))
    return gen


def _strat_scores(ev, tau, xs, delta, rule, graded, device):
    M = rule.strata
    gen = strat_generator(rule, ev.batch, device)
    U = (torch.arange(M, device=device, dtype=torch.float64)
         + torch.rand(*tau.shape, M, device=device, dtype=torch.float64, generator=gen)) / M
    lam = ev.total(tau[..., None] * U)
    base = tau * lam.mean(-1)
    del lam, U
    ug, wg = graded
    uc, wc = dense_cell_rule(rule.cell_panels); uc, wc = uc.to(device), wc.to(device)
    ranges = [(tau, xs[k]) for k in FLOORS] + [(tau, tau + delta)]
    rules = [(ug, wg)] * len(FLOORS) + [(uc, wc)]
    pts = [a[..., None] + (b - a)[..., None] * u for (a, b), (u, _) in zip(ranges, rules)]
    pts.append(torch.stack([xs[k] for k in FLOORS], -1))
    lam = ev.total(torch.cat(pts, -1))
    Lam, off = [], 0
    for (a, b), (u, w) in zip(ranges, rules):
        Lam.append(((lam[..., off:off + len(u)] * w).sum(-1)) * (b - a)); off += len(u)
    lam_pt = lam[..., off:]
    out = {}
    for j, k in enumerate(FLOORS):
        out[k] = torch.log(lam_pt[..., j]) - (base + Lam[j])
    y = Lam[-1].clamp(min=1e-300)
    out["interval"] = -base + torch.log(-torch.expm1(-y))
    return out


# ------------------------------------------------------------------ joint scores
def mark_cell_terms(lam_cell, kt, delta, u_W_C):
    """From lambda_k + eps at the cell nodes [B, N-1, G, K]: log P(k_target | cell) and the cellcheck diagnostic."""
    u, W, C = (t.to(lam_cell.device) for t in u_W_C)
    tot = lam_cell.sum(-1)
    Lc = delta * torch.einsum("bng,hg->bnh", tot, C)
    logw = torch.log(W * delta)
    lraw = torch.logsumexp(logw[None, None, :, None] + torch.log(lam_cell) - Lc[..., None], dim=-2)
    lsum = torch.logsumexp(lraw, -1)
    cond = lraw.gather(-1, kt[..., None]).squeeze(-1) - lsum
    y = (delta * (tot * W).sum(-1)).clamp(min=1e-300)
    return cond, lsum - torch.log(-torch.expm1(-y))


def joint_scores(ev: Intensity, delta, rule: Rule, ds: str, tie_thr: float, device):
    """time_scores plus the mark and joint keys (module docstring)."""
    o = time_scores(ev, delta, rule, ds, tie_thr, device)
    model, mid, batch = ev.model, ev.mid, ev.batch
    ts, dts, ks, non_pad, attn = batch
    K = model.num_event_types
    kt = ks[:, 1:].clamp(max=K - 1)                     # pad slots: any valid index, masked later
    tau = _targets(batch, tie_thr)
    if mid == "IntensityFree":
        ctx = model.forward(dts[:, :-1], ks[:, :-1])
        lp = torch.log_softmax(model.mark_linear(ctx), dim=-1).double().gather(-1, kt[..., None]).squeeze(-1)
        marks = {k: lp for k in FLOORS}; marks["rec"] = lp
        cond = lp if delta else None
        check = torch.zeros_like(lp) if delta else None
    else:
        pts = [torch.clamp(tau, min=e) for e in FLOORS.values()] + [tau]
        u_W_C = cell_rule(rule, ds, mid)
        x = torch.stack(pts, -1)
        if delta:
            x = torch.cat([x, tau[..., None] + delta * u_W_C[0].to(tau.device)], -1)
        lam = ev.marked_chunked(x)
        lmk = torch.log(lam[..., :4, :].gather(-1, kt[..., None, None].expand(*kt.shape, 4, 1)).squeeze(-1)) \
            - torch.log(lam[..., :4, :].sum(-1))
        marks = {k: lmk[..., j] for j, k in enumerate(FLOORS)}; marks["rec"] = lmk[..., 3]
        cond = check = None
        if delta:
            cond, check = mark_cell_terms(lam[..., 4:, :], kt, delta, u_W_C)
    for k in FLOORS:
        o[f"mark_{k}"] = marks[k]
        o[f"joint_{k}"] = o[k] + marks[k]
    o["mark_rec"] = marks["rec"]
    if delta:
        o["joint_interval"] = o["interval"] + cond
        o["joint_interval2"] = o["interval"] + marks["rec"]
        o["cellcheck"] = check
    return o


# ------------------------------------------------------------------ split means
def score_split(model, model_id, seqs, pad_id, delta, rule: Rule, ds: str, tie_thr: float, device, joint=False):
    """Means over target events of every score, with tie / non-tie means and the count of non-finite values."""
    acc = {}
    n = n_tie = 0
    fn = joint_scores if joint else time_scores
    with torch.no_grad():
        for bs in batches(seqs):
            b = make_batch(bs, pad_id, device)
            o = fn(Intensity(model, model_id, b), delta, rule, ds, tie_thr, device)
            m = o["mask"]; tie = o["tie"] & m
            n += int(m.sum()); n_tie += int(tie.sum())
            for k in [k for k in o if k not in ("tie", "mask")]:
                v = o[k]
                acc.setdefault(k, [0.0, 0.0, 0.0, 0])
                acc[k][0] += float(v[m].sum()); acc[k][1] += float(v[tie].sum()); acc[k][2] += float(v[m & ~o["tie"]].sum())
                acc[k][3] += int((~torch.isfinite(v[m])).sum())
    out = {"n": n, "n_tie": n_tie}
    for k, (s, st, sn, nbad) in acc.items():
        out[k] = s / n if n else None
        out[f"{k}_tie"] = st / n_tie if n_tie else None
        out[f"{k}_nontie"] = sn / (n - n_tie) if n > n_tie else None
        out[f"{k}_nonfinite"] = nbad
    return out
