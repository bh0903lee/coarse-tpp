"""Encoders and heads.

Encoders (history = recorded gaps iet[0..j], optionally marks, for the law of iet[j+1]):
  gru          GRU on log(tau + eps) (the IntensityFree input of EasyTPP), optionally concatenated with a mark
               embedding of width d
  transformer  causal Transformer in the THP style: input = linear(standardized log gap) + sinusoidal encoding of
               the time since the window start (in train-median units); pre-LN layers
Heads:
  mix          log-normal mixture (IntensityFree, Shchur et al. 2020), K components, optionally with a tie logit
               (point mass at zero: pi = sigmoid(linear(h)))
  rmtpp        lambda(s) = exp(v.h + b + w s / scale), w a global scalar (Du et al. 2016), bounded as
               w = w_max tanh(g) with w_max = 30 scale / max train gap, so that the exponent stays <= 30 on the
               train range (an unbounded w overflows float32 gradients on heavy-tailed data)
  thp          lambda(s) = softplus(v.h + b + beta s / scale), beta a global scalar (Zuo et al. 2020)
  weib         Weibull mixture, K components: log scale = m + s loc (raw log time, standardized as the mixture head),
               shape in [0.05, 20] through a sigmoid of the log shape (initialized at shape 1)
  flow         rational-quadratic spline flow (Durkan et al. 2019, K bins) on the standardized log gap with a standard
               normal base; the spline domain reaches from FLOW_LO (below the smallest floor 1e-10) to 100 x the
               largest training gap; identity outside; initialized near the identity
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import lik

EPS32 = torch.finfo(torch.float32).eps
FLOW_LO = 1e-11          # flow head: the spline domain reaches down to this gap (below the smallest floor 1e-10)


def log_gap(iet):
    """log(tau + eps), the IntensityFree input."""
    return torch.log(iet + EPS32)


def std_log_gap(iet, stats):
    """The same feature standardized with the train log-target mean / std."""
    return (log_gap(iet) - stats.log_mean) / stats.log_std


# ------------------------------------------------------------------ encoders
class GRUEncoder(nn.Module):
    """GRU input = [log(tau + eps), mark embedding (d)] with marks, or the scalar alone (time only)."""

    def __init__(self, d_model=64, n_types=1, marks=False):
        super().__init__()
        self.marks = marks
        self.type_emb = nn.Embedding(n_types + 1, d_model, padding_idx=n_types) if marks else None
        self.gru = nn.GRU(1 + (d_model if marks else 0), d_model, batch_first=True)
        self.d_out = d_model

    def forward(self, batch, stats, feats=None):
        x = log_gap(batch.iet).unsqueeze(-1)
        if self.marks:
            x = torch.cat([x, self.type_emb(batch.marks)], dim=-1)
        return self.gru(x)[0]


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0, n_types=1, marks=False):
        super().__init__()
        self.inp = nn.Linear(1, d_model)
        self.type_emb = nn.Embedding(n_types + 1, d_model) if marks else None
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = self.d_out = d_model

    def _pos(self, t):
        half = self.d_model // 2
        freq = torch.exp(-math.log(1e4) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        ang = t.unsqueeze(-1) * freq
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, batch, stats, feats=None):
        x = self.inp(std_log_gap(batch.iet, stats).unsqueeze(-1))
        if self.type_emb is not None:
            x = x + self.type_emb(batch.marks)
        t = torch.cumsum(batch.iet * batch.mask, dim=1) - batch.iet[:, :1] * batch.mask[:, :1]
        x = x + self._pos(t / stats.median)
        L = x.shape[1]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        pad = batch.mask < 0.5
        h = self.tr(x, mask=causal, src_key_padding_mask=pad, is_causal=True)
        return self.norm(h)


# ------------------------------------------------------------------ mixture head
def clamp_preserve_gradients(x, lo, hi):
    return x + (x.clamp(lo, hi) - x).detach()


MIX_PARAMS = ("standard", "sigma_cap", "sigma_sigmoid", "bounded_mean")


class MixtureHead(nn.Module):
    """sum_k w_k LogNormal(mu_k, sigma_k) on raw gaps. One linear layer gives [loc | scale | logits] (bounded_mean:
    [v | u | logits]); the parameterizations differ only in how these map to (w, mu, sigma):
      standard       (IntensityFree) mu = m + s loc, log sigma = log s + clamp(log_scale, -5, 3) with gradients
                     preserved outside the range; (m, s) = train log-target mean / std
      sigma_cap      standard with the upper end of the clamp lowered so that sigma <= sig_max
      sigma_sigmoid  standard mean; sigma = sig_min + (sig_max - sig_min) sigmoid(u), sig_min = s e^-5
      bounded_mean   log b_k = l_b + (u_b - l_b) sigmoid(v_k / c), sigma_k = sig_min + (sig_max - sig_min)
                     sigmoid(u_k), mu_k = log a + log b_k - sigma_k^2 / 2 with a = train median gap, so every
                     component mean lies in [b_lo, b_hi]
    Lower bounds of the standard head (the component-bounds experiment, Supplementary S4): ls_lo replaces the lower
    end -5 of the log_scale clamp (sigma_k >= s e^ls_lo; the default -5 leaves the head unchanged) and mu_lo floors
    the raw log-time location (value clamped, gradient passed through). Both are defined for `standard` only.
    """

    def __init__(self, d_in: int, K: int = 8, param: str = "standard", log_mean: float = 0.0, log_std: float = 1.0,
                 sig_max: float = 4.0, a: float | None = None, b_lo: float | None = None, b_hi: float | None = None,
                 mean_pre_scale: float = 1.0, ls_lo: float = -5.0, mu_lo: float | None = None):
        super().__init__()
        if param not in MIX_PARAMS:
            raise ValueError(param)
        self.K, self.param = K, param
        if (ls_lo != -5.0 or mu_lo is not None) and param != "standard":
            raise ValueError("ls_lo / mu_lo are defined for the standard parameterization")
        if ls_lo >= 3.0:
            raise ValueError(f"ls_lo {ls_lo} must be below the upper clamp 3.0")
        self.ls_lo, self.mu_lo = float(ls_lo), (None if mu_lo is None else float(mu_lo))
        self.linear = nn.Linear(d_in, 3 * K)
        self.ls_hi = 3.0 if param == "standard" else min(3.0, math.log(sig_max) - math.log(log_std))
        if param == "sigma_cap" and self.ls_hi <= -5.0:
            raise ValueError(f"sig_max {sig_max} is below the lower sigma bound s*e^-5")
        if param in ("sigma_sigmoid", "bounded_mean"):
            self.sig_min, self.sig_max = log_std * math.exp(-5.0), float(sig_max)
            if not (self.sig_min < self.sig_max):
                raise ValueError(f"sig_min {self.sig_min} >= sig_max {sig_max}")
        if param == "bounded_mean":
            if not (a and b_lo and b_hi and 0 < b_lo < b_hi):
                raise ValueError(f"invalid mean bounds a={a} b_lo={b_lo} b_hi={b_hi}")
            self.log_a = math.log(a)
            self.lb, self.ub = math.log(b_lo / a), math.log(b_hi / a)
            self.mean_pre_scale = float(mean_pre_scale)
        self.register_buffer("m", torch.tensor(float(log_mean)))
        self.register_buffer("s", torch.tensor(float(log_std)))

    def forward(self, h: torch.Tensor):
        """h (..., d) -> (log_w, mu, log_sigma) in raw log-time, each (..., K)."""
        out = self.linear(h)
        K = self.K
        if self.param == "bounded_mean":
            log_b = self.lb + (self.ub - self.lb) * torch.sigmoid(out[..., :K] / self.mean_pre_scale)
            sig = self.sig_min + (self.sig_max - self.sig_min) * torch.sigmoid(out[..., K:2 * K])
            log_sigma = torch.log(sig)
            log_w = F.log_softmax(out[..., 2 * K:], dim=-1)
            mu = self.log_a + log_b - 0.5 * sig * sig
            return log_w, mu, log_sigma
        loc = out[..., :K]
        log_w = F.log_softmax(out[..., 2 * K:], dim=-1)
        mu = self.m + self.s * loc
        if self.mu_lo is not None:
            mu = clamp_preserve_gradients(mu, self.mu_lo, float("inf"))
        if self.param == "sigma_sigmoid":
            log_sigma = torch.log(self.sig_min + (self.sig_max - self.sig_min) * torch.sigmoid(out[..., K:2 * K]))
        else:
            log_sigma = torch.log(self.s) + clamp_preserve_gradients(out[..., K:2 * K], self.ls_lo, self.ls_hi)
        return log_w, mu, log_sigma


# ------------------------------------------------------------------ model
class Model(nn.Module):
    def __init__(self, encoder, head: str, d: int, stats, K: int = 8, tie: bool = False, mark_types: int | None = None,
                 mix_param: str = "standard", mix_kwargs: dict | None = None):
        super().__init__()
        self.mark = nn.Linear(d, int(mark_types)) if mark_types else None     # categorical mark head
        self.encoder, self.kind, self.scale = encoder, head, float(stats.median)
        if head == "mix":
            self.head = MixtureHead(d, K=K, param=mix_param, log_mean=stats.log_mean, log_std=stats.log_std,
                                    **(mix_kwargs or {}))
        elif head == "weib":
            self.head = nn.Linear(d, 3 * K)
            self.K, self.m_log, self.s_log = K, float(stats.log_mean), float(stats.log_std)
            with torch.no_grad():                 # shape 1 (exponential components) at the start
                frac = (0.0 - lik.LOG_K_LO) / (lik.LOG_K_HI - lik.LOG_K_LO)
                self.head.bias[K:2 * K].fill_(math.log(frac / (1 - frac)))
        elif head == "flow":
            self.head = nn.Linear(d, 3 * K - 1)
            self.K, self.m_log, self.s_log = K, float(stats.log_mean), float(stats.log_std)
            lo, hi = math.log(FLOW_LO), math.log(max(float(stats.max_train_iet), FLOW_LO)) + math.log(100.0)
            self.flow_bound = max(abs(lo - self.m_log), abs(hi - self.m_log)) / self.s_log
            with torch.no_grad():                 # near-identity spline at the start
                self.head.weight.mul_(0.01); self.head.bias.zero_()
                self.head.bias[2 * K:].fill_(math.log(math.expm1(1.0 - lik.RQS_MIN_D)))
        elif head in ("rmtpp", "thp"):
            self.head = nn.Linear(d, 1)
            self.g = nn.Parameter(torch.tensor(0.0))
            self.w_max = 30.0 * self.scale / max(float(stats.max_train_iet), self.scale)
            with torch.no_grad():             # start near the train rate: lambda ~ 1 / median
                self.head.bias.fill_(-math.log(self.scale) if head == "rmtpp"
                                     else math.log(math.expm1(1.0 / self.scale)) if 1.0 / self.scale < 30 else 1.0 / self.scale)
        else:
            raise ValueError(head)
        self.tie = nn.Linear(d, 1) if tie else None
        if tie and head != "mix":
            raise ValueError("the point-mass tie head is defined for the mixture head")

    def forward(self, batch, stats, marks: bool = False):
        """(law, tie_logit or None) for the gaps iet[:, 1:]; with marks=True also the log-probabilities of the next
        mark, (B, L-1, n_types) float64 (needs a mark head)."""
        h = self.encoder(batch, stats)[:, :-1]
        out = self._laws(h)
        if marks:
            if self.mark is None:
                raise ValueError("the model has no mark head (config mark_head)")
            return out + (torch.log_softmax(self.mark(h).double(), dim=-1),)
        return out

    def _laws(self, h):
        if self.kind == "mix":
            law = lik.LogNormMix(*self.head(h))
        elif self.kind == "weib":
            o, K = self.head(h), self.K
            log_k = lik.LOG_K_LO + (lik.LOG_K_HI - lik.LOG_K_LO) * torch.sigmoid(o[..., K:2 * K])
            law = lik.WeibullMix(torch.log_softmax(o[..., 2 * K:], -1), self.m_log + self.s_log * o[..., :K], log_k)
        elif self.kind == "flow":
            w, hh, dd = lik.rqs_params(self.head(h), self.K)
            law = lik.SplineFlow(w, hh, dd, self.m_log, self.s_log, self.flow_bound)
        elif self.kind == "rmtpp":
            law = lik.RMTPP(self.head(h).squeeze(-1), (self.w_max * torch.tanh(self.g)).expand(h.shape[:-1]), self.scale)
        else:
            law = lik.THPIntensity(self.head(h).squeeze(-1), self.g.expand(h.shape[:-1]), self.scale)
        tl = self.tie(h).squeeze(-1) if self.tie is not None else None
        return law, tl


def component_bounds(cfg: dict, stats) -> dict:
    """Lower bounds of the mixture head from the cell config, in data units: sigma_min (raw log-time scale) becomes
    the log_scale clamp ls_lo = log(sigma_min / s); mu_min is the floor of the raw log-time location. Absent keys: no
    change (the head is bit-identical to the unbounded one)."""
    out = {}
    if cfg.get("sigma_min") is not None:
        out["ls_lo"] = math.log(float(cfg["sigma_min"])) - math.log(float(stats.log_std))
    if cfg.get("mu_min") is not None:
        out["mu_lo"] = float(cfg["mu_min"])
    if out and (cfg["head"] != "mix" or cfg.get("mix_param", "standard") != "standard"):
        raise ValueError("sigma_min / mu_min are defined for the standard mixture head")
    return out


def mix_kwargs(cfg: dict, stats) -> dict:
    """Keyword arguments of the mixture-head parameterization named in cfg["mix_param"], plus the lower bounds."""
    p = cfg.get("mix_param", "standard")
    cb = component_bounds(cfg, stats)            # raises when bounds are given for another parameterization
    if p == "standard":
        return cb
    kw = dict(sig_max=float(cfg.get("sig_max", 4.0)))
    if p == "bounded_mean":
        floor = 1e-6
        kw.update(a=stats.median, b_hi=float(cfg.get("mean_hi_mult", 10.0)) * stats.q99,
                  b_lo=0.5 * min(floor, stats.q001))
        mps = cfg.get("mean_pre_scale", 1.0)
        if mps == "match":            # slope of the mean coordinate at v = 0 equals the standard head's s
            mps = math.log(kw["b_hi"] / kw["b_lo"]) / (4.0 * stats.log_std)
        kw["mean_pre_scale"] = float(mps)
    return kw


def build(cfg: dict, stats, n_types: int = 1) -> Model:
    """Encoders see the recorded gaps (and the marks when encoder.marks is true; the likelihood is time-only)."""
    enc = dict(cfg["encoder"]); kind = enc.pop("type")
    marks = bool(enc.pop("marks", False))
    if kind == "gru":
        e = GRUEncoder(n_types=n_types, marks=marks, d_model=enc.get("d_model", 64))
    elif kind == "transformer":
        e = TransformerEncoder(n_types=n_types, marks=marks, **enc)
    else:
        raise ValueError(kind)
    if cfg.get("mark_head") and not marks:
        raise ValueError("mark_head needs encoder.marks = true (the mark history is the input of the mark law)")
    if cfg["head"] != "mix" and (cfg.get("sigma_min") is not None or cfg.get("mu_min") is not None):
        raise ValueError("sigma_min / mu_min are defined for the standard mixture head")
    return Model(e, cfg["head"], e.d_out, stats, K=cfg.get("K", 8), tie=cfg["treatment"] == "point_mass",
                 mark_types=n_types if cfg.get("mark_head") else None, mix_param=cfg.get("mix_param", "standard"),
                 mix_kwargs=mix_kwargs(cfg, stats) if cfg["head"] == "mix" else None)
