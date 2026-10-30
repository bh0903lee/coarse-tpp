"""Threshold-exceedance forecasts and remaining time on the process logs: does the choice of score change a
probability prediction made from the model? (Section 5.3, "Consequence for probability prediction"; Supplementary S4,
Table S12.) Exploratory, no training. The horizons are derived from the data and illustrate deadline (SLA-style)
monitoring; they are not observed service-level agreements, and no alarm policy or operational utility is evaluated.
The file and command names (alarms, sla_alarms) are kept for compatibility.

    python -m coarse_tpp.alarms score RUN_DIR [--base TRI_RUN] [--tasks sla,rt] [--datasets bpi_sepsis ...]
                                   [--selected] [--only CELL ...] [--limit N] [--force] [--retry-quarantined] [--cpu]
    python -m coarse_tpp.alarms report RUN_DIR [--base TRI_RUN] [--datasets ...]

RUN_DIR is a process_logs run (seven logs x six treatments x {transformer, gru} x two learning rates x five seeds;
activities are encoder inputs, no mark head); --base may add a triangular_recorded run, whose triangular cells on
the same logs form a second variant of every unit. Cells are read through coarse_tpp.analysis.load_cells (status ok,
complete seeds). Outputs go to RUN_DIR/alarms (--out).

Unit = (data set, encoder); candidates = (treatment arm, learning rate). A unit (variant) is complete only when every
planned candidate (six treatments [+ triangular] x two learning rates) has all planned seeds 0-4; every rule then
averages exactly those seeds; an incomplete unit is pending (missing items listed), no selection.
Selection rules (dev split):
  F      highest seed mean of result['dev']['floor_1e-6'] (floor 1e-6 score of best_C.pt, stored by coarse_tpp.train)
  C      highest seed mean of result['dev']['C']
  F_own  (auxiliary) only the floor_1e-6 candidates move to best_own.pt (early stopping on the floor objective), ranked
         by their dev floor 1e-6 recomputed here on that checkpoint; every other candidate keeps best_C.pt and its
         stored dev floor 1e-6. Each candidate carries its checkpoint; the chosen (candidate, checkpoint) is evaluated.
Every unit is reported without the triangular candidate and, when it exists and is complete, with it.

Threshold exceedance (primary; "SLA" in the file names). For every scored test target (masks of coarse_tpp.train.evaluate)
and grid point h,
p_h = P(recorded next gap g > h) = 1 - F(h + delta) (g > h <=> s >= h + delta, s = latent time since the last record),
F the distribution function of the predictive law of s under the treatment's readout (coarse_tpp.predictive.cdf):
  raw          F = Q (the mixture distribution function)
  smooth       F(x) = (1/delta) int_{x-delta}^{x} Q(y) dy, Q(y) = 0 for y <= 0
  point_mass   F(x) = pi min(x / delta, 1) + (1 - pi) smooth(x)   (atom pi spread uniformly over the first cell)
Check per event: F(tau + delta) - F(tau) = exp(C) (tau = the float32 batch gap that C sees); a cell is quarantined when
some event has |diff| > CELL_RTOL exp(C) + CELL_COND eps_64 (tau + delta) / delta (relative 1e-6 plus the float64
conditioning of a distribution-function difference over a cell much narrower than the gap). F_gl below is an
independent Gauss-Legendre version of F used by the tests.
h in {0, q50, q75, q90}: quantiles of the positive train recorded gaps rounded DOWN to a multiple of delta; y = 1[g > h]
from the float64 recorded gap (cell index rint(g / delta)). Metrics per h: Brier, log score (mean log p(y), p clipped
to [1e-12, 1 - 1e-12]; higher is better), AUC (rank based), decision cost for c in {1, 4, 9}: alarm iff p > 1/(1 + c),
cost = c x missed + false alarms, per event. Primary metric: mean Brier over the four h.
Statistics (descriptive): per unit the model chosen by F (or F_own) against the one chosen by C: seed-paired
differences (same seed index, C-chosen minus F-chosen) and a case-level block bootstrap (N_BOOT resamples of the test
sequences, fixed seed) of the seed-mean difference, percentile 95% interval; conditional on the selected models.

Remaining time (secondary, --tasks rt; run after sla because F_own needs the sla files). Path model: bigram Markov
chain over activities with an END state from the train split (shared by every treatment); N_ROLL paths per (test case,
prefix fraction) are drawn ONCE per job from a seed that depends only on the job (common paths for every model). Times
follow the readout: raw: t_i = t~_{i-1} + s, s ~ q; smooth: t_lat = t~ + U delta once at the rollout start, then
t_i = t_{i-1} + tau, tau ~ Q; point_mass: with probability pi t_i = t~_{i-1} + U delta, else the smooth continuation;
t~_i = delta floor(t_i / delta) is fed back with the sampled activity (the whole history is re-encoded each step).
Paths are capped at RT_CAP_FACTOR x the longest train case; the share of truncated paths is reported. Metrics: mean
absolute error of the rollout median of the completion time (time since case start of the last event), 80% interval
coverage and width; auxiliary: Brier of P(completion > H), H = train case-duration q75. RT runs and is reported only
for a (cell, checkpoint) whose SLA reproduction check passed ('verified').

Validity: test C of the checkpoint recomputed with the batches of coarse_tpp.train.evaluate must match result.json
(by_selection.<sel>.test.C) within C_TOL; for best_C.pt also dev C and dev floor 1e-6 against result['dev'];
otherwise the cell is quarantined (never a valid cache, excluded from the report; --retry-quarantined recomputes it).
Cache identity: every cached checkpoint result stores the checkpoint's SHA-256 and selected epoch; a cached file whose
checkpoints differ from the current result.json / best_<sel>.pt (or whose set of available checkpoints changed) is
stale and is recomputed, and its RT files are not reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import analysis as A
from . import data as D
from . import experiments as X
from . import lik
from . import predictive as PM
from . import readout as Ro
from .data import coarse_stats, load, score_mask
from .models import build

PRIMARY_DS = ("bpi_sepsis", "bpi_bpic12")
TRI = "triangular"
BASE_ARMS = tuple(X.PROCESS_ARMS)
LRS = tuple(X.LRS)
PLANNED_SEEDS = tuple(X.REAL_SEEDS)
OWN_ARM = "floor_1e-6"
FLOOR_KEY = "floor_1e-6"
EPS_FLOOR = 1e-6
H_KEYS = ("h0", "q50", "q75", "q90")
QUANTS = {"q50": 0.5, "q75": 0.75, "q90": 0.9}
COSTS = (1, 4, 9)
P_CLIP = 1e-12
C_TOL = 1e-5
CELL_RTOL, CELL_FLOOR, CELL_COND = 1e-6, 1e-6, 64.0
N_BOOT, BOOT_SEED = 2000, 20261021
FRACS = (0.25, 0.5, 0.75)
N_ROLL, RT_SEED, RT_CAP_FACTOR, RT_ROWS = 200, 20261022, 2, 4000
S_MAX = 1e9                          # sampled waiting times are capped here (time units; counted, reported)
VERSION = 1
SELS = ("C", "own")


# ------------------------------------------------------------------ distribution function of s
def F_B(law, pi, rule, delta, x):
    """F at x (N, M) for law parameters (N, K), pi (N,), under the readout `rule` (coarse_tpp.predictive.cdf)."""
    return PM.cdf(law, pi, rule, delta, x)


def F_B_gl(law, pi, rule, delta, x, n=64, panels=8):
    """Independent Gauss-Legendre version of F (tests): smooth = (1/delta) int_{max(x-delta,0)}^{x} Q(y) dy with a
    composite rule (`panels` x `n` nodes) in y."""
    if rule == "raw":
        return PM.q_cdf(law, x)
    u, w = lik.gauss_legendre(n, device=x.device)
    node = ((torch.arange(panels, dtype=torch.float64, device=x.device).unsqueeze(-1) + u) / panels).reshape(-1)
    wt = w.repeat(panels) / panels
    a = (x - delta).clamp(min=0.0)
    y = a.unsqueeze(-1) + (x - a).unsqueeze(-1) * node                          # (N, M, P)
    N, M, P = y.shape
    Q = PM.q_cdf(law, y.reshape(N, M * P)).reshape(N, M, P)
    sm = (Q * wt).sum(-1) * (x - a) / delta
    sm = torch.where(x > 0, sm, torch.zeros_like(sm))
    if rule == "smooth":
        return sm
    p = pi.unsqueeze(-1)
    return p * (x / delta).clamp(0.0, 1.0) + (1 - p) * sm


def _law(P, sl):
    return lik.LogNormMix(P["log_w"][sl], P["mu"][sl], P["log_sigma"][sl])


def sla_probs(P, rule, delta, kh, chunk=2048):
    """p (N, H) = 1 - F((kh + 1) delta) for the grid indices kh (H,)."""
    N = P["mu"].shape[0]
    x0 = torch.tensor([(k + 1) * delta for k in kh], dtype=torch.float64, device=P["mu"].device)
    out = []
    for i in range(0, N, chunk):
        sl = slice(i, min(i + chunk, N))
        x = x0.expand(sl.stop - sl.start, len(kh))
        out.append(1.0 - F_B(_law(P, sl), P["pi"][sl], rule, delta, x))
    return torch.cat(out, 0).cpu().numpy() if out else np.zeros((0, len(kh)))


def cell_tol(pC, tau, delta):
    """CELL_RTOL relative plus the float64 conditioning of a distribution-function difference at x = tau + delta over a
    cell of width delta (the closed-form smooth F loses ~eps x / delta absolutely)."""
    return CELL_RTOL * pC + CELL_COND * 2.0 ** -52 * (tau + delta) / delta


def cell_check(P, rule, delta, chunk=2048):
    """Per event: F(tau + delta) - F(tau) against exp(C) (tau = float32 batch gap). The cell fails when some event has
    |diff| > cell_tol. Also reported: the plain relative error with denominator max(p, CELL_FLOOR)."""
    N = P["mu"].shape[0]
    rel, ab, ratio = [], [], []
    for i in range(0, N, chunk):
        sl = slice(i, min(i + chunk, N))
        t = P["tau32"][sl]
        Fc = F_B(_law(P, sl), P["pi"][sl], rule, delta, torch.stack([t, t + delta], -1))
        pc = Fc[:, 1] - Fc[:, 0]
        pC = torch.exp(P["C"][sl])
        d = (pc - pC).abs()
        ab.append(d); rel.append(d / pC.clamp(min=CELL_FLOOR)); ratio.append(d / cell_tol(pC, t, delta))
    rel, ab, ratio = (torch.cat(x) if x else torch.zeros(0, dtype=torch.float64) for x in (rel, ab, ratio))
    mr = float(ratio.max()) if ratio.numel() else 0.0
    return dict(cell_max_rel=float(rel.max()) if rel.numel() else 0.0, cell_max_abs=float(ab.max()) if ab.numel() else 0.0,
                cell_max_tol_ratio=mr, cell_n_bad=int((ratio > 1).sum()), cell_ok=bool(mr <= 1.0))


# ------------------------------------------------------------------ data-side quantities
def h_grid(train_views, delta):
    """{key: (kh, h, raw quantile)}; kh = floor(q / delta + 1e-6) (q rounded down to the grid; the 1e-6 absorbs float
    noise of a q that is itself a grid multiple)."""
    tg = np.concatenate([x[1:] for x in train_views.coarse.iet])
    pos = tg[tg > 0]
    out = {"h0": (0, 0.0, 0.0)}
    for k, q in QUANTS.items():
        v = float(np.quantile(pos, q))
        kh = int(math.floor(v / delta + 1e-6))
        out[k] = (kh, kh * delta, v)
    return out


def outcomes(tau64, kh, delta):
    """y (N, H) = 1[g > h] with the cell index rint(g / delta); also the max |g / delta - rint| (grid residual)."""
    r = tau64 / delta
    kg = np.rint(r)
    return (kg[:, None] > np.asarray(kh)[None, :]).astype(np.float64), float(np.max(np.abs(r - kg))) if len(r) else 0.0


# ------------------------------------------------------------------ per-event parameters
@torch.no_grad()
def per_event(model, views, stats, cfg, device, bs=64):
    """Same batches / masks / readout call as coarse_tpp.train.evaluate, plus the sequence index of every scored target
    and its floor 1e-6 score. Flat tensors over scored targets."""
    model.eval()
    delta = float(cfg["delta"]); r = Ro.RULE[cfg["treatment"]]
    sp = views.coarse
    acc = {k: [] for k in ("log_w", "mu", "log_sigma", "pi", "tau32", "tau64", "C", "floor", "seq")}
    c_sum = e_sum = n_sum = 0.0
    for i0 in range(0, len(sp), bs):
        idx = np.arange(i0, min(i0 + bs, len(sp)))
        b = D.make_batch(sp, idx, device)
        L = b.iet.shape[1]
        msk = (b.tmask * score_mask(views, idx, L, device)).double()
        tau = b.iet[:, 1:].double()
        law, tl = model(b, stats)
        c = Ro.C(law, tl, tau, delta, r)
        e6 = lik.event_loglik(law, tau, "floor", eps=EPS_FLOOR)
        c = torch.where(msk > 0, c, torch.zeros_like(c)); e6 = torch.where(msk > 0, e6, torch.zeros_like(e6))
        c_sum += float((c * msk).sum()); e_sum += float((e6 * msk).sum()); n_sum += float(msk.sum())
        sel = msk > 0
        acc["log_w"].append(law.log_w[sel]); acc["mu"].append(law.mu[sel]); acc["log_sigma"].append(law.log_sigma[sel])
        acc["pi"].append(torch.sigmoid(tl.double())[sel] if tl is not None else torch.zeros_like(tau[sel]))
        acc["tau32"].append(tau[sel]); acc["C"].append(c[sel]); acc["floor"].append(e6[sel])
        t64 = torch.zeros_like(tau)
        sq = torch.zeros_like(tau, dtype=torch.long)
        for bi, i in enumerate(idx):
            g = torch.tensor(sp.iet[i][1:], dtype=torch.float64, device=device)
            t64[bi, :len(g)] = g
            sq[bi] = int(i)
        acc["tau64"].append(t64[sel]); acc["seq"].append(sq[sel])
    out = {k: torch.cat(v, 0) for k, v in acc.items()}
    out.update(C_mean=c_sum / n_sum, floor_mean=e_sum / n_sum, n=int(round(n_sum)), n_seq=len(sp))
    return out


# ------------------------------------------------------------------ metrics
def auc(p, y):
    """Rank-based AUC (average ranks for ties); None when y has one class."""
    from scipy.stats import rankdata
    y = np.asarray(y) > 0.5
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def event_terms(p, y):
    """Per-event terms: brier (N, H), logs (N, H) = log p(y), cost (N, C, H)."""
    pc = np.clip(p, P_CLIP, 1 - P_CLIP)
    brier = (p - y) ** 2
    logs = y * np.log(pc) + (1 - y) * np.log1p(-pc)
    cost = np.stack([c * (y * (p <= 1 / (1 + c))) + (1 - y) * (p > 1 / (1 + c)) for c in COSTS], 1)
    return brier, logs, cost


def seq_sums(seq, n_seq, arr):
    """Per-sequence sums of the event rows of arr (N, ...) -> (n_seq, ...)."""
    out = np.zeros((n_seq,) + arr.shape[1:])
    np.add.at(out, seq, arr)
    return out


def sla_metrics(p, y):
    brier, logs, cost = event_terms(p, y)
    m = dict(brier={}, logscore={}, auc={}, cost={f"c{c}": {} for c in COSTS}, base_rate={})
    for j, h in enumerate(H_KEYS):
        m["brier"][h] = float(brier[:, j].mean()); m["logscore"][h] = float(logs[:, j].mean())
        m["auc"][h] = auc(p[:, j], y[:, j]); m["base_rate"][h] = float(y[:, j].mean())
        for ci, c in enumerate(COSTS):
            m["cost"][f"c{c}"][h] = float(cost[:, ci, j].mean())
    m["brier_mean"] = float(np.mean(list(m["brier"].values())))
    return m, brier, logs, cost


# ------------------------------------------------------------------ cells and selection
def load_results(run, bases=(), datasets=None):
    """(cells, info): status-ok, complete-seed results of the process_logs run and of the triangular cells of the base
    runs, process logs only."""
    cells, info = [], {}
    for i, r in enumerate([Path(run)] + [Path(b) for b in bases]):
        if not (r / "grid.jsonl").exists():
            info[str(r)] = "absent"; continue
        c = A.load_cells(r)
        c = [x for x in c if x["config"]["dataset"].startswith("bpi_") and x["config"]["head"] == "mix"
             and (i == 0 or x["config"]["treatment"] == TRI)]
        if datasets:
            c = [x for x in c if x["config"]["dataset"] in datasets]
        info[str(r)] = f"{len(c)} cells used"
        cells += c
    return cells, info


def unit_of(res):
    c = res["config"]
    return (c["dataset"], c["encoder"]["type"])


def candidates(cells):
    """{unit: {(arm, lr_tag): {seed: result}}} over all runs merged."""
    out = {}
    for r in cells:
        out.setdefault(unit_of(r), {}).setdefault((r["arm"], r["config"]["lr_tag"]), {})[r["seed"]] = r
    return out


def expected_keys(with_tri: bool):
    arms = BASE_ARMS + ((TRI,) if with_tri else ())
    return [(a, l) for a in arms for l in LRS]


def unit_status(cands: dict, with_tri: bool):
    """A unit (variant) is complete iff every planned candidate has every planned seed. Returns (complete, [missing])."""
    missing = []
    for k in expected_keys(with_tri):
        have = set(cands.get(k, {}))
        miss = [s for s in PLANNED_SEEDS if s not in have]
        if miss:
            missing.append(f"{k[0]} {k[1]}: " + ("absent" if not have else f"seeds {miss} missing"))
    return not missing, missing


def variants_of(cands: dict):
    """'no_tri' always; 'with_tri' when the unit has any triangular candidate (active only when complete)."""
    return ("no_tri",) + (("with_tri",) if any(k[0] == TRI for k in cands) else ())


class Rule:
    """A selection rule: score(key, result) on the dev split and the checkpoint selection sel(key) of each candidate."""

    def __init__(self, name, score, sel):
        self.name, self.score, self.sel = name, score, sel


def make_rules(own_lookup):
    """F and C on best_C.pt (stored result['dev']); F_own: ONLY the floor_1e-6 candidates move to best_own.pt and are
    ranked by own_lookup(result) (their dev floor 1e-6 recomputed on that checkpoint); every other candidate keeps
    best_C.pt and its stored dev floor 1e-6."""
    return {"F": Rule("F", lambda k, r: r["dev"][FLOOR_KEY], lambda k: "C"),
            "C": Rule("C", lambda k, r: r["dev"]["C"], lambda k: "C"),
            "F_own": Rule("F_own", lambda k, r: own_lookup(r) if k[0] == OWN_ARM else r["dev"][FLOOR_KEY],
                          lambda k: "own" if k[0] == OWN_ARM else "C")}


def choose(cands: dict, rule: Rule, with_tri: bool):
    """Highest mean over PLANNED_SEEDS of rule.score among the planned candidates (complete units only). A None or
    non-finite score makes the candidate unrankable and then no choice is made. Ties: sorted key order.
    Returns (key or None, sel or None, {key: mean}, [unrankable keys])."""
    means, bad = {}, []
    for key in expected_keys(with_tri):
        vals = [rule.score(key, cands[key][s]) for s in PLANNED_SEEDS]
        if any(v is None or not math.isfinite(v) for v in vals):
            bad.append(key); continue
        means[key] = float(np.mean(vals))
    if not means or bad:
        return None, None, means, bad
    best = max(sorted(means), key=lambda k: means[k])
    return best, rule.sel(best), means, bad


def own_lookup_fn(out_dir: Path):
    """dev floor 1e-6 of best_own.pt from the SLA file; None (not rankable) unless that checkpoint is verified for the
    current result (current version and identity, present, not quarantined)."""
    def f(r):
        if verification(out_dir, r["cell_id"], "own", r if "config" in r else None) != "verified":
            return None
        return sla_record(out_dir, r["cell_id"])["by_sel"]["own"].get("dev_floor")
    return f


def all_choices(cands_by_unit, out_dir: Path, own_lookup=None):
    """{(unit, variant): dict(complete, missing, rules={rule: (key, sel, means, bad)} or None when pending)}."""
    rules = make_rules(own_lookup or own_lookup_fn(out_dir))
    out = {}
    for u, cands in sorted(cands_by_unit.items()):
        for v in variants_of(cands):
            ok, missing = unit_status(cands, v == "with_tri")
            out[(u, v)] = dict(complete=ok, missing=missing,
                               rules={n: choose(cands, r, v == "with_tri") for n, r in rules.items()} if ok else None)
    return out


def same_model(cands, a, b):
    """Two (key, sel) choices are the same model iff same candidate and, per seed, the same selected epoch."""
    if a[0] is None or b[0] is None or a[0] != b[0]:
        return False
    if a[1] == b[1]:
        return True
    return all(cands[a[0]][s]["by_selection"][a[1]]["selected_epoch"] == cands[a[0]][s]["by_selection"][b[1]]["selected_epoch"]
               for s in PLANNED_SEEDS)


# ------------------------------------------------------------------ cache / verification state
def ckpt_sha(path) -> str:
    """SHA-256 of a checkpoint file: the identity a cached result is bound to."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sla_record(out_dir: Path, cell_id: str):
    p = Path(out_dir) / "cells" / f"{cell_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def identity_ok(j: dict, res: dict) -> bool:
    """The cached checkpoint results of `j` belong to the current result `res`: for every selection, the checkpoint
    file either is absent now and was recorded missing, or exists and matches the recorded selected epoch and SHA-256."""
    out_dir = Path(res["config"]["out_dir"])
    for sel in SELS:
        ck = out_dir / f"best_{sel}.pt"
        r = (j.get("by_sel") or {}).get(sel)
        if r is None:
            return False
        if not ck.exists() or sel not in res.get("by_selection", {}):
            if not r.get("missing"):
                return False
            continue
        if r.get("missing") or r.get("selected_epoch") != res["by_selection"][sel].get("selected_epoch") \
                or r.get("ckpt_sha256") != ckpt_sha(ck):
            return False
    return True


def sla_cache_state(out_dir: Path, cell_id: str, res: dict | None = None):
    """'absent' | 'stale' (other version, or with `res`: the cached checkpoints are not the current ones) |
    'quarantined' (some checkpoint failed the reproduction check: not a valid cache) | 'valid'."""
    j = sla_record(out_dir, cell_id)
    if j is None:
        return "absent"
    if j.get("version") != VERSION or (res is not None and not identity_ok(j, res)):
        return "stale"
    return "quarantined" if j.get("quarantined_sels") else "valid"


def verification(out_dir: Path, cell_id: str, sel: str, res: dict | None = None):
    """'verified' | 'quarantined' | 'unverified' (no current SLA record for that checkpoint, or a record of other
    checkpoints when `res` is given) of one (cell, checkpoint selection)."""
    j = sla_record(out_dir, cell_id)
    if j is None or j.get("version") != VERSION or (res is not None and not identity_ok(j, res)):
        return "unverified"
    r = j.get("by_sel", {}).get(sel)
    if not r or r.get("missing"):
        return "unverified"
    return "quarantined" if r.get("quarantined") else "verified"


def rt_collect(out_dir: Path, seeds_res: dict, sel: str):
    """RT records of one chosen model: (rows over seeds that are verified for the current checkpoint AND have a current
    RT file, [pending items])."""
    rows, pending = [], []
    for s in PLANNED_SEEDS:
        r = seeds_res.get(s)
        if r is None:
            pending.append(f"s{s}: no cell"); continue
        st = verification(out_dir, r["cell_id"], sel, r if "config" in r else None)
        p = Path(out_dir) / "rt" / f"{r['cell_id']}__{sel}.json"
        if st != "verified":
            pending.append(f"s{s}: {st}"); continue
        if not p.exists():
            pending.append(f"s{s}: no RT file"); continue
        j = json.loads(p.read_text())
        if not rt_identity_ok(j, r, sel):
            pending.append(f"s{s}: stale RT file"); continue
        rows.append(j)
    return rows, pending


def rt_identity_ok(row, res, sel):
    """An RT result must describe the currently selected checkpoint."""
    if (not res.get("config") or row.get("version") != VERSION
            or row.get("cell_id") != res.get("cell_id") or row.get("sel") != sel):
        return False
    ck = Path(res["config"]["out_dir"]) / f"best_{sel}.pt"
    selected = res.get("by_selection", {}).get(sel)
    return bool(selected is not None and ck.exists() and row.get("ckpt_sha256") == ckpt_sha(ck)
                and row.get("selected_epoch") == selected.get("selected_epoch"))


# ------------------------------------------------------------------ scoring
def rebuild(res, device):
    cfg = res["config"]
    ds, delta = cfg["dataset"], float(cfg["delta"])
    tr = load(ds, "train")
    stats = coarse_stats(tr, delta)
    model = build(cfg, stats, tr.coarse.n_types).to(device)
    if cfg["head"] != "mix":
        raise ValueError("the alarms need the mixture head")
    return cfg, tr, stats, model


def score_cell(res, device, out_dir: Path):
    t0 = time.time()
    cfg, tr, stats, model = rebuild(res, device)
    ds, delta, rule = cfg["dataset"], float(cfg["delta"]), Ro.RULE[cfg["treatment"]]
    te, dv = load(ds, "test"), load(ds, "dev")
    hg = h_grid(tr, delta)
    kh = [hg[k][0] for k in H_KEYS]
    rec = dict(version=VERSION, cell_id=res["cell_id"], experiment=res.get("experiment"), dataset=ds,
               encoder=cfg["encoder"]["type"], arm=res["arm"], treatment=cfg["treatment"], rule=rule, lr_tag=cfg["lr_tag"],
               seed=res["seed"], delta=delta, h_grid={k: dict(kh=hg[k][0], h=hg[k][1], q_raw=hg[k][2]) for k in H_KEYS},
               by_sel={}, device=str(device))
    quarantine = []
    for sel in SELS:
        ck = Path(cfg["out_dir"]) / f"best_{sel}.pt"
        if not ck.exists() or sel not in res["by_selection"]:
            rec["by_sel"][sel] = dict(missing=True); continue
        model.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
        P = per_event(model, te, stats, cfg, device)
        Pd = per_event(model, dv, stats, cfg, device)
        stored = float(res["by_selection"][sel]["test"]["C"])
        r = dict(missing=False, n_events=P["n"], C_recomputed=P["C_mean"], C_stored=stored, C_absdiff=abs(P["C_mean"] - stored),
                 dev_C=Pd["C_mean"], dev_floor=Pd["floor_mean"], test_floor=P["floor_mean"],
                 selected_epoch=res["by_selection"][sel].get("selected_epoch"), ckpt_sha256=ckpt_sha(ck))
        bad = r["C_absdiff"] > C_TOL
        if sel == "C" and res.get("dev"):
            r["dev_C_absdiff"] = abs(Pd["C_mean"] - float(res["dev"]["C"]))
            r["dev_floor_absdiff"] = abs(Pd["floor_mean"] - float(res["dev"][FLOOR_KEY]))
            bad = bad or r["dev_C_absdiff"] > C_TOL or r["dev_floor_absdiff"] > C_TOL
        r.update(cell_check(P, rule, delta))
        p = sla_probs(P, rule, delta, kh)
        y, resid = outcomes(P["tau64"].cpu().numpy(), kh, delta)
        r.update(p_min_raw=float(p.min()), p_max_raw=float(p.max()), n_p_clipped=int(((p < 0) | (p > 1)).sum()))
        p = np.clip(p, 0.0, 1.0)            # float32 mixture weights: total mass 1 +- ~1e-7, so 1 - F can be -1e-7
        m, brier, logs, cost = sla_metrics(p, y)
        r.update(m, grid_residual=resid)
        r["quarantined"] = bool(bad or not r["cell_ok"] or not np.all(np.isfinite(p)) or r["p_min_raw"] < -1e-6 or r["p_max_raw"] > 1 + 1e-6)
        if r["quarantined"]:
            quarantine.append(sel)
        seq = P["seq"].cpu().numpy()
        n_seq = P["n_seq"]
        np.savez_compressed(out_dir / "cells" / f"{res['cell_id']}__{sel}.npz", n=seq_sums(seq, n_seq, np.ones(len(seq))),
                            brier=seq_sums(seq, n_seq, brier), logs=seq_sums(seq, n_seq, logs), cost=seq_sums(seq, n_seq, cost))
        rec["by_sel"][sel] = r
    rec["quarantined_sels"] = quarantine
    rec["wall_sec"] = round(time.time() - t0, 1)
    return rec


# ------------------------------------------------------------------ remaining time
def markov_chain(train_split, n_types):
    """Row-stochastic (n_types, n_types + 1) bigram matrix with END = column n_types, from the train cases; a state
    without outgoing train transitions gets the pooled transition distribution."""
    cnt = np.zeros((n_types, n_types + 1))
    for mk in train_split.marks:
        for a, b in zip(mk[:-1], mk[1:]):
            cnt[a, b] += 1
        cnt[mk[-1], n_types] += 1
    pooled = cnt.sum(0) / cnt.sum()
    rows = cnt.sum(1, keepdims=True)
    return np.where(rows > 0, cnt / np.maximum(rows, 1), pooled[None, :])


def sample_paths(T, last, R, max_steps, rng):
    """R activity paths from state `last` until END (exclusive) or max_steps. Returns (list of arrays, truncated flags)."""
    n_types = T.shape[0]
    cum = np.cumsum(T, 1)
    cur = np.full(R, last, dtype=np.int64)
    alive = np.ones(R, bool)
    trunc = np.zeros(R, bool)
    steps = []
    for _ in range(max_steps):
        u = rng.random(R)
        nxt = (u[:, None] > cum[cur]).sum(1)
        nxt = np.minimum(nxt, n_types)
        end = alive & (nxt == n_types)
        alive = alive & ~end
        if not alive.any():
            break
        steps.append(np.where(alive, nxt, -1))
        cur = np.where(alive, nxt, cur)
    else:
        trunc = alive.copy()
    if steps:
        S = np.stack(steps, 1)
        paths = [S[i][S[i] >= 0] for i in range(R)]
    else:
        paths = [np.zeros(0, np.int64) for _ in range(R)]
    return paths, trunc


def sample_mix(lw, mu, ls, rng):
    """One draw per row from the log-normal mixture (numpy): component by log weight (Gumbel max), then exp(mu + s z).
    Returns (draws, n_capped)."""
    g = -np.log(-np.log(rng.random(lw.shape)))
    k = np.argmax(lw + g, 1)
    r = np.arange(len(k))
    lx = mu[r, k] + np.exp(ls[r, k]) * rng.standard_normal(len(k))
    cap = lx > math.log(S_MAX)
    return np.exp(np.minimum(lx, math.log(S_MAX))), int(cap.sum())


@torch.no_grad()
def rt_cell(res, sel, device, max_cases=None):
    cfg, tr, stats, model = rebuild(res, device)
    ds, delta, rule = cfg["dataset"], float(cfg["delta"]), Ro.RULE[cfg["treatment"]]
    model.load_state_dict(torch.load(Path(cfg["out_dir"]) / f"best_{sel}.pt", map_location=device, weights_only=True))
    model.eval()
    te = load(ds, "test").coarse
    n_types = te.n_types
    T = markov_chain(tr.coarse, n_types)
    cap_len = RT_CAP_FACTOR * max(len(x) for x in tr.coarse.iet)
    H = float(np.quantile([x[-1] for x in tr.coarse.times], 0.75))
    jobs = []
    n_cases = len(te) if max_cases is None else min(max_cases, len(te))
    for i in range(n_cases):
        n = len(te.iet[i])
        for fi, f in enumerate(FRACS):
            m = min(max(1, int(math.floor(f * n))), n - 1)
            jobs.append((i, fi, m))
    trng = np.random.default_rng([RT_SEED, 1])
    J = max(1, RT_ROWS // N_ROLL)
    out = dict(case=[], frac=[], abs_err=[], covered=[], width=[], brier_H=[], n_trunc=[], n_paths=[])
    n_capped = 0
    for j0 in range(0, len(jobs), J):
        chunk = jobs[j0:j0 + J]
        rows_hist, rows_path, rows_t, n_tr = [], [], [], []
        for (i, fi, m) in chunk:
            prng = np.random.default_rng([RT_SEED, i, fi])          # common paths for every model
            paths, trunc = sample_paths(T, int(te.marks[i][m - 1]), N_ROLL, max(0, cap_len - m), prng)
            n_tr.append(int(trunc.sum()))
            for pth in paths:
                rows_hist.append((te.iet[i][:m], te.marks[i][:m])); rows_path.append(pth); rows_t.append(float(te.times[i][m - 1]))
        B = len(rows_hist)
        lens = np.array([len(h[0]) for h in rows_hist]); plen = np.array([len(p) for p in rows_path])
        Wm = int((lens + plen).max()) + 1
        iet = np.zeros((B, Wm)); mk = np.zeros((B, Wm), np.int64)
        for b, (g, a) in enumerate(rows_hist):
            iet[b, :len(g)] = g; mk[b, :len(a)] = a
        Tt = np.array(rows_t)                                          # current recorded time t~
        u = trng.random(B) * delta if rule in ("smooth", "point_mass") else np.zeros(B)   # latent offset t - t~
        cur = lens.copy()
        step = 0
        while True:
            act = np.nonzero(plen > step)[0]
            if len(act) == 0:
                break
            W = int(cur[act].max()) + 1
            bi = torch.as_tensor(iet[act, :W], dtype=torch.float32, device=device)
            bm = torch.as_tensor(mk[act, :W], device=device)
            ar = np.arange(W)[None, :]
            mask = torch.as_tensor((ar < cur[act, None]).astype(np.float32), device=device)
            batch = D.Batch(bi * mask, torch.cumsum(bi * mask, 1), bm, mask, mask[:, 1:] * mask[:, :-1],
                            torch.as_tensor(cur[act], device=device), act)
            law, tl = model(batch, stats)
            pos = torch.as_tensor(cur[act] - 1, device=device)
            ga = lambda x: x[torch.arange(len(act), device=device), pos].cpu().numpy()
            lw, mu, ls = ga(law.log_w), ga(law.mu), ga(law.log_sigma)
            s, nc = sample_mix(lw, mu, ls, trng)
            n_capped += nc
            if rule == "raw":
                g = delta * np.floor(s / delta)
            else:
                v = u[act] + s
                if rule == "point_mass":
                    pi = torch.sigmoid(tl.double()[torch.arange(len(act), device=device), pos]).cpu().numpy()
                    atom = trng.random(len(act)) < pi
                    v = np.where(atom, trng.random(len(act)) * delta, v)
                g = delta * np.floor(v / delta)
                u[act] = v - g
            iet[act, cur[act]] = g
            mk[act, cur[act]] = np.array([rows_path[b][step] for b in act], dtype=np.int64)
            Tt[act] += g
            cur[act] += 1
            step += 1
        for c, (i, fi, m) in enumerate(chunk):
            comp = Tt[c * N_ROLL:(c + 1) * N_ROLL]
            truth = float(te.times[i][-1])
            med, lo, hi = np.quantile(comp, [0.5, 0.1, 0.9])
            out["case"].append(i); out["frac"].append(fi)
            out["abs_err"].append(abs(med - truth)); out["covered"].append(float(lo <= truth <= hi)); out["width"].append(hi - lo)
            out["brier_H"].append((float(np.mean(comp > H)) - float(truth > H)) ** 2)
            out["n_trunc"].append(n_tr[c]); out["n_paths"].append(N_ROLL)
    arr = {k: np.asarray(v) for k, v in out.items()}
    summ = dict(cell_id=res["cell_id"], sel=sel, rule=rule, n_jobs=len(jobs), n_cases=n_cases, H=H, cap_len=cap_len,
                mae_median=float(arr["abs_err"].mean()), coverage80=float(arr["covered"].mean()), width80=float(arr["width"].mean()),
                brier_H=float(arr["brier_H"].mean()), trunc_share=float(arr["n_trunc"].sum() / arr["n_paths"].sum()),
                n_capped_draws=n_capped,
                by_frac={f"{f}": dict(mae_median=float(arr["abs_err"][arr["frac"] == fi].mean()),
                                      coverage80=float(arr["covered"][arr["frac"] == fi].mean()),
                                      width80=float(arr["width"][arr["frac"] == fi].mean())) for fi, f in enumerate(FRACS)})
    return summ, arr, n_cases


# ------------------------------------------------------------------ commands
def _write_json(p: Path, obj):
    tmp = p.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1)); os.replace(tmp, p)


def selected_cell_sels(cands_by_unit, out_dir, rules=("F", "C", "F_own")):
    """{cell_id: set(sel)} of the chosen model of every complete unit / variant / rule (each with its own checkpoint)."""
    need = {}
    for (u, v), row in all_choices(cands_by_unit, out_dir).items():
        if row["rules"] is None:
            continue
        for rule in rules:
            key, sel, _, _ = row["rules"][rule]
            if key is None:
                continue
            for s in PLANNED_SEEDS:
                need.setdefault(cands_by_unit[u][key][s]["cell_id"], set()).add(sel)
    return need


def _sla_pass(cells, device, out_dir, a):
    for i, res in enumerate(cells):
        st = sla_cache_state(out_dir, res["cell_id"], res)
        if st == "valid" and not a.force:
            continue
        if st == "quarantined" and not (a.force or a.retry_quarantined):
            print(f"[sla {i + 1}/{len(cells)}] {res['cell_id']}: quarantined cache, not a valid result; not retried "
                  "(use --retry-quarantined or --force)", flush=True)
            continue
        rec = score_cell(res, device, out_dir)
        _write_json(out_dir / "cells" / f"{res['cell_id']}.json", rec)
        c = rec["by_sel"].get("C", {})
        print(f"[sla {i + 1}/{len(cells)}] {res['cell_id']} Cdiff {c.get('C_absdiff', float('nan')):.2e} "
              f"cell_rel {c.get('cell_max_rel', float('nan')):.2e} tol_ratio {c.get('cell_max_tol_ratio', float('nan')):.2e} "
              f"brier_mean {c.get('brier_mean', float('nan')):.5f} quarantined {rec['quarantined_sels']} ({rec['wall_sec']}s)", flush=True)


def _filter(cells, a, keep=None):
    if keep is not None:
        cells = [r for r in cells if r["cell_id"] in keep]
    if a.only:
        cells = [r for r in cells if r["cell_id"] in set(a.only)]
    if a.limit:
        cells = cells[:a.limit]
    return cells


def cmd_score(a):
    torch.set_num_threads(int(a.threads))
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    out_dir = out_dir_of(a); (out_dir / "cells").mkdir(parents=True, exist_ok=True); (out_dir / "rt").mkdir(exist_ok=True)
    tasks = set(a.tasks.split(","))
    cells, info = load_results(a.run, a.base or (), a.datasets)
    print(f"[alarms] device {device}; " + "; ".join(f"{k}: {v}" for k, v in info.items()), flush=True)
    cands = candidates(cells)
    for (u, v), row in all_choices(cands, out_dir).items():
        if not row["complete"]:
            print(f"[alarms] {u[0]} {u[1]} {v}: pending (incomplete): {'; '.join(row['missing'][:6])}", flush=True)
    if "sla" in tasks:
        if a.selected:
            # pass 1: the F / C choices and every floor_1e-6 cell (F_own ranks them by their own checkpoint); pass 2: the F_own choice
            keep = set(selected_cell_sels(cands, out_dir, rules=("F", "C"))) | {r["cell_id"] for r in cells if r["arm"] == OWN_ARM}
            _sla_pass(_filter(cells, a, keep), device, out_dir, a)
            _sla_pass(_filter(cells, a, set(selected_cell_sels(cands, out_dir))), device, out_dir, a)
        else:
            _sla_pass(_filter(cells, a), device, out_dir, a)
    if "rt" in tasks:
        need = selected_cell_sels(cands, out_dir)
        base = _filter(cells, a)
        todo = [(r, s) for r in base for s in (SELS if a.rt_all else sorted(need.get(r["cell_id"], ())))]
        for i, (res, sel) in enumerate(todo):
            st = verification(out_dir, res["cell_id"], sel, res)
            if st != "verified":
                print(f"[rt {i + 1}/{len(todo)}] {res['cell_id']} {sel}: skipped, reproduction {st} (run sla first)", flush=True)
                continue
            p = out_dir / "rt" / f"{res['cell_id']}__{sel}.json"
            if p.exists() and not a.force:
                try:
                    old = json.loads(p.read_text())
                    if rt_identity_ok(old, res, sel) and (old.get("complete") or a.rt_max_cases is not None):
                        continue
                except Exception:
                    pass
            t0 = time.time()
            summ, arr, n_cases = rt_cell(res, sel, device, a.rt_max_cases)
            v = sla_record(out_dir, res["cell_id"])["by_sel"][sel]
            n_seq = len(load(res["config"]["dataset"], "test").coarse)
            summ.update(wall_sec=round(time.time() - t0, 1), version=VERSION, complete=bool(a.rt_max_cases is None or n_cases >= n_seq),
                        ckpt_sha256=v["ckpt_sha256"], selected_epoch=v["selected_epoch"],
                        verified_snapshot=dict(C_absdiff=v["C_absdiff"], cell_max_tol_ratio=v["cell_max_tol_ratio"]))
            np.savez_compressed(out_dir / "rt" / f"{res['cell_id']}__{sel}.npz",
                                n=seq_sums(arr["case"], n_seq, np.ones(len(arr["case"]))),
                                abs_err=seq_sums(arr["case"], n_seq, arr["abs_err"]), covered=seq_sums(arr["case"], n_seq, arr["covered"]),
                                width=seq_sums(arr["case"], n_seq, arr["width"]), brier_H=seq_sums(arr["case"], n_seq, arr["brier_H"]))
            _write_json(p, summ)
            print(f"[rt {i + 1}/{len(todo)}] {res['cell_id']} {sel}: MAE_med {summ['mae_median']:.4g} cov80 {summ['coverage80']:.3f} "
                  f"width80 {summ['width80']:.4g} trunc {summ['trunc_share']:.4f} capped {summ['n_capped_draws']} ({summ['wall_sec']}s)", flush=True)


# ------------------------------------------------------------------ report
def boot_diff(A_, B_, n: np.ndarray, seed: int = BOOT_SEED, n_boot: int = N_BOOT):
    """Seed-mean difference (B - A) of per-event means, A / B lists over paired seeds of per-sequence sums (n_seq, ...),
    n (n_seq,) per-sequence event counts (shared). Point value and percentile 95% interval of a case-level bootstrap
    (sequences resampled with replacement, the same draws for every seed and both models; fixed RNG seed)."""
    Dm = np.mean([b - a for a, b in zip(A_, B_)], 0)                        # (n_seq, ...)
    flat = Dm.reshape(len(n), -1)
    point = flat.sum(0) / n.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(n_boot, len(n)))
    Wt = np.zeros((n_boot, len(n)))
    np.add.at(Wt, (np.repeat(np.arange(n_boot), len(n)), idx.reshape(-1)), 1.0)
    bs = (Wt @ flat) / (Wt @ n)[:, None]
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    sh = Dm.shape[1:]
    return point.reshape(sh), lo.reshape(sh), hi.reshape(sh)


def data_header(datasets, deltas):
    """Per data set: delta, floor-delta relation, h grid, test base rates, float32 grid offset of the batch gaps."""
    rows = {}
    for ds in datasets:
        tr, te = load(ds, "train"), load(ds, "test")
        delta = deltas[ds]
        hg = h_grid(tr, delta)
        tg = np.concatenate([x[1:] for x in te.coarse.iet])
        kh = [hg[k][0] for k in H_KEYS]
        y, resid = outcomes(tg, kh, delta)
        off = np.abs(tg.astype(np.float32).astype(np.float64) - tg) / delta
        rows[ds] = dict(delta=delta, delta_sec=delta * 3600.0, floor_vs_delta=("floor 1e-6 > delta" if EPS_FLOOR > delta else "floor 1e-6 < delta"),
                        tie_share=float(np.mean(tg <= 0)), n_test_events=int(len(tg)), n_test_cases=len(te.coarse),
                        h={k: hg[k][1] for k in H_KEYS}, q_raw={k: hg[k][2] for k in H_KEYS},
                        base_rate={k: float(y[:, j].mean()) for j, k in enumerate(H_KEYS)}, grid_residual=resid,
                        f32_offset_max_cells=float(off.max()), f32_offset_share_ge_half=float(np.mean(off >= 0.5)))
    return rows


def out_dir_of(a):
    return Path(a.out) if a.out else Path(a.run) / "alarms"


def cmd_report(a):
    out_dir = out_dir_of(a)
    cells, info = load_results(a.run, a.base or (), a.datasets)
    cands = candidates(cells)
    deltas = {r["config"]["dataset"]: float(r["config"]["delta"]) for r in cells}
    scored, quarantined, stale = {}, [], []
    for r in cells:
        st = sla_cache_state(out_dir, r["cell_id"], r)
        if st == "stale":
            stale.append(r["cell_id"]); continue
        if st in ("valid", "quarantined"):
            j = sla_record(out_dir, r["cell_id"])
            scored[r["cell_id"]] = j
            quarantined += [f"{r['cell_id']}:{s}" for s in j.get("quarantined_sels", [])]
    datasets = sorted({u[0] for u in cands}, key=lambda d: (d not in PRIMARY_DS, d))
    hdr = data_header(datasets, deltas)
    L = ["# SLA alarms and remaining time on the process logs (exploratory, no training)", "",
         "Supplementary S4, Table S12. BPIC12 and Sepsis (more than 1% ties) are the main logs, the others auxiliary; the "
         "analysis is descriptive.", "",
         "Inputs: " + "; ".join(f"{k}: {v}" for k, v in info.items()) + ".",
         f"SLA-scored cells: {len(scored)} of {len(cells)}; stale (cached for other checkpoints; re-run score): {len(stale)}"
         + (f" — {', '.join(stale[:30])}" if stale else "") + f"; quarantined (cell:checkpoint): {len(quarantined)}"
         + (f" — {', '.join(quarantined[:30])}" if quarantined else "") + ".", ""]
    cs = [s for j in scored.values() for s in j["by_sel"].values() if not s.get("missing")]
    if cs:
        L += [f"Reproduction: max |test C recomputed - stored| {max(s['C_absdiff'] for s in cs):.2e}; "
              f"max |dev C|, |dev floor 1e-6| diff (best_C) {max([s.get('dev_C_absdiff', 0) for s in cs]):.2e}, "
              f"{max([s.get('dev_floor_absdiff', 0) for s in cs]):.2e}; cell-probability check F against exp(C): max relative mismatch "
              f"(denominator max(p, {CELL_FLOOR})) {max(s['cell_max_rel'] for s in cs):.2e}, max |diff| / tolerance "
              f"{max(s['cell_max_tol_ratio'] for s in cs):.2e} (gate <= 1).", ""]
    L += ["## 0. Data sets: resolution, floor, SLA grid", "",
          "| data set | delta (h) | delta (s) | floor vs delta | test events (cases) | test ties | h q50 / q75 / q90 (h) | "
          "base rate P(g > h) h0 / q50 / q75 / q90 | float32 batch-gap offset max (cells), share >= 0.5 |",
          "|---|---:|---:|---|---|---:|---|---|---|"]
    for ds, h in hdr.items():
        L.append(f"| {ds}{' (main)' if ds in PRIMARY_DS else ''} | {h['delta']:.3g} | {h['delta_sec']:.3g} | {h['floor_vs_delta']} | "
                 f"{h['n_test_events']} ({h['n_test_cases']}) | {h['tie_share']:.3f} | "
                 + " / ".join(f"{h['h'][k]:.4g}" for k in H_KEYS[1:]) + " | " + " / ".join(f"{h['base_rate'][k]:.3f}" for k in H_KEYS)
                 + f" | {h['f32_offset_max_cells']:.3g}, {h['f32_offset_share_ge_half']:.3f} |")
    L += ["", "The float32 offset column: coarse_tpp.train.evaluate (and so the stored C and the F / C rules) sees the gaps as "
          "float32 batch values; where delta is small relative to the gaps (BPIC12, 1 ms in hours) the float32 gap sits off the "
          "recorded cell. The SLA outcome uses the float64 recorded gap; the cell-probability check uses the float32 gap, as C does.", ""]
    choices = all_choices(cands, out_dir)
    L += ["## 1. Selections (dev split)", "",
          f"Candidates per unit: {len(BASE_ARMS)} treatments (+ triangular in the with_tri variant) x {len(LRS)} learning rates; a "
          f"unit / variant is complete only when every candidate has all planned seeds {list(PLANNED_SEEDS)}. F and C use best_C.pt "
          "and the stored result['dev']; F_own moves only the floor_1e-6 candidates to best_own.pt (ranked by their dev floor 1e-6 "
          "recomputed here), all other candidates stay at best_C.pt. The selected checkpoint is the one evaluated.", "",
          "| data set | encoder | variant | status | F (dev floor 1e-6) | C (dev C) | F_own | F = C | F_own = C |", "|---|---|---|---|---|---|---|---|---|"]
    n_same = {}
    for (u, v), row in choices.items():
        if row["rules"] is None:
            L.append(f"| {u[0]} | {u[1]} | {v} | pending: {'; '.join(row['missing'][:8])}{' ...' if len(row['missing']) > 8 else ''} | | | | | |")
            continue
        R_ = row["rules"]
        def lab(rule):
            key, sel, means, bad = R_[rule]
            return ("— (unrankable: " + ", ".join(f"{b[0]} {b[1]}" for b in bad) + ")") if key is None else \
                f"{key[0]} {key[1]} @best_{sel} ({means[key]:+.4f})"
        sF = same_model(cands[u], R_["F"][:2], R_["C"][:2]); sO = same_model(cands[u], R_["F_own"][:2], R_["C"][:2])
        n_same.setdefault(v, [0, 0, 0, 0]); n_same[v][0] += sF; n_same[v][1] += sO; n_same[v][2] += 1
        n_same[v][3] += R_["F_own"][0] is not None
        L.append(f"| {u[0]} | {u[1]} | {v} | complete | {lab('F')} | {lab('C')} | {lab('F_own')} | "
                 f"{'yes' if sF else 'no'} | {'yes' if sO else 'no' if R_['F_own'][0] else '—'} |")
    L += [""] + [f"- {v}: F and C pick the same model in {a_}/{n_} complete units; F_own and C in {b_}/{m_} units with an F_own choice."
                 for v, (a_, b_, n_, m_) in n_same.items()] + [""]
    L += ["## 2. SLA alarms: model chosen by F (or F_own) against the one chosen by C (test; seed means; diff = C-chosen - F-chosen)", "",
          "Primary metric: mean Brier over h in {0, q50, q75, q90} (lower is better). 95% interval: case-level block bootstrap "
          f"({N_BOOT} resamples, seed {BOOT_SEED}) of the seed-mean difference; seed-paired differences listed. Descriptive only.", "",
          "| data set | encoder | variant | comparison | F-type model | C model | Brier F | Brier C | diff [95% boot] | per-seed diffs | "
          "log score F / C | AUC F / C (mean over h) | cost c=1/4/9 F | cost c=1/4/9 C |",
          "|---|---|---|---|---|---|---:|---:|---|---|---|---|---|---|"]
    summary = []
    for (u, v), row0 in choices.items():
        if row0["rules"] is None:
            continue
        row = row0["rules"]
        for comp in ("F", "F_own"):
            kF, selF = row[comp][:2]; kC, selC = row["C"][:2]
            if kF is None or kC is None:
                continue
            seeds = list(PLANNED_SEEDS)
            def load_npz(key, sel):
                zs, js = [], []
                for s in seeds:
                    r = cands[u][key][s]
                    j = scored.get(r["cell_id"])
                    z = out_dir / "cells" / f"{r['cell_id']}__{sel}.npz"
                    if j is None or j["by_sel"].get(sel, {}).get("missing", True) or j["by_sel"][sel]["quarantined"] or not z.exists():
                        return None, None
                    zs.append(dict(np.load(z))); js.append(j["by_sel"][sel])
                return zs, js
            zF, jF = load_npz(kF, selF); zC, jC = load_npz(kC, selC)
            name = lambda k, s: f"{k[0]} {k[1]} @best_{s}"
            if zF is None or zC is None:
                L.append(f"| {u[0]} | {u[1]} | {v} | {comp} vs C | {name(kF, selF)} | {name(kC, selC)} | pending / quarantined | | | | | | | |")
                continue
            n = zF[0]["n"]
            assert all(np.array_equal(z["n"], n) for z in zF + zC), "per-sequence event counts differ"
            brF = [z["brier"].sum(0).mean() / n.sum() for z in zF]; brC = [z["brier"].sum(0).mean() / n.sum() for z in zC]
            same = same_model(cands[u], row[comp][:2], row["C"][:2])
            pt, lo, hi = boot_diff([z["brier"].mean(1) for z in zF], [z["brier"].mean(1) for z in zC], n)
            pd_ = [c - f for f, c in zip(brF, brC)]
            m = lambda js, f: float(np.mean([f(j) for j in js]))
            aucm = lambda j: float(np.mean([x for x in j["auc"].values() if x is not None])) if any(x is not None for x in j["auc"].values()) else float("nan")
            costs = lambda js: " / ".join(f"{m(js, lambda j, c=c: np.mean(list(j['cost'][f'c{c}'].values()))):.4f}" for c in COSTS)
            L.append(f"| {u[0]} | {u[1]} | {v} | {comp} vs C | {name(kF, selF)} | {name(kC, selC)} | {np.mean(brF):.5f} | {np.mean(brC):.5f} | "
                     + ("same model |" if same else f"{float(pt):+.5f} [{float(lo):+.5f}, {float(hi):+.5f}] |")
                     + f" {', '.join(f'{x:+.4f}' for x in pd_)} | {m(jF, lambda j: np.mean(list(j['logscore'].values()))):.4f} / "
                     f"{m(jC, lambda j: np.mean(list(j['logscore'].values()))):.4f} | {m(jF, aucm):.4f} / {m(jC, aucm):.4f} | {costs(jF)} | {costs(jC)} |")
            per_h = {h: dict(F=m(jF, lambda j, h=h: j["brier"][h]), C=m(jC, lambda j, h=h: j["brier"][h])) for h in H_KEYS}
            ptc, loc, hic = boot_diff([z["cost"] for z in zF], [z["cost"] for z in zC], n)
            ptl, lol, hil = boot_diff([z["logs"] for z in zF], [z["logs"] for z in zC], n)
            summary.append(dict(dataset=u[0], encoder=u[1], variant=v, comparison=f"{comp} vs C", model_F=name(kF, selF), model_C=name(kC, selC),
                                same_model=bool(same), seeds=seeds, brier_mean_F=float(np.mean(brF)), brier_mean_C=float(np.mean(brC)),
                                diff=float(pt), boot95=[float(lo), float(hi)], per_seed_diff=pd_, brier_by_h=per_h,
                                cost_diff={f"c{c}": {h: [float(ptc[ci, j]), float(loc[ci, j]), float(hic[ci, j])] for j, h in enumerate(H_KEYS)} for ci, c in enumerate(COSTS)},
                                logscore_diff={h: [float(ptl[j]), float(lol[j]), float(hil[j])] for j, h in enumerate(H_KEYS)}))
    L += ["", "Per-h Brier (F-type / C model, seed means):", ""]
    for s in summary:
        L.append(f"- {s['dataset']} {s['encoder']} {s['variant']} {s['comparison']}: " + "; ".join(
            f"{h} {s['brier_by_h'][h]['F']:.4f} / {s['brier_by_h'][h]['C']:.4f}" for h in H_KEYS))
    L += ["", "## 3. All candidates: test mean Brier at best_C (seed mean; descriptive)", ""]
    for u in sorted(cands, key=lambda x: (x[0] not in PRIMARY_DS, x)):
        parts = []
        for key in sorted(cands[u]):
            vals = [scored[r["cell_id"]]["by_sel"]["C"]["brier_mean"] for r in cands[u][key].values()
                    if r["cell_id"] in scored and not scored[r["cell_id"]]["by_sel"]["C"].get("missing") and not scored[r["cell_id"]]["by_sel"]["C"]["quarantined"]]
            if vals:
                parts.append(f"{key[0]} {key[1]} {np.mean(vals):.5f} (n={len(vals)})")
        if parts:
            L.append(f"- {u[0]} {u[1]}: " + "; ".join(parts))
    rt_rows = []
    for (u, v), row in choices.items():
        if row["rules"] is None:
            continue
        for rule in ("F", "C", "F_own"):
            key, sel = row["rules"][rule][:2]
            if key is None:
                continue
            js, pend = rt_collect(out_dir, cands[u][key], sel)
            if js or any("RT" not in x for x in pend):
                rt_rows.append((u, v, rule, f"{key[0]} {key[1]} @best_{sel}", js, pend))
    if rt_rows:
        L += ["", f"## 4. Remaining time (secondary): completion time from {N_ROLL} rollouts per (case, prefix fraction) (seed means)", "",
              "Shown only when all planned seeds of the chosen model are verified (SLA reproduction check passed for that cell and "
              "checkpoint) and have an RT file; otherwise the row is pending and lists why. 'all cases' = every test case processed.", "",
              "| data set | encoder | variant | rule | model | status | all cases | MAE of median | 80% coverage | 80% width | Brier P(dur > H) | truncated paths | capped draws |",
              "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for u, v, rule, nm, js, pend in rt_rows:
            if pend:
                L.append(f"| {u[0]} | {u[1]} | {v} | {rule} | {nm} | pending: {'; '.join(pend)} | | | | | | | |")
                continue
            mm = lambda k: float(np.mean([j[k] for j in js]))
            L.append(f"| {u[0]} | {u[1]} | {v} | {rule} | {nm} | verified ({len(js)} seeds) | {all(j.get('complete') for j in js)} | {mm('mae_median'):.4g} | "
                     f"{mm('coverage80'):.3f} | {mm('width80'):.4g} | {mm('brier_H'):.4f} | {mm('trunc_share'):.4f} | {sum(j['n_capped_draws'] for j in js)} |")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    _write_json(out_dir / "summary.json", dict(data=hdr, comparisons=summary, n_scored=len(scored), stale=stale, quarantined=quarantined,
                                               selections={f"{u[0]}|{u[1]}|{v}": dict(
                                                   complete=row["complete"], missing=row["missing"],
                                                   rules=None if row["rules"] is None else {
                                                       k: dict(key=list(x[0]) if x[0] else None, sel=x[1], means={f'{a_}|{b_}': m for (a_, b_), m in x[2].items()},
                                                               unrankable=[list(b) for b in x[3]]) for k, x in row["rules"].items()})
                                                   for (u, v), row in choices.items()}))
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("run", help="process_logs run directory")
    s.add_argument("--base", nargs="*", default=None, help="further runs whose triangular cells join the candidates")
    s.add_argument("--tasks", default="sla", help="sla, rt or sla,rt")
    s.add_argument("--datasets", nargs="*")
    s.add_argument("--only", nargs="*")
    s.add_argument("--limit", type=int)
    s.add_argument("--selected", action="store_true", help="only the cells chosen by the F or C rule (any variant)")
    s.add_argument("--force", action="store_true", help="recompute every requested file (valid, stale or quarantined)")
    s.add_argument("--retry-quarantined", action="store_true", help="recompute quarantined SLA files (never a valid cache)")
    s.add_argument("--cpu", action="store_true")
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--rt-max-cases", type=int, default=None, help="smoke: first N test cases only (marked incomplete)")
    s.add_argument("--rt-all", action="store_true", help="RT for every cell and checkpoint, not only the chosen ones")
    s.add_argument("--out", default=None, help="output directory (default RUN_DIR/alarms)")
    r = sub.add_parser("report")
    r.add_argument("run")
    r.add_argument("--base", nargs="*", default=None)
    r.add_argument("--datasets", nargs="*")
    r.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "score":
        cmd_score(a)
    else:
        print("\n".join(cmd_report(a)))


if __name__ == "__main__":
    main()
