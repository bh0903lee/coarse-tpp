"""Predictive accuracy of saved models of the rank_change experiment (Supplementary S3, Table S3). No training.

    python -m coarse_tpp.predictive score RUN_DIR [--only CELL_ID] [--limit N] [--force]
    python -m coarse_tpp.predictive report RUN_DIR [--audit results/audit/audit.json]

score: for every unit (data set, encoder, head, K, treatment) of RUN_DIR at its learning rate chosen by the dev mean
of C (coarse_tpp.analysis.select_lr, as in Table 4), the checkpoint best_C.pt is loaded into the rebuilt model
(coarse_tpp.models.build with coarse_tpp.data.coarse_stats of the train split, as in coarse_tpp.train.run) and
evaluated on the test split. The model's law q (and tie probability pi of the point-mass model) is converted by the
treatment's readout (coarse_tpp.readout) into the law q_B of the waiting time s = t_i - t~_{i-1}, with distribution
function F:
  raw          F = Q
  smooth       F(x) = (1/delta) int_{x - delta}^{x} Q(u) du
  point_mass   F(x) = pi min(x / delta, 1) + (1 - pi) smooth
Per scored test event, against the recorded cell [tau~, tau~ + delta) and its midpoint y = tau~ + delta / 2:
  C           recorded-cell score, recomputed with the batches of coarse_tpp.train.evaluate; it must match the
              stored test C within C_TOL, else the cell is flagged (quarantined) and left out of the report
  rmse        root mean squared error of the mean of q_B (inf when the mixture mean overflows)
  mae_median / rmse_median   absolute / squared error of the median of q_B
  crps        cell CRPS (ranked probability score x delta) over the grid u_j = r + j delta, r = tau~ - m delta,
              m = floor(tau~ / delta): delta [sum_{j<=m} F(u_j)^2 + sum_{j>m} (1 - F(u_j))^2]
  twcrps      the same sum restricted to the points u_j <= the largest gap of the train split (truncated CRPS)
The N0 terms next to each boundary of the sums are added exactly, the rest by the midpoint rule as an integral
(Gauss-Legendre in log x). Writes RUN_DIR/predictive/cells/<cell_id>.json; a later call skips cells already scored.

report: per (data set, encoder), the top treatment by the floor score (stored test floor_1e-6) and by the interval
score (stored test C), each by the mean over seeds; the seed-paired difference (C-top minus floor-top) of every
measure with its one-sided 95% upper bound; Kendall's tau between the ranking of the treatments by each measure
and by the two scores. Writes RUN_DIR/predictive/report.md and summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import analysis as A
from . import data as D
from . import experiments as X
from . import lik
from . import readout as Ro
from .data import coarse_stats, load, score_mask
from .models import build

C_TOL = 1e-5              # largest accepted |recomputed - stored| test C of a cell
N0 = 1024                 # exact terms next to each boundary of the cell CRPS sum
N_PANELS, N_GL = 128, 16  # log-x Gauss-Legendre for the midpoint-rule remainder
LOG_BIG = math.log(1e300)
MEASURES = {"rmse": "RMSE", "rmse_median": "RMSE_med", "mae_median": "MAE_med", "crps": "CRPS", "twcrps": "twCRPS"}


# ------------------------------------------------------------------ predictive law of s (closed forms)
def _phi(z):
    return torch.special.ndtr(z)


def q_cdf(law, x):
    """Q(x) = P(tau < x) of the log-normal mixture; law parameters (N, K), x (N, M) -> (N, M); 0 for x <= 0."""
    xp = x.clamp(min=1e-300)
    z = (torch.log(xp).unsqueeze(-1) - law.mu.unsqueeze(-2)) / law.sigma.unsqueeze(-2)
    v = (law.log_w.exp().unsqueeze(-2) * _phi(z)).sum(-1)
    return torch.where(x > 0, v, torch.zeros_like(v))


def smooth_cdf(law, x, delta):
    """(1/delta) int_{x-delta}^{x} Q(u) du without cancellation: per component, with a = max(x - delta, 0), b = x,
    int_a^b Phi_k = (b - a) Phi_k(z_b) - int_a^b (u - a) p_k(u) du; the last term is lik._mix_log_piece (left
    lever, safe for narrow intervals). 0 for x <= 0."""
    N, M = x.shape
    xb = x.clamp(min=1e-300)
    a = (xb - delta).clamp(min=0.0)
    lb = lik._broadcast_law(law, M)                               # parameters (N, M, K)
    zb = (torch.log(xb).unsqueeze(-1) - lb.mu) / lb.sigma
    first = ((xb - a) / delta).unsqueeze(-1) * _phi(zb)
    piece = torch.exp(lik._mix_log_piece(lb, a, xb, left=True, delta=delta))
    v = (lb.log_w.exp() * (first - piece)).sum(-1)
    # far out (x > 1e9 delta; float64 loses x - delta beyond 2^53 delta) the average over the last cell is
    # Q(x - delta / 2) to O(delta^2 q'), i.e. Q(x) to relative precision ~1e-9 of the cell scale
    far = x > 1e9 * delta
    v = torch.where(far, q_cdf(law, torch.where(far, x - 0.5 * delta, torch.ones_like(x))), v)
    return torch.where(x > 0, v.clamp(0.0, 1.0), torch.zeros_like(v))


def cdf(law, pi, rule, delta, x):
    """Distribution function F of q_B at x (N, M) under the readout `rule` (coarse_tpp.readout.RULE)."""
    if rule == "raw":
        return q_cdf(law, x)
    sm = smooth_cdf(law, x, delta)
    if rule == "smooth":
        return sm
    if rule != "point_mass":
        raise ValueError(rule)
    p = pi.unsqueeze(-1)
    return p * (x / delta).clamp(0.0, 1.0) + (1 - p) * sm


def mean(law, pi, rule, delta):
    """E[s] under q_B (N,); +inf when the mixture mean exceeds 1e300."""
    lm = torch.logsumexp(law.log_w + law.mu + 0.5 * law.sigma ** 2, dim=-1)
    m = torch.where(lm < LOG_BIG, torch.exp(lm.clamp(max=LOG_BIG)), torch.full_like(lm, math.inf))
    if rule == "raw":
        return m
    if rule == "smooth":
        return m + 0.5 * delta
    return pi * 0.5 * delta + (1 - pi) * (m + 0.5 * delta)


def median(law, pi, rule, delta, iters=64):
    """Median of q_B by bisection in log x (N,)."""
    lo = torch.minimum((law.mu - 12 * law.sigma).min(-1).values, torch.full_like(law.mu[:, 0], math.log(delta) - 30))
    hi = (law.mu + 12 * law.sigma).max(-1).values.clamp(min=math.log(delta) + 1.0)
    lo, hi = lo.clamp(min=-690.0), hi.clamp(max=700.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = cdf(law, pi, rule, delta, torch.exp(mid).unsqueeze(-1)).squeeze(-1)
        up = f >= 0.5
        hi = torch.where(up, mid, hi)
        lo = torch.where(up, lo, mid)
    return torch.exp(0.5 * (lo + hi))


def _gl_log_integral(fn, a, b, panels=N_PANELS):
    """int_a^b fn(x) dx for 0 < a, per row (N,), by Gauss-Legendre in log x (panels x N_GL nodes); 0 where b <= a."""
    x, w = lik.gauss_legendre(N_GL, device=a.device)
    ok = b > a
    la = torch.log(a.clamp(min=1e-300))
    lb = torch.where(ok, torch.log(b.clamp(min=1e-300)), la + 1e-3)
    node = ((torch.arange(panels, dtype=torch.float64, device=a.device).unsqueeze(-1) + x) / panels).reshape(-1)
    wt = w.repeat(panels) / panels
    xs = torch.exp(la.unsqueeze(-1) + (lb - la).unsqueeze(-1) * node)          # (N, M)
    val = (fn(xs) * xs * wt).sum(-1) * (lb - la)
    return torch.where(ok, val, torch.zeros_like(val))


def cell_crps(law, pi, rule, delta, tau, n0=N0, panels=N_PANELS, cap=None):
    """Cell CRPS (time units) of q_B against the recorded cell [tau, tau + delta), per event (N,).

    Returns (crps, twcrps, lc): twcrps counts only the points u_j <= cap (NaN when cap is None), and
    lc = log(F(tau + delta) - F(tau)) is the cell log-probability from the same F (a cross-check of C)."""
    tau = tau.double()
    m = torch.floor(tau / delta + 1e-7)
    r = (tau - m * delta).clamp(min=0.0)
    F = lambda x: cdf(law, pi, rule, delta, x)
    j = torch.arange(n0, dtype=torch.float64, device=tau.device)
    x1 = r.unsqueeze(-1) + j * delta                                   # sum_{j=0}^{m} F(r + j delta)^2
    g1 = F(x1) ** 2
    in1 = j.unsqueeze(0) <= m.unsqueeze(-1)
    x2 = r.unsqueeze(-1) + (m.unsqueeze(-1) + 1 + j) * delta           # sum_{j>m} (1 - F(r + j delta))^2
    g2 = (1 - F(x2)) ** 2
    a1 = r + (n0 - 0.5) * delta
    b1 = torch.where(m >= n0, r + (m + 0.5) * delta, a1)
    a2 = r + (m + n0 + 0.5) * delta
    zmax = torch.clamp(0.5 * law.sigma + 10.0, min=12.0)
    hi = torch.exp((law.mu + law.sigma * zmax).max(-1).values.clamp(max=700.0)) + 2 * delta
    f1 = lambda xs: F(xs) ** 2
    f2 = lambda xs: (1 - F(xs)) ** 2

    def total(H):
        s1 = (g1 * (in1 & (x1 <= H)).double()).sum(-1)
        s1 = s1 + _gl_log_integral(f1, a1, torch.clamp(b1, max=H + 0.5 * delta), panels) / delta
        s2 = (g2 * (x2 <= H).double()).sum(-1)
        s2 = s2 + _gl_log_integral(f2, a2, torch.maximum(torch.clamp(hi, max=H + 0.5 * delta), a2), panels) / delta
        return delta * (s1 + s2)

    crps = total(math.inf)
    tw = total(float(cap)) if cap is not None else torch.full_like(crps, math.nan)
    Fc = F(torch.stack([tau, tau + delta], -1))
    lc = torch.log((Fc[:, 1] - Fc[:, 0]).clamp(min=1e-300))
    return crps, tw, lc


# ------------------------------------------------------------------ saved models
@torch.no_grad()
def per_event(model, views, stats, cfg, device, bs=64):
    """Mixture parameters, tie probabilities, recorded gaps and C of every scored test target (flat tensors), with
    the batches, masks and readout call of coarse_tpp.train.evaluate. tau32 is the gap as the model saw it (float32),
    tau64 the stored gap (float64)."""
    model.eval()
    delta = float(cfg["delta"])
    rule = Ro.RULE[cfg["treatment"]]
    sp = views.coarse
    out = {k: [] for k in ("log_w", "mu", "log_sigma", "pi", "tau32", "tau64", "C")}
    c_sum, n_sum = 0.0, 0.0
    for i0 in range(0, len(sp), bs):
        idx = np.arange(i0, min(i0 + bs, len(sp)))
        b = D.make_batch(sp, idx, device)
        L = b.iet.shape[1]
        msk = (b.tmask * score_mask(views, idx, L, device)).double()
        tau = b.iet[:, 1:].double()
        law, tl = model(b, stats)
        c = Ro.C(law, tl, tau, delta, rule)
        c = torch.where(msk > 0, c, torch.zeros_like(c))
        c_sum += float((c * msk).sum()); n_sum += float(msk.sum())
        sel = msk > 0
        out["log_w"].append(law.log_w[sel]); out["mu"].append(law.mu[sel]); out["log_sigma"].append(law.log_sigma[sel])
        out["pi"].append(torch.sigmoid(tl.double())[sel] if tl is not None else torch.zeros_like(tau[sel]))
        out["tau32"].append(tau[sel]); out["C"].append(c[sel])
        t64 = torch.zeros_like(tau)
        for bi, i in enumerate(idx):
            g = torch.tensor(sp.iet[i][1:], dtype=torch.float64, device=device)
            t64[bi, :len(g)] = g
        out["tau64"].append(t64[sel])
    P = {k: torch.cat(v, 0) for k, v in out.items()}
    P.update(C_mean=c_sum / n_sum, n=int(round(n_sum)))
    return P


def measures(P, rule, delta, chunk=512, n0=None, panels=None, cap=None):
    """Mean measures of the module docstring over the events of P (output of per_event)."""
    N = P["mu"].shape[0]
    acc = {k: [] for k in ("mean", "med", "crps", "lc", "tw")}
    for i in range(0, N, chunk):
        sl = slice(i, i + chunk)
        law = lik.LogNormMix(P["log_w"][sl], P["mu"][sl], P["log_sigma"][sl])
        pi = P["pi"][sl]
        acc["mean"].append(mean(law, pi, rule, delta)); acc["med"].append(median(law, pi, rule, delta))
        cr, tw, lc = cell_crps(law, pi, rule, delta, P["tau64"][sl], n0=n0 or N0, panels=panels or N_PANELS, cap=cap)
        acc["crps"].append(cr); acc["lc"].append(lc); acc["tw"].append(tw)
    mu_, med, crps, lc, tw = (torch.cat(acc[k]) for k in ("mean", "med", "crps", "lc", "tw"))
    y = P["tau64"] + 0.5 * delta
    se = (mu_ - y) ** 2
    fse = torch.isfinite(se)
    tie = P["tau64"] <= 0
    return dict(rmse=float(torch.sqrt(se.mean())) if bool(fse.all()) else math.inf,
                n_mean_nonfinite=int((~torch.isfinite(mu_)).sum()), n_sqerr_nonfinite=int((~fse).sum()),
                rmse_finite=float(torch.sqrt(se[fse].mean())) if bool(fse.any()) else None,
                mae_median=float((med - y).abs().mean()), rmse_median=float(torch.sqrt(((med - y) ** 2).mean())),
                crps=float(crps.mean()), twcrps=float(tw.mean()), twcrps_cap=cap,
                crps_tie=float(crps[tie].mean()) if bool(tie.any()) else None,
                crps_nontie=float(crps[~tie].mean()) if bool((~tie).any()) else None,
                crps_median_event=float(crps.median()), crps_nonfinite=int((~torch.isfinite(crps)).sum()),
                C_from_cdf=float(lc.mean()), max_abs_C_cdf_vs_C=float((lc - P["C"])[P["C"] > -30].abs().max()),
                n_ties=int(tie.sum()))


def score_cell(res, device):
    cfg = res["config"]
    ds, delta = cfg["dataset"], float(cfg["delta"])
    rule = Ro.RULE[cfg["treatment"]]
    raw = bool(cfg.get("raw_gaps"))
    tr, te = load(ds, "train", raw=raw), load(ds, "test", raw=raw)
    stats = coarse_stats(tr, delta)
    model = build(cfg, stats, tr.coarse.n_types).to(device)
    model.load_state_dict(torch.load(Path(cfg["out_dir"]) / "best_C.pt", map_location=device))
    P = per_event(model, te, stats, cfg, device)
    stored = float(res["by_selection"]["C"]["test"]["C"])
    met = measures(P, rule, delta, cap=float(stats.max_train_iet))
    diff = abs(P["C_mean"] - stored)
    u = A.unit_key(res)
    return dict(cell_id=res["cell_id"], dataset=ds, encoder=u[1], head=u[2], K=u[3], arm=res["arm"],
                treatment=cfg["treatment"], rule=rule, seed=res["seed"], lr_tag=cfg["lr_tag"], delta=delta,
                n_events=P["n"], C_recomputed=P["C_mean"], C_stored=stored, C_absdiff=diff,
                quarantined=bool(diff > C_TOL),
                floor_stored=float(res["by_selection"]["C"]["test"]["floor_1e-6"]), **met)


def selected_cells(run):
    """Results of the units at their dev-C learning rate (every seed), in a fixed order."""
    units = A.select_lr(A.load_cells(Path(run)))
    return [res for u, seeds in sorted(units.items()) for sd, res in sorted(seeds.items())]


def score(run, only=None, limit=None, force=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cells = selected_cells(run)
    if only:
        cells = [c for c in cells if c["cell_id"] == only]
    if limit:
        cells = cells[:limit]
    od = Path(run) / "predictive" / "cells"
    od.mkdir(parents=True, exist_ok=True)
    for i, res in enumerate(cells):
        p = od / f"{res['cell_id']}.json"
        if p.exists() and not force:
            continue
        t0 = time.time()
        r = score_cell(res, device)
        r["wall_sec"] = round(time.time() - t0, 1)
        tmp = p.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(r, indent=1))
        os.replace(tmp, p)
        print(f"[{i + 1}/{len(cells)}] {res['cell_id']}: |C - stored| {r['C_absdiff']:.2e}, MAE_med {r['mae_median']:.5g}, "
              f"twCRPS {r['twcrps']:.5g} ({r['wall_sec']} s)", flush=True)


# ------------------------------------------------------------------ report
def t_upper(d):
    """(mean, one-sided 95% upper bound) of seed-paired differences."""
    m, lo, _ = A.t_one_sided(d)
    return m, (m + (m - lo)) if math.isfinite(lo) else float("nan")


def kendall(a, b):
    """Kendall's tau; non-finite values rank last."""
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    a = np.where(np.isfinite(a), a, -1e308)
    b = np.where(np.isfinite(b), b, -1e308)
    return float(stats.kendalltau(a, b)[0])


def summarize(rows, audit, arms=X.RANK_ARMS, n_seeds=len(X.REAL_SEEDS), controls=X.RANK_CONTROLS, tied_min=0.01):
    """Summary rows per (data set, encoder) from the per-cell rows (complete groups only) and the list of
    incomplete groups."""
    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["encoder"]), {}).setdefault(r["arm"], {})[r["seed"]] = r
    avg = lambda d, k: float(np.mean([x[k] for x in d.values()]))
    out, incomplete = [], []
    for g in sorted(groups):
        tr = groups[g]
        ms = [m for m in arms if m in tr and len(tr[m]) == n_seeds]
        if len(ms) < len(arms):
            incomplete.append(dict(dataset=g[0], encoder=g[1], complete_treatments=len(ms)))
            continue
        fl = [avg(tr[m], "floor_stored") for m in ms]
        cc = [avg(tr[m], "C_stored") for m in ms]
        ftop, ctop = ms[int(np.argmax(fl))], ms[int(np.argmax(cc))]
        tie = (audit.get(g[0]) or {}).get("tie_share", float("nan"))
        row = dict(dataset=g[0], encoder=g[1], tie_share=tie, tied=bool(tie >= tied_min and g[0] not in controls),
                   floor_top=ftop, C_top=ctop, changed=ftop != ctop)
        for k in MEASURES:
            vf, vc = avg(tr[ftop], k), avg(tr[ctop], k)
            seeds = sorted(set(tr[ftop]) & set(tr[ctop]))
            d = [tr[ctop][s][k] - tr[ftop][s][k] for s in seeds]
            dm, up = t_upper(d) if all(math.isfinite(x) for x in d) else (float("nan"), float("nan"))
            row.update({f"{k}_floor": vf, f"{k}_C": vc, f"d_{k}": dm, f"d_{k}_up95": up, f"{k}_C_no_worse": bool(vc <= vf)})
        for k in MEASURES:
            neg = [-avg(tr[m], k) for m in ms]                  # higher = better
            row[f"tau_floor_{k}"] = kendall(fl, neg)
            row[f"tau_C_{k}"] = kendall(cc, neg)
        row["tau_floor_C"] = kendall(fl, cc)
        row["by_treatment"] = {m: {k: avg(tr[m], k) for k in ("C_stored", "floor_stored") + tuple(MEASURES)} for m in ms}
        out.append(row)
    return out, incomplete


def report(run, audit_path=None, **kw):
    run = Path(run)
    cells = selected_cells(run)
    rows, missing = {}, []
    for res in cells:
        p = run / "predictive" / "cells" / f"{res['cell_id']}.json"
        if p.exists():
            rows[res["cell_id"]] = json.loads(p.read_text())
        else:
            missing.append(res["cell_id"])
    quarantined = sorted(k for k, r in rows.items() if r["quarantined"])
    good = [r for r in rows.values() if not r["quarantined"]]
    audit = A.audit_rows(audit_path)
    if not audit:
        print(f"warning: no audit file at {audit_path or X.DEFAULT_AUDIT}; tie shares unknown, no combination is marked tied",
              file=sys.stderr)
    out_rows, incomplete = summarize(good, audit, **kw)
    mx = lambda k: max([r[k] for r in rows.values()] or [float("nan")])
    L = ["# Predictive accuracy of the selected models (no training)", "",
         "Checkpoints best_C.pt at the learning rate chosen by dev C; test split.",
         f"Cells selected: {len(cells)}; scored: {len(rows)}; missing: {len(missing)}; "
         f"quarantined (|C recomputed - stored| > {C_TOL}): {len(quarantined)}.",
         f"Max |C recomputed - stored|: {mx('C_absdiff'):.3e}. Max per-event |log(F(cell)) - C| for C > -30 (a cross-check; "
         f"the log of a difference of F loses its digits when the cell probability is below ~1e-9 next to F = 1): "
         f"{mx('max_abs_C_cdf_vs_C'):.3e}.",
         f"Cells with a non-finite predictive mean for some event: {sum(r['n_mean_nonfinite'] > 0 for r in rows.values())}.", ""]
    if quarantined:
        L += ["Quarantined: " + ", ".join(quarantined), ""]
    if incomplete:
        L += ["Incomplete groups (left out): " + ", ".join(f"{x['dataset']}/{x['encoder']}" for x in incomplete), ""]
    L += ["## 1. Top treatment by the floor score (floor_1e-6) and by the interval score (C): seed means; "
          "d = C-top - floor-top with its one-sided 95% upper bound", "",
          "| data set | encoder | tied | floor top | C top | " + " | ".join(f"{n} floor | {n} C | d{n} [up95]" for n in MEASURES.values()) + " |",
          "|---|---|---|---|---|" + "---:|---:|---|" * len(MEASURES)]
    for r in out_rows:
        txt = []
        for k in MEASURES:
            txt += [f"{r[f'{k}_floor']:.6g}", f"{r[f'{k}_C']:.6g}", f"{r[f'd_{k}']:+.3g} [{r[f'd_{k}_up95']:+.3g}]"]
        L.append(f"| {r['dataset']} | {r['encoder']} | {'yes' if r['tied'] else 'no'} | {r['floor_top']} | {r['C_top']} | "
                 + " | ".join(txt) + " |")
    L += ["", "## 2. Kendall's tau between the treatment rankings (measures negated so that higher = better)", "",
          "| data set | encoder | tied | tau(floor, C) | " + " | ".join(f"tau(floor, {n}) | tau(C, {n})" for n in MEASURES.values()) + " |",
          "|---|---|---|---:|" + "---:|---:|" * len(MEASURES)]
    for r in out_rows:
        L.append(f"| {r['dataset']} | {r['encoder']} | {'yes' if r['tied'] else 'no'} | {r['tau_floor_C']:+.2f} | "
                 + " | ".join(f"{r[f'tau_floor_{k}']:+.2f} | {r[f'tau_C_{k}']:+.2f}" for k in MEASURES) + " |")
    T = [r for r in out_rows if r["tied"]]
    Tc = [r for r in T if r["changed"]]
    L += ["", f"## 3. Tied combinations: {len(T)}; top treatment changed: {len(Tc)}", ""]
    for k, n in MEASURES.items():
        L.append(f"- {n}: C-top lower in {sum(r[f'd_{k}'] < 0 for r in Tc)}/{len(Tc)}, upper bound below 0 in "
                 f"{sum(r[f'd_{k}_up95'] < 0 for r in Tc)}/{len(Tc)}; tau(C, {n}) > tau(floor, {n}) in "
                 f"{sum(r[f'tau_C_{k}'] > r[f'tau_floor_{k}'] for r in T)}/{len(T)}")
    od = run / "predictive"
    od.mkdir(parents=True, exist_ok=True)
    (od / "report.md").write_text("\n".join(L) + "\n")
    (od / "summary.json").write_text(json.dumps(out_rows, indent=1))
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("score", "report"))
    ap.add_argument("run")
    ap.add_argument("--only", default=None, help="score only this cell id")
    ap.add_argument("--limit", type=int, default=None, help="score at most N cells")
    ap.add_argument("--force", action="store_true", help="re-score cells already scored")
    ap.add_argument("--audit", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "score":
        score(a.run, a.only, a.limit, a.force)
    else:
        print("\n".join(report(a.run, a.audit)))


if __name__ == "__main__":
    main()
