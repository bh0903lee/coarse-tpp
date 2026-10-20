"""Diagnostics of trained models on synthetic data: the density inside a recorded cell (Section 8, S3). No training.

    python -m coarse_tpp.diagnostics true_law [--settings SC:1.0 SP:1.0] [--realizations 1 2 3 4 5] [--out FILE]
    python -m coarse_tpp.diagnostics models --run RUN_DIR [--cells GLOB] [--procs N] [--out FILE]

Profile: points s_j = tau~ + delta (j + 0.5) / 10, j = 0..9, inside the recorded cell; model: log q_B(s_j) with the
readout of its treatment (coarse_tpp.readout.B); true law: the true log density of t_i at (recorded t~_{i-1}) + s_j
given the latent past (0 before the latent t_{i-1}). Per event: log ratio = max - min over the points with positive
density. Per cell (or realization): median over the scored test events; reported ratio = exp(mean over cells of the
median log ratio), the cells taken at the learning rate chosen per (treatment, data set) by the mean dev C.
For every model also: B, C, and B_flat = C - log delta (the cell probability spread uniformly over the cell, a valid
density of s); B - B_flat is the value of the model's in-cell shape.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy import stats as SS

from . import data as D
from . import readout as Ro
from . import synthetic as Syn

NG = 10


def grid(delta):
    return delta * (np.arange(NG) + 0.5) / NG


def true_log_density_at(o: Syn.Oracle, x: np.ndarray) -> np.ndarray:
    """True log density of t_i at x_i given the latent past (x aligned with the events); -inf for x_i < t_{i-1}."""
    p = o.prev
    ok = x >= p
    xs = np.where(ok, x, p)
    d = xs - p
    proc = o.proc
    if proc == "SR":
        lp = SS.lognorm.logpdf(np.maximum(d, 1e-300), Syn.LN_SIG, scale=math.exp(Syn.LN_MU))
    elif proc == "NSR":
        lp = SS.gamma.logpdf(Syn._R(xs) - Syn._R(p), Syn.G_SHAPE, scale=Syn.G_SCALE) + np.log(Syn._r(xs))
    else:
        if proc == "SP":
            ll = np.zeros_like(xs)
        elif proc == "NSP":
            ll = np.log(Syn._r(xs))
        elif proc == "SC":
            ll = xs - np.arange(len(xs))
        elif proc in ("H1", "H2"):
            mu, ker = Syn.HAWKES[proc]
            ll = np.log(mu + sum(a * b * o.Ap[k] * np.exp(-b * d) for k, (a, b) in enumerate(ker)))
        else:
            raise ValueError(proc)
        lp = ll - o.Lam(xs)
    return np.where(ok, lp, -np.inf)


def profile_stats(lq: np.ndarray):
    """lq (n, NG) log densities (-inf = zero density) -> (log ratio over finite points (nan when none), entropy)."""
    fin = np.isfinite(lq)
    mx = np.where(fin, lq, -np.inf).max(-1)
    mn = np.where(fin, lq, np.inf).min(-1)
    anyf = fin.any(-1)
    lr = np.where(anyf, mx - mn, np.nan)
    m = np.where(anyf, mx, 0.0)[:, None]
    w = np.where(fin, np.exp(lq - m), 0.0)
    p = w / np.maximum(w.sum(-1, keepdims=True), 1e-300)
    H = -np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0).sum(-1)
    return lr, H


def true_law_unit(proc: str, r: int, delta: float, split: str = "test"):
    """Median log ratio and mean entropy of the true in-cell profile over the scored events of a split."""
    z = Syn.arrays(proc, r)
    o = Syn.Oracle(proc, z["t"])
    ev = Syn.event_arrays(proc, r, delta)
    start = z["t"] - ev["s"] + ev["gap"]                 # recorded cell start of t_i: t~_{i-1} + tau~_i
    sg = grid(delta)
    lq = np.stack([true_log_density_at(o, start + sg[j]) for j in range(NG)], axis=-1)
    idx = np.concatenate([np.arange(f, b) for a, b, f in Syn.windows(split, len(z["t"]))])
    lr, H = profile_stats(lq[idx])
    return dict(median_logratio=float(np.median(lr[np.isfinite(lr)])), entropy_mean=float(H.mean()), n=int(len(idx)))


def true_law(settings, out=None, realizations=Syn.REALIZATIONS):
    res = {}
    for proc, d in settings:
        per = [true_law_unit(proc, r, d) for r in realizations]
        m = float(np.mean([x["median_logratio"] for x in per]))
        res[f"{proc}|{d}"] = dict(proc=proc, delta=d, ratio=math.exp(m), entropy=float(np.mean([x["entropy_mean"] for x in per])),
                                  per_realization=per)
        print(f"{proc} delta={d}: in-cell density ratio of the true law {math.exp(m):.3f}", flush=True)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=1))
    return res


@torch.no_grad()
def model_cell(path: Path, bs: int = 128):
    from .models import build
    res = json.loads((path / "result.json").read_text())
    cfg = res["config"]
    delta = float(cfg["delta"])
    rule = Ro.RULE[cfg["treatment"]]
    v = D.load(cfg["dataset"], "test")
    st = D.coarse_stats(D.load(cfg["dataset"], "train"), delta)
    dev = torch.device("cpu")
    model = build(cfg, st, v.coarse.n_types).to(dev)
    model.load_state_dict(torch.load(path / "best_C.pt", map_location=dev, weights_only=True))
    model.eval()
    sg = torch.tensor(grid(delta), dtype=torch.float64)
    LR, H, Bs, Cs, TIE = [], [], [], [], []
    sp = v.coarse
    for i0 in range(0, len(sp), bs):
        idx = np.arange(i0, min(i0 + bs, len(sp)))
        b = D.make_batch(sp, idx, dev)
        L = b.iet.shape[1]
        m = (b.tmask * D.score_mask(v, idx, L, dev)) > 0
        tau = b.iet[:, 1:].double()
        law, tl = model(b, st)[:2]
        lq = torch.stack([Ro.B(law, tl, tau + sg[j], delta, rule) for j in range(NG)], dim=-1)
        lr, h = profile_stats(lq[m].numpy())
        sv = D.per_target(v, "lat_s", idx, L, dev)
        sv = torch.where(m, sv, torch.ones_like(sv))
        LR.append(lr); H.append(h); TIE.append((tau <= Ro.TIE0)[m].numpy())
        Bs.append(Ro.B(law, tl, sv, delta, rule)[m].numpy()); Cs.append(Ro.C(law, tl, tau, delta, rule)[m].numpy())
    lr, h, tie = np.concatenate(LR), np.concatenate(H), np.concatenate(TIE)
    Bv, Cv = np.concatenate(Bs), np.concatenate(Cs)
    med = lambda x: float(np.median(x[np.isfinite(x)])) if np.isfinite(x).any() else None
    return dict(cell=path.name, arm=res["arm"], dataset=cfg["dataset"], lr=cfg["lr_tag"], seed=cfg["seed"], delta=delta,
                rule=rule, n=int(len(lr)), n_tie=int(tie.sum()), dev_C=res["by_selection"]["C"]["dev_criterion"],
                median_logratio=med(lr), median_logratio_tie=med(lr[tie]) if tie.any() else None,
                median_logratio_nontie=med(lr[~tie]), entropy_mean=float(h.mean()),
                B=float(Bv.mean()), C=float(Cv.mean()), B_flat=float((Cv - math.log(delta)).mean()))


def _mc(p):
    torch.set_num_threads(1)
    return model_cell(Path(p))


def group_of(ds: str) -> str:
    pr = Syn.parse(ds)
    return f"synth_{pr[0]}_d{Syn.dcode(pr[2])}" if pr else ds


def pool_cells(rows):
    """Per (treatment, setting): lr chosen per (treatment, data set) by the mean dev C over seeds; means over cells."""
    by = {}
    for x in rows:
        by.setdefault((x["arm"], x["dataset"]), {}).setdefault(x["lr"], []).append(x)
    agg = {}
    for (arm, ds), lrs in by.items():
        lr = max(lrs, key=lambda k: np.mean([x["dev_C"] for x in lrs[k]]))
        agg.setdefault((arm, group_of(ds)), []).extend(lrs[lr])
    out = {}
    for (arm, g), xs in sorted(agg.items()):
        mm = lambda k: float(np.mean([x[k] for x in xs if x[k] is not None])) if any(x[k] is not None for x in xs) else None
        d = dict(arm=arm, group=g, n_cells=len(xs), median_logratio=mm("median_logratio"), entropy_mean=mm("entropy_mean"),
                 B=mm("B"), C=mm("C"), B_flat=mm("B_flat"))
        d["ratio"] = math.exp(d["median_logratio"]) if d["median_logratio"] is not None else None
        out[f"{arm}|{g}"] = d
    return out


def models(run, pattern="*", procs=4, out=None):
    import multiprocessing as mp
    run = Path(run).resolve()
    paths = [str(p) for p in sorted((run / "cells").iterdir())
             if fnmatch.fnmatch(p.name, pattern) and (p / "best_C.pt").exists() and (p / "result.json").exists()]
    with mp.get_context("fork").Pool(procs) as pool:
        rows = sorted(pool.map(_mc, paths), key=lambda x: x["cell"])
    agg = pool_cells(rows)
    dest = Path(out) if out else run / "inbin_profile.json"
    dest.write_text(json.dumps(dict(by_setting=agg, cells=rows), indent=1))
    for k, d in agg.items():
        print(f"{k}: n={d['n_cells']} in-cell ratio {d['ratio']:.2f}  B {d['B']:.3f}  B_flat {d['B_flat']:.3f}", flush=True)
    return agg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("true_law", "models"))
    ap.add_argument("--settings", nargs="*", default=["SC:1.0", "SP:1.0"])
    ap.add_argument("--realizations", nargs="*", type=int, default=list(Syn.REALIZATIONS))
    ap.add_argument("--run", default=None)
    ap.add_argument("--cells", default="*")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "true_law":
        true_law([(s.split(":")[0], float(s.split(":")[1])) for s in a.settings], a.out, a.realizations)
    else:
        models(a.run, a.cells, a.procs, a.out)


if __name__ == "__main__":
    main()
