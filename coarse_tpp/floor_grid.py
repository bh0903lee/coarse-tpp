"""Floor score on a continuous grid of floors: the trained models of a rank_change run re-scored at 37 floors
(Supplementary S4, "Floors between and beyond the three", Fig. S4). Exploratory, no training.

    python -m coarse_tpp.floor_grid score RUN_DIR [--shard i/n] [--limit N] [--cpu] [--force] [--retry-quarantined]
    python -m coarse_tpp.floor_grid report RUN_DIR

RUN_DIR is a rank_change run. Models: all eight treatments of every (data set, encoder) of the mixture head (K = 8),
the learning rate of each (unit, treatment) chosen by the dev-C mean over seeds (coarse_tpp.analysis.select_lr, the
rule of Table 4), checkpoint best_C.pt, the seeds common to all treatments. These are the models of Table 4.
Work item = (unit, seed).

Floor score at floor eps: mean over the scored test targets of log q(max(tau~, eps)) on the raw model density, as
coarse_tpp.train.evaluate computes floor_1e-10 / floor_1e-6 / floor_1e-5 (the tie logit of the point-mass treatment
is not used). GRID: 37 floors from 1e-12 to 1e-3, a quarter decade apart, in the time unit of each data set; it
contains the three training floors. The interval score C of each model is the stored test C (result.json,
by_selection.C.test.C); it does not depend on eps.

Validity: the model is rebuilt as coarse_tpp.train.run does; every model must reproduce its stored test floor_1e-6
within REPRO_TOL and its stored event and tie counts, else the work item is quarantined and left out of the report.
A cached file is bound to the grid, the selected cells and the SHA-256 of their checkpoints; a quarantined file is
recomputed only with --retry-quarantined (for instance on another device) or --force.

Report (OUT/report.md, OUT/curves.csv, OUT/summary.json), per unit, on seed means:
  (a) number of changes of the top treatment by the floor score along the grid
  (b) share of grid floors whose top treatment is the top by C
  (c) share of grid floors whose top treatment is the model trained with the nearest training floor (log distance)
  (d) min / median / max over the grid of Kendall's tau between the floor ranking and the C ranking
  check: the top at eps = 1e-6 equals the top by the stored test floor_1e-6 (Table 4)
and the structure quoted in Supplementary S4: whether each floor-trained model is first at its own training floor,
the grid floors other than the three at which a floor-trained model is first, the deficit of the model trained with
1e-6 behind the top treatment a quarter decade on either side of 1e-6, and the largest spread of the eight treatments
over the grid. curves.csv holds the seed-mean floor score of every treatment at every grid floor; summary.json the
top treatment per floor (the rows of Fig. S4).
"""
from __future__ import annotations

import argparse
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
from . import readout as Ro
from .component_edits import ckpt_sha, item_name
from .data import coarse_stats, load, score_mask
from .models import build

ARMS = X.RANK_ARMS
TRAINED_FLOORS = dict(X.FLOOR_ARMS)
GRID = tuple(float(10.0 ** (-12 + 0.25 * i)) for i in range(37))
TIE_MIN = 0.01
REPRO_TOL = 1e-5
MIN_CHANGES = 2
VERSION = 1


# ------------------------------------------------------------------ pure functions (tests/test_floor_grid.py)
def grid_index(eps: float) -> int:
    """Index of the grid floor equal to eps (relative tolerance 1e-9)."""
    for i, g in enumerate(GRID):
        if abs(g - eps) <= 1e-9 * eps:
            return i
    raise KeyError(eps)


def nearest_trained(eps: float) -> str:
    """Treatment trained with the floor nearest to eps in log distance (ties broken towards the smaller floor)."""
    return min(TRAINED_FLOORS, key=lambda a: (abs(math.log(TRAINED_FLOORS[a]) - math.log(eps)), TRAINED_FLOORS[a]))


def top(scores: dict) -> str:
    """Treatment with the highest score; exact ties broken by the order of ARMS."""
    return max(scores, key=lambda a: (scores[a], -ARMS.index(a)))


def n_changes(tops) -> int:
    return int(sum(a != b for a, b in zip(tops[:-1], tops[1:])))


def kendall(x, y) -> float:
    from scipy import stats
    return float(stats.kendalltau(x, y)[0])


def unit_curve(seed_recs, c_means):
    """seed_recs: per-seed records {arm: {"tie": [grid], "non": [grid], "n": n}}; c_means {arm: mean C}. Returns the
    seed-mean floor scores {arm: array over GRID} and the metrics (a)-(d)."""
    S = {a: np.mean([(np.asarray(r[a]["tie"]) + np.asarray(r[a]["non"])) / r[a]["n"] for r in seed_recs], axis=0)
         for a in ARMS}
    tops = [top({a: S[a][i] for a in ARMS}) for i in range(len(GRID))]
    top_c = top(c_means)
    taus = [kendall([S[a][i] for a in ARMS], [c_means[a] for a in ARMS]) for i in range(len(GRID))]
    met = dict(changes=n_changes(tops), share_top_eq_c=float(np.mean([t == top_c for t in tops])),
               share_top_nearest_trained=float(np.mean([t == nearest_trained(g) for t, g in zip(tops, GRID)])),
               tau_min=float(np.min(taus)), tau_median=float(np.median(taus)), tau_max=float(np.max(taus)),
               top_c=top_c, tops=tops, top_at_1e6=tops[grid_index(1e-6)])
    return S, met


def structure(S, tops):
    """Where the floor-trained models are first: at their own training floor (per arm), at how many other grid floors;
    deficit (top minus the 1e-6 model) a quarter decade below / above 1e-6; largest spread of the treatments."""
    own = {a: tops[grid_index(e)] == a for a, e in TRAINED_FLOORS.items()}
    trained_idx = {grid_index(e) for e in TRAINED_FLOORS.values()}
    other = sum(t in TRAINED_FLOORS for i, t in enumerate(tops) if i not in trained_idx)
    i6 = grid_index(1e-6)
    deficit = lambda i: float(max(S[a][i] for a in ARMS) - S["floor_1e-6"][i])
    lo, hi = deficit(i6 - 1), deficit(i6 + 1)
    spread = float(max(max(S[a][i] for a in ARMS) - min(S[a][i] for a in ARMS) for i in range(len(GRID))))
    return dict(own_top=own, all_own_top=all(own.values()), n_other_floor_top=int(other), deficit_below_1e6=lo,
                deficit_above_1e6=hi, loses_next_to_1e6=bool(lo > 0 and hi > 0), max_spread=spread)


# ------------------------------------------------------------------ units and cache state
def selected(run):
    """{(dataset, encoder): {arm: {seed: result}}} at the dev-C lr of each (unit, arm), mixture head, K = 8."""
    out = {}
    for (ds, enc, head, K, arm), seeds in A.select_lr(A.load_cells(Path(run))).items():
        if head == "mix" and K == 8 and arm in ARMS:
            out.setdefault((ds, enc), {})[arm] = seeds
    return out


def unit_seeds(arms):
    """Seeds common to all treatments of a unit (all treatments must be present)."""
    if set(arms) != set(ARMS):
        raise ValueError(f"missing treatments {sorted(set(ARMS) - set(arms))}")
    return sorted(set.intersection(*(set(arms[a]) for a in ARMS)))


def tie_share(arms, sel="C") -> float:
    """Stored test tie share of a unit (identical across its cells; checked)."""
    ts = {round(r["by_selection"][sel]["test"]["tie_share"], 12) for s in arms.values() for r in s.values()}
    if len(ts) != 1:
        raise SystemExit(f"test tie share differs across the cells of a unit: {ts}")
    return ts.pop()


def work_items(sel):
    return [(u[0], u[1], s) for u in sorted(sel) for s in unit_seeds(sel[u])]


def expected_models(sel, ds, enc, seed):
    """{arm: (cell_id, checkpoint sha)} of the models the current selection assigns to a work item."""
    out = {}
    for arm in ARMS:
        cfg = sel[(ds, enc)][arm][seed]["config"]
        out[arm] = (cfg["cell_id"], ckpt_sha(Path(cfg["out_dir"]) / "best_C.pt"))
    return out


def cache_state(path, expected=None):
    """'ok', 'quarantined', 'stale' (cached for other cells or checkpoints than `expected` = {arm: (cell_id, sha)})
    or 'invalid' (absent, unreadable, other version or grid, incomplete)."""
    try:
        r = json.loads(Path(path).read_text())
        if r.get("version") != VERSION or r.get("grid") != list(GRID) or set(r["models"]) != set(ARMS):
            return "invalid"
        for m in r["models"].values():
            s = np.asarray(m["tie"], float) + np.asarray(m["non"], float)
            if s.shape != (len(GRID),) or m["n"] <= 0:
                return "invalid"
        if expected is not None:
            for arm, (cid, sha) in expected.items():
                m = r["models"][arm]
                if m.get("cell_id") != cid or m.get("ckpt_sha256") != sha:
                    return "stale"
        return "quarantined" if r.get("quarantined", True) else "ok"
    except Exception:
        return "invalid"


def should_score(path, force=False, retry_quarantined=False, expected=None):
    st = cache_state(path, expected)
    if force or st in ("invalid", "stale"):
        return True
    return st == "quarantined" and retry_quarantined


# ------------------------------------------------------------------ scoring
@torch.no_grad()
def grid_sums(model, views, stats, cfg, device, bs=64):
    """(tie sums, non-tie sums) over GRID of the floor score of the scored test targets, n, n_tie; masks and tie rule
    of coarse_tpp.train.evaluate."""
    model.eval()
    delta = float(cfg["delta"])
    sp = views.coarse
    tie_s, non_s = np.zeros(len(GRID)), np.zeros(len(GRID))
    n = nt = 0.0
    for i0 in range(0, len(sp), bs):
        idx = np.arange(i0, min(i0 + bs, len(sp)))
        b = D.make_batch(sp, idx, device)
        m = (b.tmask * score_mask(views, idx, b.iet.shape[1], device)).double()
        tau = b.iet[:, 1:].double()
        tie = Ro.tie_mask(tau, delta).double() * m
        law, _ = model(b, stats)
        for j, eps in enumerate(GRID):
            v = lik.event_loglik(law, tau, "floor", eps=eps)
            v = torch.where(m > 0, v, torch.zeros_like(v))
            tie_s[j] += float((v * tie).sum()); non_s[j] += float((v * (m - tie)).sum())
        n += float(m.sum()); nt += float(tie.sum())
    return tie_s, non_s, int(round(n)), int(round(nt))


def score_item(ds, enc, seed, sel, device, cache):
    if ds not in cache:
        cache.clear()
        cache[ds] = (load(ds, "train"), load(ds, "test"))
    tr, te = cache[ds]
    rec = dict(version=VERSION, grid=list(GRID), dataset=ds, encoder=enc, seed=seed, device=str(device), models={},
               quarantined=False)
    i6 = grid_index(1e-6)
    for arm in ARMS:
        res = sel[(ds, enc)][arm][seed]
        cfg = res["config"]
        stats = coarse_stats(tr, float(cfg["delta"]))
        model = build(cfg, stats, tr.coarse.n_types).to(device)
        ckpt = Path(cfg["out_dir"]) / "best_C.pt"
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        tie_s, non_s, n, nt = grid_sums(model, te, stats, cfg, device)
        st = res["by_selection"]["C"]["test"]
        e6 = (tie_s[i6] + non_s[i6]) / n
        ok = (np.isfinite(tie_s).all() and np.isfinite(non_s).all() and abs(e6 - st["floor_1e-6"]) <= REPRO_TOL
              and n == st["n_events"] and nt == st["n_ties"])
        rec["models"][arm] = dict(cell_id=cfg["cell_id"], lr_tag=cfg["lr_tag"], ckpt_sha256=ckpt_sha(ckpt),
                                  tie=tie_s.tolist(), non=non_s.tolist(), n=n, n_tie=nt, floor_1e6=e6,
                                  stored=st["floor_1e-6"], abs_diff=abs(e6 - st["floor_1e-6"]), C=st["C"], ok=bool(ok))
        rec["quarantined"] |= not ok
    return rec


def cmd_score(run, out, shard="0/1", limit=0, cpu=False, threads=4, force=False, retry_quarantined=False):
    torch.set_num_threads(int(threads))
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    sel = selected(run)
    items = work_items(sel)
    i, n = (int(x) for x in shard.split("/"))
    todo = [it for k, it in enumerate(items) if k % n == i]
    if limit:
        todo = todo[:limit]
    (out / "cells").mkdir(parents=True, exist_ok=True)
    cache = {}
    for ds, enc, seed in todo:
        dst = out / "cells" / f"{item_name(ds, enc, seed)}.json"
        exp = expected_models(sel, ds, enc, seed)
        if not should_score(dst, force, retry_quarantined, exp):
            if cache_state(dst, exp) == "quarantined":
                print(f"{dst.name}: quarantined cache, not re-scored (use --retry-quarantined or --force)", flush=True)
            continue
        t0 = time.time()
        rec = score_item(ds, enc, seed, sel, device, cache)
        rec["wall_sec"] = round(time.time() - t0, 1)
        tmp = dst.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(rec)); os.replace(tmp, dst)
        print(f"{dst.stem}: max|floor 1e-6 - stored| {max(m['abs_diff'] for m in rec['models'].values()):.2e} "
              f"quarantined={rec['quarantined']} ({rec['wall_sec']}s)", flush=True)


# ------------------------------------------------------------------ report
def cmd_report(run, out):
    sel = selected(run)
    L = ["# Floor score on a continuous grid of floors (exploratory; no training)", "",
         f"Grid: {len(GRID)} floors from {GRID[0]:.0e} to {GRID[-1]:.0e}, 0.25 decade apart. Seed means of the floor "
         "score of the Table 4 models; C from the stored results.", "",
         "| data set | enc | ties | seeds | quarantined | changes (a) | share top = top C (b) | share top = nearest trained (c) "
         "| tau min / median / max (d) | top at 1e-6 = Table 4 |", "|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    rows, curve_rows, missing = [], [], []
    for u in sorted(sel):
        try:
            seeds = unit_seeds(sel[u])
        except ValueError as exc:
            missing.append(f"{u[0]}/{u[1]}: {exc}")
            continue
        ts = tie_share(sel[u])
        good, n_q, n_stale = [], 0, 0
        for seed in seeds:
            path = out / "cells" / f"{item_name(*u, seed)}.json"
            st = cache_state(path, expected_models(sel, *u, seed))
            if st == "ok":
                good.append(json.loads(path.read_text()))
            n_q += st == "quarantined"
            n_stale += st == "stale"
        if not seeds or len(good) < len(seeds):
            missing.append(f"{u[0]}/{u[1]} ({len(good)} of {len(seeds)} seeds; stale {n_stale})")
            L.append(f"| {u[0]} | {u[1]} | {ts:.1%} | {len(good)} | {n_q} | pending | | | | |")
            continue
        test = lambda arm, r, k: sel[u][arm][r["seed"]]["by_selection"]["C"]["test"][k]
        c_means = {a: float(np.mean([test(a, r, "C") for r in good])) for a in ARMS}
        f6_stored = {a: float(np.mean([test(a, r, "floor_1e-6") for r in good])) for a in ARMS}
        S, met = unit_curve([r["models"] for r in good], c_means)
        same = met["top_at_1e6"] == top(f6_stored)
        if not same:
            missing.append(f"{u[0]}/{u[1]}: the top at 1e-6 does not reproduce Table 4")
            L.append(f"| {u[0]} | {u[1]} | {ts:.1%} | {len(good)} | {n_q} | pending | | | | NO |")
            continue
        L.append(f"| {u[0]} | {u[1]} | {ts:.1%} | {len(good)} | {n_q} | {met['changes']} | {met['share_top_eq_c']:.2f} | "
                 f"{met['share_top_nearest_trained']:.2f} | {met['tau_min']:+.2f} / {met['tau_median']:+.2f} / "
                 f"{met['tau_max']:+.2f} | yes |")
        rows.append(dict(dataset=u[0], encoder=u[1], tie_share=ts, **met, **structure(S, met["tops"]),
                         table4_top_reproduced=same))
        for i, g in enumerate(GRID):
            for arm in ARMS:
                curve_rows.append(f"{u[0]},{u[1]},{g:.6e},{arm},{S[arm][i]:.10f}")
    tied = [r for r in rows if r["tie_share"] >= TIE_MIN]
    lost = [r for r in tied if r["loses_next_to_1e6"]]
    defs = [d for r in lost for d in (r["deficit_below_1e6"], r["deficit_above_1e6"])]
    L += ["", "## Where the floor-trained models are first", "",
          "| data set | enc | ties | first at own floor (1e-10 / 1e-6 / 1e-5) | other floors with a floor-trained top | "
          "deficit of the 1e-6 model at 10^-6.25 / 10^-5.75 | max spread over the grid |", "|---|---|---:|---|---:|---|---:|"]
    yn = lambda x: "yes" if x else "no"
    for r in rows:
        L.append(f"| {r['dataset']} | {r['encoder']} | {r['tie_share']:.1%} | "
                 f"{' / '.join(yn(r['own_top'][a]) for a in TRAINED_FLOORS)} | {r['n_other_floor_top']} | "
                 f"{r['deficit_below_1e6']:.3f} / {r['deficit_above_1e6']:.3f} | {r['max_spread']:.4f} |")
    L += ["", f"Units with test ties >= {TIE_MIN:.0%}: {len(tied)}. At least {MIN_CHANGES} changes of the top treatment: "
          f"{sum(r['changes'] >= MIN_CHANGES for r in tied)}. Each floor-trained model first at its own training floor: "
          f"{sum(r['all_own_top'] for r in tied)}. No floor-trained top at any other grid floor: "
          f"{sum(r['n_other_floor_top'] == 0 for r in tied)}. The 1e-6 model loses first place a quarter decade on both "
          f"sides of 1e-6: {len(lost)}" + (f" (deficit {min(defs):.2f} to {max(defs):.2f} nats per event)" if defs else "")
          + f". No grid floor selects the top by C: {sum(r['share_top_eq_c'] == 0 for r in tied)}."]
    if missing:
        L += ["", "Pending: " + "; ".join(missing)]
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "curves.csv").write_text("dataset,encoder,eps,arm,mean_floor_score\n" + "\n".join(curve_rows) + "\n")
    (out / "summary.json").write_text(json.dumps(dict(grid=list(GRID), units=rows, complete=not missing,
                                                      pending=missing), indent=1))
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("score", "report"))
    ap.add_argument("run", help="rank_change run directory")
    ap.add_argument("--out", default=None, help="output directory (default RUN_DIR/floor_grid)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="re-score every item, ignoring cached files")
    ap.add_argument("--retry-quarantined", action="store_true", help="re-score items whose cached file is quarantined")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else Path(a.run) / "floor_grid"
    if a.cmd == "score":
        cmd_score(Path(a.run), out, a.shard, a.limit, a.cpu, a.threads, a.force, a.retry_quarantined)
    else:
        print("\n".join(cmd_report(Path(a.run), out)))


if __name__ == "__main__":
    main()
