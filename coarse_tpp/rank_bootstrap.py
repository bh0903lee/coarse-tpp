"""Bootstrap of the rankings of Table 4: how often the top treatment by the floor score (floor 1e-6) and by the
interval score C differ under resampling of test sequences and training seeds (Supplementary S4, "Resampling",
Table S7). Exploratory; no training and no re-scoring.

    python -m coarse_tpp.rank_bootstrap run RUN_DIR [--B 2000] [--out DIR]
    python -m coarse_tpp.rank_bootstrap check RUN_DIR       validity checks only, no outcome computed

RUN_DIR is a rank_change run. Models: those of Table 4 (coarse_tpp.floor_grid.selected: dev-C learning rate,
checkpoint best_C.pt, all eight treatments, seeds common to all treatments). Inputs: the per-sequence test sums
perseq_C.npz that coarse_tpp.train writes for every cell (n = scored targets, C, floor_1e-6). A mean per-event score
is (sum over sequences) / (number of scored targets), averaged over seeds, as in Table 4.

Validity per cell: sum(n) equals the stored test n_events, and sum(C) / n and sum(floor_1e-6) / n reproduce the stored
test C and floor_1e-6 within CHECK_TOL; the scored-target counts per sequence are identical across the cells of a unit
(same test sequences); the tops on the full sample equal the tops of the stored test means. A unit failing a check is
listed and left out.

Bootstrap (B draws, generator numpy default_rng(SEED), one generator per unit), two stages per draw: (1) the test
sequences are resampled with replacement, the same draw for every treatment and seed; (2) the seeds are resampled
with replacement, the same draw for every treatment (seeds are paired across treatments). The learning rate is not
re-selected. Per unit:
  p_diff  share of draws in which the top by floor_1e-6 differs from the top by C
  p_self  share of draws in which the top by floor_1e-6 is the model trained with eps = 1e-6
  tau     Kendall's tau of the two rankings on the full sample; 2.5% / 50% / 97.5% quantiles over the draws
The top treatment is chosen on the test means; the bootstrap does not remove this selection effect.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .floor_grid import ARMS, TIE_MIN, kendall, selected, tie_share, top, unit_seeds

SEED = 20261027
B_DEFAULT = 2000
CHECK_TOL = 1e-9
P_MIN = 0.9
SELF = "floor_1e-6"


# ------------------------------------------------------------------ data
def load_unit(arms):
    """Arrays of one unit: n (M,) scored targets per test sequence, C and F6 {arm: (S, M)} per-sequence sums, seeds.
    Raises ValueError when a validity check fails."""
    seeds = unit_seeds(arms)
    if not seeds:
        raise ValueError("no seed common to all treatments")
    C, F6, n_ref = {}, {}, None
    for arm in ARMS:
        cs, fs = [], []
        for s in seeds:
            res = arms[arm][s]
            cid = res["config"]["cell_id"]
            with np.load(Path(res["config"]["out_dir"]) / "perseq_C.npz") as z:
                n, c, f6 = z["n"].astype(float), z["C"].astype(float), z["floor_1e-6"].astype(float)
            if (n.ndim != 1 or c.shape != n.shape or f6.shape != n.shape or
                    not all(np.isfinite(x).all() for x in (n, c, f6)) or
                    np.any(n < 0) or np.any(n != np.floor(n)) or n.sum() <= 0):
                raise ValueError(f"{cid}: invalid per-sequence arrays")
            st = res["by_selection"]["C"]["test"]
            if int(round(n.sum())) != st["n_events"]:
                raise ValueError(f"{cid}: per-sequence target count does not reproduce the stored count")
            for name, x, ref in (("C", c, st["C"]), ("floor_1e-6", f6, st["floor_1e-6"])):
                if not np.isfinite(ref) or abs(x.sum() / n.sum() - ref) > CHECK_TOL:
                    raise ValueError(f"{cid}: per-sequence {name} does not reproduce the stored score")
            if n_ref is None:
                n_ref = n
            elif n.shape != n_ref.shape or not np.array_equal(n, n_ref):
                raise ValueError(f"{cid}: scored targets per sequence differ within the unit")
            cs.append(c); fs.append(f6)
        C[arm], F6[arm] = np.stack(cs), np.stack(fs)
    return n_ref, C, F6, seeds


# ------------------------------------------------------------------ bootstrap (pure; tests/test_rank_bootstrap.py)
def scores(n, X, w, seed_idx):
    """{arm: mean over the drawn seeds of (sum_m w_m X[arm][s, m]) / (sum_m w_m n_m)}; w are sequence counts."""
    den = float(n @ w)
    if not np.isfinite(den) or den <= 0:
        raise ValueError("resampled targets must have a positive finite total")
    return {a: float(np.mean((X[a] @ w)[seed_idx]) / den) for a in X}


def bootstrap(n, C, F6, B=B_DEFAULT, seed=SEED):
    if B <= 0:
        raise ValueError("B must be positive")
    rng = np.random.default_rng(seed)
    M, S = len(n), next(iter(C.values())).shape[0]
    diff = selfp = 0
    taus = np.empty(B)
    for b in range(B):
        w = np.bincount(rng.integers(0, M, M), minlength=M).astype(float)
        si = rng.integers(0, S, S)
        sc, sf = scores(n, C, w, si), scores(n, F6, w, si)
        tc, tf = top(sc), top(sf)
        diff += tc != tf
        selfp += tf == SELF
        taus[b] = kendall([sf[a] for a in ARMS], [sc[a] for a in ARMS])
    valid = np.isfinite(taus)
    q = [float(x) for x in np.quantile(taus[valid], [0.025, 0.5, 0.975])] if valid.any() else [None] * 3
    return dict(p_diff=diff / B, p_self=selfp / B, tau_q025=q[0], tau_q50=q[1], tau_q975=q[2],
                tau_n_undefined=int((~valid).sum()))


def original(n, C, F6):
    w, si = np.ones(len(n)), np.arange(next(iter(C.values())).shape[0])
    sc, sf = scores(n, C, w, si), scores(n, F6, w, si)
    tau = kendall([sf[a] for a in ARMS], [sc[a] for a in ARMS])
    return dict(top_C=top(sc), top_floor=top(sf), tau=tau if np.isfinite(tau) else None)


def summary_counts(rows):
    """Counts over the units with test ties >= TIE_MIN (the statements of Supplementary S4)."""
    tied = [r for r in rows if r["tie_share"] >= TIE_MIN]
    changed = [r for r in tied if r["top_C"] != r["top_floor"]]
    return dict(n_tied=len(tied), n_changed=len(changed), n_changed_p_diff_ge=sum(r["p_diff"] >= P_MIN for r in changed),
                min_p_diff_changed=min((r["p_diff"] for r in changed), default=None),
                n_changed_p_diff_1=sum(r["p_diff"] == 1.0 for r in changed),
                n_tied_p_self_ge=sum(r["p_self"] >= P_MIN for r in tied),
                min_p_self_tied=min((r["p_self"] for r in tied), default=None))


# ------------------------------------------------------------------ commands
def stored_top(arms, seeds, key):
    return top({a: float(np.mean([arms[a][s]["by_selection"]["C"]["test"][key] for s in seeds])) for a in ARMS})


def cmd_run(run, out, B=B_DEFAULT):
    sel = selected(run)
    rows, failed = [], []
    for u in sorted(sel):
        try:
            n, C, F6, seeds = load_unit(sel[u])
        except (ValueError, OSError, KeyError) as e:
            failed.append(f"{u[0]}/{u[1]}: {e}")
            continue
        o = original(n, C, F6)
        sc, sf = stored_top(sel[u], seeds, "C"), stored_top(sel[u], seeds, "floor_1e-6")
        if (o["top_C"], o["top_floor"]) != (sc, sf):
            failed.append(f"{u[0]}/{u[1]}: tops on the full sample {o['top_C']}/{o['top_floor']} != stored {sc}/{sf}")
            continue
        r = dict(dataset=u[0], encoder=u[1], tie_share=tie_share(sel[u]), seeds=seeds, n_sequences=len(n), **o,
                 **bootstrap(n, C, F6, B))
        rows.append(r)
        print(f"{u[0]}/{u[1]}: top C {r['top_C']}, top floor_1e-6 {r['top_floor']}, p_diff {r['p_diff']:.3f}, "
              f"p_self {r['p_self']:.3f}", flush=True)
    cnt = summary_counts(rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "boot.json").write_text(json.dumps(dict(B=B, seed=SEED, rows=rows, failed=failed, counts=cnt,
                                                   complete=not failed), indent=1))
    fq = lambda x: "undefined" if x is None or not np.isfinite(x) else f"{x:+.2f}"
    L = ["# Bootstrap of the rankings of Table 4 (exploratory; no training)", "",
         f"B = {B} draws (seed {SEED}); test sequences and seeds resampled; learning rate not re-selected.", "",
         "| data set | enc | ties | top by C | top by floor 1e-6 | p_diff | p_self | tau (full sample) | tau 2.5% / 50% / 97.5% |",
         "|---|---|---:|---|---|---:|---:|---:|---|"]
    for r in rows:
        L.append(f"| {r['dataset']} | {r['encoder']} | {r['tie_share']:.1%} | {r['top_C']} | {r['top_floor']} | "
                 f"{r['p_diff']:.3f} | {r['p_self']:.3f} | {fq(r['tau'])} | {fq(r['tau_q025'])} / {fq(r['tau_q50'])} / "
                 f"{fq(r['tau_q975'])} |")
    L += ["", "Tau quantiles use defined draws only; undefined counts: " +
          "; ".join(f"{r['dataset']}/{r['encoder']}={r['tau_n_undefined']}" for r in rows)]
    if failed:
        L += ["", "Excluded units (validity checks):"] + [f"- {x}" for x in failed]
    f3 = lambda x: "—" if x is None else f"{x:.3f}"
    L += ["", f"Units with test ties >= {TIE_MIN:.0%}: {cnt['n_tied']}; top changed between the two scores: {cnt['n_changed']}. "
          f"Among these, p_diff >= {P_MIN} in {cnt['n_changed_p_diff_ge']} (minimum {f3(cnt['min_p_diff_changed'])}; "
          f"p_diff = 1 in {cnt['n_changed_p_diff_1']}). p_self >= {P_MIN} in {cnt['n_tied_p_self_ge']} of {cnt['n_tied']} "
          f"(minimum {f3(cnt['min_p_self_tied'])}).",
          "", "The top treatment is chosen on the test means; the bootstrap does not remove this selection effect."]
    (out / "report.md").write_text("\n".join(L) + "\n")
    return L


def cmd_check(run):
    """Validity checks of every unit without computing any outcome."""
    sel = selected(run)
    bad = 0
    for u in sorted(sel):
        try:
            n, C, F6, seeds = load_unit(sel[u])
            print(f"{u[0]}/{u[1]}: ok ({len(seeds)} seeds, {len(n)} test sequences)")
        except (ValueError, OSError, KeyError) as e:
            bad += 1
            print(f"{u[0]}/{u[1]}: FAILED: {e}")
    print(f"validity check: {'FAILED' if bad else 'PASS'}; {len(sel)} units inspected")
    return bad == 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "check"))
    ap.add_argument("run", help="rank_change run directory")
    ap.add_argument("--B", type=int, default=B_DEFAULT, help="number of bootstrap draws")
    ap.add_argument("--out", default=None, help="output directory (default RUN_DIR/rank_bootstrap)")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        if not cmd_check(Path(a.run)):
            raise SystemExit(1)
        return
    out = Path(a.out) if a.out else Path(a.run) / "rank_bootstrap"
    print("\n".join(cmd_run(Path(a.run), out, a.B)))


if __name__ == "__main__":
    main()
