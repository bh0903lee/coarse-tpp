"""Validation selection among the candidates of a rank_change run (paper Section 5.3; Online Resource 1, Table S6,
middle part, and Section S2.11). No training and no re-scoring: reads the saved results only.

Per combination (data set, encoder), the candidates are the (treatment, learning rate) pairs of the grid, 16 for the
eight treatments and two learning rates of `rank_change`; each candidate is the mean over its seeds at the checkpoint
selected by the validation interval score (best_C.pt). A score S (a floor or C) chooses the candidate with the largest
mean validation S. Reported for each floor: the treatment it chooses, the rank of its choice among the candidates by
mean test C, and the deficit, mean test C of the interval score's choice minus that of the floor's choice, paired by
seed, with a two-sided 95% t interval.

    python scripts/validation_selection.py runs/rank_change --audit results/audit/audit.json

writes `validation_selection/report.md` and `summary.json` in the run directory (or in --out).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from . import analysis as A
from . import experiments as X

FLOORS = ("floor_1e-10", "floor_1e-6", "floor_1e-5")
SCORES = FLOORS + ("C",)


def t_interval(d, level=0.95):
    d = np.asarray(d, float)
    m = float(d.mean())
    if len(d) < 2:
        return m, float("nan"), float("nan")
    h = float(stats.t.ppf(0.5 + level / 2, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d)))
    return m, m - h, m + h


def candidates(cells):
    """{(dataset, encoder): {(arm, lr_tag): {seed: result}}} for the mixture-head cells selected by dev C."""
    G = {}
    for r in cells:
        if r.get("primary_selection", "C") != "C":
            raise ValueError(f"{r['cell_id']}: dev and test scores must be those of the dev-C checkpoint")
        c = r["config"]
        G.setdefault((c["dataset"], c["encoder"]["type"]), {}).setdefault((r["arm"], c["lr_tag"]), {})[r["seed"]] = r
    return G


def mean(seeds, split, key):
    return float(np.mean([seeds[s][split][key] for s in sorted(seeds)]))


def select(cand, score):
    return max(cand, key=lambda k: mean(cand[k], "dev", score))


def summarize(G, audit, controls=X.RANK_CONTROLS, tied_min=0.01, arms=X.RANK_ARMS):
    rows = []
    for g in sorted(G):
        cand = G[g]
        if {a for a, _ in cand} != set(arms):
            continue
        test_c = {k: mean(v, "test", "C") for k, v in cand.items()}
        order = sorted(cand, key=lambda k: -test_c[k])
        pick_c = select(cand, "C")
        tie = float((audit.get(g[0]) or {}).get("tie_share", float("nan")))
        row = dict(dataset=g[0], encoder=g[1], tie_share=tie, tied=bool(tie >= tied_min and g[0] not in controls),
                   n_candidates=len(cand), pick_C=list(pick_c), pick_C_test_C=test_c[pick_c])
        for f in FLOORS:
            p = select(cand, f)
            seeds = sorted(set(cand[p]) & set(cand[pick_c]))
            d = [cand[pick_c][s]["test"]["C"] - cand[p][s]["test"]["C"] for s in seeds]
            m, lo, hi = t_interval(d)
            row[f] = dict(pick=list(p), self_choice=p[0] == f, rank=order.index(p) + 1, test_C=test_c[p],
                          deficit=m, lo95=lo, hi95=hi, seeds=len(seeds), seeds_positive=int(sum(x > 0 for x in d)))
        rows.append(row)
    return rows


def report(run, audit_path=None, out=None, floor="floor_1e-6"):
    run = Path(run)
    cells = A.load_cells(run)
    audit = A.audit_rows(audit_path)
    if not audit:
        print(f"warning: no audit file at {audit_path or X.DEFAULT_AUDIT}; no combination is marked tied", file=sys.stderr)
    rows = summarize(candidates(cells), audit)
    T = [r for r in rows if r["tied"]]
    L = ["# Validation selection among the candidates (no training)", "",
         "Candidates: (treatment, learning rate), means over seeds at best_C.pt. A score chooses the candidate with the "
         "largest mean validation score. Rank: rank of a floor's choice among the candidates by mean test C. Deficit: "
         "test C of the interval score's choice minus that of the floor's choice, seed-paired, two-sided 95% t interval.",
         "", f"Combinations: {len(rows)}; tied (>= 1% ties, controls excluded): {len(T)}.", "",
         "| data set | encoder | tied | C chooses | " + " | ".join(f"{f} chooses | rank | deficit [95%]" for f in FLOORS) + " |",
         "|---|---|---|---|" + "---|---:|---|" * len(FLOORS)]
    for r in rows:
        txt = []
        for f in FLOORS:
            x = r[f]
            txt += [f"{x['pick'][0]} {x['pick'][1]}", f"{x['rank']}/{r['n_candidates']}",
                    f"{x['deficit']:+.4f} [{x['lo95']:+.4f}, {x['hi95']:+.4f}]"]
        L.append(f"| {r['dataset']} | {r['encoder']} | {'yes' if r['tied'] else 'no'} | {r['pick_C'][0]} {r['pick_C'][1]} | "
                 + " | ".join(txt) + " |")
    L += ["", "## Tied combinations", ""]
    for f in FLOORS:
        L.append(f"- {f}: chooses the model trained with it on {sum(r[f]['self_choice'] for r in T)}/{len(T)}")
    L.append(f"- C chooses a treatment other than each floor's choice on "
             f"{sum(all(r['pick_C'][0] != r[f]['pick'][0] for f in FLOORS) for r in T)}/{len(T)}")
    if T:
        x = [r[floor] for r in T]
        dm = [v["deficit"] for v in x]
        L += [f"- {floor}: rank 9 or lower on {sum(v['rank'] >= 9 for v in x)}/{len(T)}; trails the interval score's "
              f"choice on {sum(d > 0 for d in dm)}/{len(T)}, by up to {max(dm):.4f} (median {float(np.median(dm)):.4f}); "
              f"95% interval above zero on {sum(v['lo95'] > 0 for v in x)}/{len(T)}"]
    od = Path(out) if out else run / "validation_selection"
    od.mkdir(parents=True, exist_ok=True)
    (od / "report.md").write_text("\n".join(L) + "\n")
    (od / "summary.json").write_text(json.dumps(rows, indent=1))
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--audit", default=None)
    ap.add_argument("--out", default=None, help="output directory (default: RUN/validation_selection)")
    a = ap.parse_args(argv)
    print("\n".join(report(a.run, a.audit, a.out)))


if __name__ == "__main__":
    main()
