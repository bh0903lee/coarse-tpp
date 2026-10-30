"""Joint rankings of time and mark: rankings of the training treatments of a marks run by the joint floor score and
the joint interval score (Supplementary S4, "Marks", Table S13 and Fig. S6). Exploratory; no training and no
re-scoring.

    python -m coarse_tpp.joint_rankings RUN_DIR [--out DIR]

RUN_DIR is a marks run: cells with a categorical mark head whose checkpoint was selected by the dev joint score
J = C + mark (coarse_tpp.train, selection J). The control trained on marks alone (arm mark_only) is not a time
treatment and is left out. Learning rate per (unit, treatment): the best dev-J mean over seeds
(coarse_tpp.analysis.select_lr(cells, "J")). Unit = (data set, encoder). All values are the stored test means of
by_selection.J (seed means):
  joint floor     floor_1e-6 + mark   (floor 1e-6 on the raw density, plus the mark log-probability)
  joint interval  J = C + mark
  time floor      floor_1e-6;  time interval  C   (time-only rankings of the same checkpoints)
Per unit and for the joint and the time scores: the top treatment by the floor and by the interval score, Kendall's
tau of the two rankings, whether the floor's top is the model trained with eps = 1e-6, the rank of the floor's top
among the treatments by the interval score and its deficit (top interval score minus its interval score, nats per
event). Exact score ties are broken by the treatment name. summary.json holds the seed-mean scores of every
treatment (the ranks of Fig. S6).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import analysis as A
from . import experiments as X

TIE_MIN = 0.01
SELF = "floor_1e-6"
EXCLUDED = ("mark_only",)
SCORES = {"joint_floor": lambda t: t["floor_1e-6"] + t["mark"], "joint_interval": lambda t: t["J"],
          "time_floor": lambda t: t["floor_1e-6"], "time_interval": lambda t: t["C"]}


def kendall(x, y) -> float:
    from scipy import stats
    return float(stats.kendalltau(x, y)[0])


def top(scores: dict) -> str:
    """Treatment with the highest score; exact ties broken by the treatment name."""
    return max(sorted(scores), key=lambda a: scores[a])


def unit_scores(arms):
    """{score name: {arm: mean over seeds}} of one unit; arms {arm: {seed: result}} at the chosen lr."""
    mean = lambda arm, f: float(np.mean([f(r["by_selection"]["J"]["test"]) for r in arms[arm].values()]))
    return {k: {a: mean(a, f) for a in arms} for k, f in SCORES.items()}


def unit_row(u, arms):
    sc = unit_scores(arms)
    ts = {round(r["by_selection"]["J"]["test"]["tie_share"], 12) for s in arms.values() for r in s.values()}
    if len(ts) != 1:
        raise SystemExit(f"{u}: test tie share differs across cells {ts}")
    names = sorted(arms)
    row = dict(dataset=u[0], encoder=u[1], tie_share=ts.pop(), n_arms=len(names),
               seeds=sorted(set.intersection(*(set(arms[a]) for a in names))))
    for kind in ("joint", "time"):
        f, i = sc[f"{kind}_floor"], sc[f"{kind}_interval"]
        tf, ti = top(f), top(i)
        order = sorted(names, key=lambda a: -i[a])
        row.update({f"{kind}_top_floor": tf, f"{kind}_top_interval": ti, f"{kind}_changed": tf != ti,
                    f"{kind}_tau": kendall([f[a] for a in names], [i[a] for a in names]),
                    f"{kind}_floor_self": tf == SELF, f"{kind}_rank_of_floor_top": order.index(tf) + 1,
                    f"{kind}_deficit_of_floor_top": i[ti] - i[tf]})
    row["scores"] = sc
    return row


def summary_counts(rows):
    """Counts over the units with test ties >= TIE_MIN (the statements of Supplementary S4)."""
    tied = [r for r in rows if r["tie_share"] >= TIE_MIN]
    d = [r["joint_deficit_of_floor_top"] for r in tied]
    return dict(n_tied=len(tied), n_joint_changed=sum(r["joint_changed"] for r in tied),
                n_time_changed=sum(r["time_changed"] for r in tied),
                n_joint_floor_self=sum(r["joint_floor_self"] for r in tied),
                joint_rank_of_floor_top=sorted({r["joint_rank_of_floor_top"] for r in tied}),
                joint_deficit_range=[min(d), max(d)] if d else None)


def cmd_report(run, out):
    cells, st = A.load_cells(Path(run), with_status=True)
    cells = [c for c in cells if c["arm"] not in EXCLUDED]
    for c in cells:
        if "J" not in c["by_selection"]:
            raise SystemExit(f"{c['config']['cell_id']}: no joint selection J (needs a mark head)")
    units = {}
    for (ds, enc, head, K, arm), seeds in A.select_lr(cells, "J").items():
        units.setdefault((ds, enc), {})[arm] = seeds
    rows = [unit_row(u, units[u]) for u in sorted(units)]
    arms = set(X.MARK_ARMS)
    complete = bool(st["finished"]) and all(set(units[(r["dataset"], r["encoder"])]) == arms and
                                            set(r["seeds"]) == set(X.REAL_SEEDS) for r in rows)
    cnt = summary_counts(rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(rows=rows, counts=cnt, complete=complete, n_cells=len(cells),
                                                      failed=st["failed"]), indent=1))
    L = ["# Joint rankings of time and mark, floor 1e-6 vs interval score (exploratory; no training)", "",
         f"Cells used (time treatments, after the complete-seed restriction): {len(cells)}; failed {len(st['failed'])}; "
         f"complete: {complete}. Learning rate and checkpoint by dev J = C + mark; seed means of the test scores.", "",
         "| data set | enc | ties | arms | joint top: floor / interval | joint tau | joint floor top = floor_1e-6 | "
         "rank of joint floor top by joint interval | deficit | time top: floor / interval | time tau |",
         "|---|---|---:|---:|---|---:|---|---:|---:|---|---:|"]
    for r in rows:
        L.append(f"| {r['dataset']} | {r['encoder']} | {r['tie_share']:.1%} | {r['n_arms']} | {r['joint_top_floor']} / "
                 f"{r['joint_top_interval']} | {r['joint_tau']:+.2f} | {'yes' if r['joint_floor_self'] else 'no'} | "
                 f"{r['joint_rank_of_floor_top']} of {r['n_arms']} | {r['joint_deficit_of_floor_top']:.3f} | "
                 f"{r['time_top_floor']} / {r['time_top_interval']} | {r['time_tau']:+.2f} |")
    rng = cnt["joint_deficit_range"]
    L += ["", f"Units with test ties >= {TIE_MIN:.0%}: {cnt['n_tied']}. Joint top changes between the floor and the "
          f"interval score in {cnt['n_joint_changed']}, the time-only top in {cnt['n_time_changed']}; the joint floor "
          f"ranks the model trained with eps = 1e-6 first in {cnt['n_joint_floor_self']}; by the joint interval score "
          f"the joint floor's top ranks {' or '.join(map(str, cnt['joint_rank_of_floor_top']))}"
          + (f", {rng[0]:.3f} to {rng[1]:.3f} nats per event below the top." if rng else ".")]
    (out / "report.md").write_text("\n".join(L) + "\n")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="marks run directory")
    ap.add_argument("--out", default=None, help="output directory (default RUN_DIR/joint_rankings)")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else Path(a.run) / "joint_rankings"
    print("\n".join(cmd_report(Path(a.run), out)))


if __name__ == "__main__":
    main()
