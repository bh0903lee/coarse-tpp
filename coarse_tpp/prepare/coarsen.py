"""Real data with few ties coarsened on purpose (Supplementary S4, "Coarsened real data" and "Smoothing objectives"): absolute
times floored to a grid delta = r x (median positive train gap), r in R. The original time of every event is kept as
time_since_start_fine, so the latent-time score B can be computed (coarse_tpp.latent_score).

    python -m coarse_tpp.prepare.coarsen [--data DATA_DIR]           ContTime, Taxi, StackOverflow (coarsened_real)
    python -m coarse_tpp.prepare.coarsen --more [--data DATA_DIR]    Earthquake, Taobao (coarsened_more)

writes DATA_DIR/coarsened_<base>_r<code>/{train,dev,test}.pkl and resolution.json.

Storage grain (--more; a rule fixed before those data were generated): the original times must be finer than the
grid. The grain c of a base is the largest c in {1, 2, 5} x 10^e such that at least TICK_SHARE of the positive train
gaps lie within TICK_TOL c of a positive multiple of c (None when no such c exists: the times are treated as
continuous). A level r is written only when delta >= TICK_FACTOR x c; coarse_tpp.experiments.COARSEN_MORE_LEVELS
lists the levels that survive this rule for the two bases of the paper, and the command refuses to write a level that
the rule drops.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import data as D
from ..experiments import COARSEN_BASES, COARSEN_MORE_LEVELS, COARSEN_R, coarsened_name
from .common import FLOOR, SPLITS, items, read, write_resolution, write_split

TICK_SHARE, TICK_TOL, TICK_FACTOR = 0.8, 0.05, 10.0


def positive_train_gaps(src: Path):
    d = read(src, "train")
    g = np.concatenate([np.diff([e["time_since_start"] for e in s]) for _, s in items(d, "train")[0] if len(s) > 1])
    return g[g > FLOOR]


def median_gap(src: Path):
    return float(np.median(positive_train_gaps(src)))


def storage_tick(pos: np.ndarray):
    """Largest c in {1, 2, 5} x 10^e (e in -9..3) with at least TICK_SHARE of the gaps within TICK_TOL c of a positive
    multiple of c; None when there is none."""
    best = None
    for e in range(-9, 4):
        for m in (1, 2, 5):
            c = m * 10.0 ** e
            x = pos / c
            k = np.round(x)
            if np.mean((np.abs(x - k) <= TICK_TOL) & (k >= 1)) >= TICK_SHARE:
                best = c
    return best


def coarsen(root: Path, base: str, r: float):
    delta = r * median_gap(root / base)
    ds = coarsened_name(base, r)
    ties, n = 0, 0
    for sp in SPLITS:
        d = read(root / base, sp)
        its, is_dict = items(d, sp)
        new = []
        for k, s in its:
            t = np.array([e["time_since_start"] for e in s], dtype=np.float64)
            q = np.floor(t / delta) * delta
            recs, prev = [], 0.0
            for i, (e, qt) in enumerate(zip(s, q)):
                g = qt - prev
                if i > 0 and sp == "train":
                    n += 1; ties += g <= FLOOR
                recs.append({"time_since_start": float(qt), "time_since_last_event": float(max(g, FLOOR)),
                             "type_event": int(e["type_event"]), "time_since_start_fine": float(t[i])})
                prev = qt
            new.append((k, recs))
        write_split(root / ds, sp, {sp: ({k: v for k, v in new} if is_dict else [v for _, v in new]),
                                    "dim_process": d.get("dim_process", 1)})
    write_resolution(root / ds, delta, f"absolute times of {base} floored to r = {r:g} x median train gap")
    return dict(dataset=ds, base=base, r=r, delta=delta, train_tie_share=ties / max(n, 1))


def coarsen_more(root: Path, base: str, levels):
    """Coarsen `base` at the given levels after the storage-grain rule; returns the rows written and the grain."""
    pos = positive_train_gaps(root / base)
    med, c = float(np.median(pos)), storage_tick(pos)
    rows = []
    for r in levels:
        delta = r * med
        if c is not None and delta < TICK_FACTOR * c:
            raise SystemExit(f"{base} r={r:g}: delta {delta:.3g} is below {TICK_FACTOR:g} x the storage grain {c:g}; "
                             "this level is dropped by the fixed rule and is not written")
        rows.append(dict(coarsen(root, base, r), storage_tick=c))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    ap.add_argument("--more", action="store_true", help="Earthquake and Taobao at the levels of COARSEN_MORE_LEVELS")
    a = ap.parse_args(argv)
    root = Path(a.data) if a.data else D.data_dir()
    if a.more:
        for b, levels in COARSEN_MORE_LEVELS.items():
            for row in coarsen_more(root, b, levels):
                print(json.dumps(row))
        return
    for b in COARSEN_BASES:
        for r in COARSEN_R:
            print(json.dumps(coarsen(root, b, r)))


if __name__ == "__main__":
    main()
