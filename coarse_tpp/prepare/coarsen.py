"""Real data with few ties coarsened on purpose (S3, "Coarsened real data"): absolute times floored to a grid
delta = r x (median positive train gap), r in R. The original time of every event is kept as time_since_start_fine,
so the latent-time score B can be computed (coarse_tpp.latent_score).

    python -m coarse_tpp.prepare.coarsen [--data DATA_DIR]

writes DATA_DIR/coarsened_<base>_r<code>/{train,dev,test}.pkl and resolution.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import data as D
from ..experiments import COARSEN_BASES, COARSEN_R, coarsened_name
from .common import FLOOR, SPLITS, items, read, write_resolution, write_split


def median_gap(src: Path):
    d = read(src, "train")
    g = np.concatenate([np.diff([e["time_since_start"] for e in s]) for _, s in items(d, "train")[0] if len(s) > 1])
    return float(np.median(g[g > FLOOR]))


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    a = ap.parse_args(argv)
    root = Path(a.data) if a.data else D.data_dir()
    for b in COARSEN_BASES:
        for r in COARSEN_R:
            print(json.dumps(coarsen(root, b, r)))


if __name__ == "__main__":
    main()
