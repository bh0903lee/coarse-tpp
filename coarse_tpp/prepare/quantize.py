"""Quantized ContTime: absolute times floored to a grid of 0.02 or 0.1 (same sequences and splits), so that ties appear
as in data recorded at a finite clock resolution (Section 2).

    python -m coarse_tpp.prepare.quantize [--data DATA_DIR] [--source conttime]

writes DATA_DIR/conttime_q02 and DATA_DIR/conttime_q10 with resolution.json (resolution known by construction).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .. import data as D
from .common import FLOOR, SPLITS, items, read, write_resolution, write_split

GRIDS = {"conttime_q02": 0.02, "conttime_q10": 0.10}


def quantize(src: Path, dst: Path, r: float):
    for sp in SPLITS:
        d = read(src, sp)
        its, is_dict = items(d, sp)
        new = []
        for k, s in its:
            t = np.array([e["time_since_start"] for e in s], dtype=np.float64)
            q = np.floor(t / r) * r
            prev = 0.0
            recs = []
            for e, qt in zip(s, q):
                recs.append({"time_since_start": float(qt), "time_since_last_event": float(max(qt - prev, FLOOR)),
                             "type_event": int(e["type_event"])})
                prev = qt
            new.append((k, recs))
        write_split(dst, sp, {sp: ({k: v for k, v in new} if is_dict else [v for _, v in new]),
                              "dim_process": d.get("dim_process", 1)})
    write_resolution(dst, r, f"absolute times of {src.name} floored to a grid of {r:g}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    ap.add_argument("--source", default="conttime")
    a = ap.parse_args(argv)
    root = Path(a.data) if a.data else D.data_dir()
    for name, r in GRIDS.items():
        quantize(root / a.source, root / name, r)
        print(f"{name}: grid {r}")


if __name__ == "__main__":
    main()
