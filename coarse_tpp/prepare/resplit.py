"""New random splits (Section 5.3, S3): the sequences of train, dev and test of each parent data set are pooled and
re-dealt by one fixed permutation into splits of the original sizes. Sequences, recorded times, marks and the
recording grid are unchanged.

    python -m coarse_tpp.prepare.resplit [--data DATA_DIR]

writes DATA_DIR/<parent>_resplit/{train,dev,test}.pkl (and the parent's resolution.json, if any).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from .. import data as D
from ..experiments import RESPLIT_PARENTS
from .common import SPLITS, read, write_split

SEED = 20261014
SUFFIX = "_resplit"


def load_parent(path: Path):
    """({split: [sequence]}, dim_process)."""
    out, dim = {}, None
    for sp in SPLITS:
        d = read(path, sp)
        inner = d[sp]
        out[sp] = list(inner.values()) if isinstance(inner, dict) else list(inner)
        dim = int(d.get("dim_process", dim or 1))
    return out, dim


def resplit(by_split: dict, seed: int = SEED):
    """Pool the splits (train, dev, test order) and deal one permutation into splits of the original sizes."""
    pool = [s for sp in SPLITS for s in by_split[sp]]
    perm = np.random.default_rng(seed).permutation(len(pool))
    out, i = {}, 0
    for sp in SPLITS:
        n = len(by_split[sp])
        out[sp] = [pool[k] for k in perm[i:i + n]]
        i += n
    return out


def seq_key(s):
    return tuple(round(float(e["time_since_start"]), 9) for e in s) + tuple(int(e["type_event"]) for e in s)


def build(root: Path, parent: str, seed: int = SEED):
    by_split, dim = load_parent(root / parent)
    new = resplit(by_split, seed)
    assert sorted(map(seq_key, (s for sp in SPLITS for s in by_split[sp]))) == \
        sorted(map(seq_key, (s for sp in SPLITS for s in new[sp]))), "resplit must preserve the multiset of sequences"
    for sp in SPLITS:
        write_split(root / (parent + SUFFIX), sp, {sp: new[sp], "dim_process": dim}, overwrite=True)
    if (root / parent / "resolution.json").exists():          # a constructed resolution carries over
        shutil.copyfile(root / parent / "resolution.json", root / (parent + SUFFIX) / "resolution.json")
    return {sp: len(new[sp]) for sp in SPLITS}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    a = ap.parse_args(argv)
    root = Path(a.data) if a.data else D.data_dir()
    for p in RESPLIT_PARENTS:
        print(p + SUFFIX, build(root, p))


if __name__ == "__main__":
    main()
