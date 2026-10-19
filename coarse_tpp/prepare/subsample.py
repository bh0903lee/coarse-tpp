"""The Retweet subsample (900 / 100 / 200 sequences of the NHP Retweet data): train and dev come from the NHP train
split, test from the NHP test split; the sequence list is in coarse_tpp/resources/retweet_subsample.json.

    python -m coarse_tpp.prepare.subsample [--data DATA_DIR]

reads DATA_DIR/retweet and writes DATA_DIR/retweet_sub (marks 0-based).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import data as D
from .common import SPLITS, items, read, write_split

INDEX = Path(__file__).resolve().parents[1] / "resources" / "retweet_subsample.json"


def build(root: Path, index: Path = INDEX):
    spec = json.loads(index.read_text())
    parent = {sp: dict(items(read(root / spec["parent"], sp), sp)[0]) for sp in SPLITS}
    out = {}
    for sp in SPLITS:
        rows = spec["splits"][sp]["sequences"]
        seqs = {k: parent[psp][pk] for k, psp, pk in rows}
        write_split(root / "retweet_sub", sp, {sp: seqs, "dim_process": spec["dim_process"]}, overwrite=True)
        out[sp] = len(seqs)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    a = ap.parse_args(argv)
    print("retweet_sub", build(Path(a.data) if a.data else D.data_dir()))


if __name__ == "__main__":
    main()
