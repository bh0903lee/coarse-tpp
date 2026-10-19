"""NHP-style pickles (Mei and Eisner 2017; also the RMTPP-preprocessed StackOverflow, BookOrder and MIMIC-II folds):
one file per split, {split: [sequence], "dim_process": n}, events {time_since_start, time_since_last_event,
type_event}. This is the layout coarse_tpp.data reads, so the files are used as distributed; this script copies them
into DATA_DIR/<name>/ and stores zero gaps as 1e-6 (first event included).

    python -m coarse_tpp.prepare.nhp_pickle --src RAW/conttime --name conttime
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .. import data as D
from .common import SPLITS, floor_zero_gaps, items, read, write_split


def convert(src: Path, name: str, root: Path):
    out = {}
    for sp in SPLITS:
        d = read(src, sp)
        its, is_dict = items(d, sp)
        seqs = [(k, floor_zero_gaps(s)) for k, s in its]
        write_split(root / name, sp, {sp: ({k: v for k, v in seqs} if is_dict else [v for _, v in seqs]),
                                      "dim_process": d.get("dim_process", 1)}, overwrite=True)
        out[sp] = len(seqs)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default=None)
    a = ap.parse_args(argv)
    root = Path(a.data) if a.data else D.data_dir()
    print(a.name, convert(Path(a.src), a.name, root))


if __name__ == "__main__":
    main()
