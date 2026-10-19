"""JODIE interaction logs (Kumar, Zhang and Leskovec, KDD 2019; http://snap.stanford.edu/jodie/) as event sequences.

    python -m coarse_tpp.prepare.jodie --csv RAW/mooc.csv --name mooc [--data DATA_DIR]

Reads the CSV (header "user,item,timestamp,state_label,features..."; only the first three columns are used, streamed),
builds one sequence per user ordered by time (mark = item index in order of appearance, time unit as recorded:
seconds), drops users with fewer than 2 interactions, and deals the sequences by one fixed permutation (SEED) into
train / dev / test with proportions 80 / 10 / 10. Zero gaps (same recorded second) are stored as 1e-6;
time_since_start keeps the recorded value. Writes DATA_DIR/<name>/{train,dev,test}.pkl.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import data as D
from .common import FLOOR, md5, write_split

SEED = 20261015
SPLIT = (0.8, 0.1, 0.1)


def read_interactions(path: Path):
    """{user: [(timestamp, item_index)]} streamed from the CSV; items indexed 0..n_items-1 in order of appearance."""
    by_user, items = defaultdict(list), {}
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        assert [h.strip().removesuffix("_id") for h in header[:3]] == ["user", "item", "timestamp"], header[:3]
        for row in rd:
            if len(row) < 3:
                continue
            u, it, t = row[0], row[1], float(row[2])
            k = items.setdefault(it, len(items))
            by_user[u].append((t, k))
    return by_user, len(items)


def to_sequences(by_user):
    seqs = []
    for u in sorted(by_user):
        ev = sorted(by_user[u], key=lambda x: x[0])        # stable: tied events keep their file order
        if len(ev) < 2:
            continue
        t0 = ev[0][0]
        out, prev = [], None
        for t, k in ev:
            gap = 0.0 if prev is None else t - prev
            out.append(dict(time_since_start=float(t - t0), time_since_last_event=float(FLOOR if gap <= 0 else gap),
                            type_event=int(k)))
            prev = t
        seqs.append(out)
    return seqs


def deal(seqs, seed=SEED):
    perm = np.random.default_rng(seed).permutation(len(seqs))
    n_tr = int(round(SPLIT[0] * len(seqs))); n_dv = int(round(SPLIT[1] * len(seqs)))
    idx = dict(train=perm[:n_tr], dev=perm[n_tr:n_tr + n_dv], test=perm[n_tr + n_dv:])
    return {sp: [seqs[i] for i in ix] for sp, ix in idx.items()}


def convert(csv_path: Path, name: str, root: Path):
    by_user, n_items = read_interactions(csv_path)
    seqs = to_sequences(by_user)
    row = dict(dataset=name, source_md5=md5(csv_path), n_users=len(by_user), n_items=n_items, n_seq=len(seqs), splits={})
    for sp, ss in deal(seqs).items():
        p = write_split(root / name, sp, {sp: ss, "dim_process": n_items}, protocol=5, overwrite=True)
        g = np.asarray([e["time_since_last_event"] for s in ss for e in s[1:]], float)
        row["splits"][sp] = dict(n_seq=len(ss), n_targets=int(g.size), tie_share=float((g <= 1.000001e-6).mean()), md5=md5(p))
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default=None)
    a = ap.parse_args(argv)
    root = Path(a.data) if a.data else D.data_dir()
    print(json.dumps(convert(Path(a.csv), a.name, root), indent=1))


if __name__ == "__main__":
    main()
