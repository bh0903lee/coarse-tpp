"""Email-EU and CollegeMsg (SNAP temporal networks) as event sequences, one sequence per sender.

    python -m coarse_tpp.prepare.snap_messages --email RAW/email-Eu-core-temporal.txt --college RAW/CollegeMsg.txt

Sources: https://snap.stanford.edu/data/email-Eu-core-temporal.html (Paranjape, Benson and Leskovec, WSDM 2017) and
https://snap.stanford.edu/data/CollegeMsg.html (Panzarasa, Opsahl and Carley, JASIST 2009); lines "SRC DST TIME".
Per sender: the event times (sorted), senders with at least 5 events (Email-EU: the last 80 events of each;
CollegeMsg: all events). The sequences are
shuffled (Python `random`, seed 42; Email-EU first, then CollegeMsg from the same generator), capped at
int(1000 / 0.7) sequences, and split 70 / 10 / 20. Time-only (single mark); zero gaps are stored as 1e-6.
Both data sets must be converted in one call, in this order, to reproduce the splits.
"""
from __future__ import annotations

import argparse
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import data as D

MIN_EV, MAX_TRAIN = 5, 1000
MAX_LEN = {"email_eu": 80, "collegemsg": None}


def build_sequences(path, max_len=None):
    ev = defaultdict(list)
    for line in open(path):
        p = line.split()
        if len(p) < 3:
            continue
        try:
            s = int(p[0]); t = float(p[2])
        except ValueError:
            continue
        ev[s].append(t)
    seqs = []
    for s, ts in ev.items():
        ts = sorted(ts)
        if len(ts) >= MIN_EV:
            seqs.append(np.array(ts[-max_len:] if max_len else ts, dtype=np.float64))
    return seqs


def to_records(seq):
    t0, prev, recs = seq[0], seq[0], []
    for i, t in enumerate(seq):
        iet = 0.0 if i == 0 else max(t - prev, 1e-6)
        recs.append({"time_since_start": float(t - t0), "time_since_last_event": float(iet), "type_event": 0})
        prev = t
    return recs


def write(out_dir: Path, seqs):
    random.shuffle(seqs)
    if len(seqs) > int(MAX_TRAIN / 0.7):
        seqs = seqs[: int(MAX_TRAIN / 0.7)]
    n = len(seqs); n_tr = int(n * 0.7); n_dev = int(n * 0.1)
    parts = dict(train=seqs[:n_tr], dev=seqs[n_tr:n_tr + n_dev], test=seqs[n_tr + n_dev:])
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, S in parts.items():
        with open(out_dir / f"{split}.pkl", "wb") as fh:
            pickle.dump({split: {i: to_records(s) for i, s in enumerate(S)}, "dim_process": 1}, fh, pickle.HIGHEST_PROTOCOL)
    return {k: len(v) for k, v in parts.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--email", required=True)
    ap.add_argument("--college", required=True)
    ap.add_argument("--data", default=None)
    a = ap.parse_args(argv)
    root = Path(a.data) if a.data else D.data_dir()
    random.seed(42); np.random.seed(42)
    for name, path in (("email_eu", a.email), ("collegemsg", a.college)):
        seqs = build_sequences(path, MAX_LEN[name])
        print(name, write(root / name, [s.copy() for s in seqs]))


if __name__ == "__main__":
    main()
