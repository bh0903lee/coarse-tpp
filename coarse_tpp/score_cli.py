"""Command-line front end of coarse_tpp.score for any model and any event file.

    python -m coarse_tpp.score_cli audit --pkl FILE.pkl [FILE.pkl ...] [--constructed D] [--tie-stored 0]
    python -m coarse_tpp.score_cli score --csv preds.csv [--delta D]

audit: gaps of every given file are pooled, first events excluded; prints the audit rule's output as JSON (tie share,
       lattice candidates with share >= 0.5, delta and the rule that set it). Files are pickles {split: sequences,
       "dim_process": n} with events {"time_since_last_event": gap, ...}; sequences may be a list or a dict.
score: the CSV has one row per target event with columns tau (recorded gap; stored ties <= 1.000001e-6 count as 0),
       F_lo (model distribution function of the time since the last recorded event at tau~) and F_hi (at
       tau~ + delta); for a law with atoms at positive times, the left limits of the distribution function at both
       edges. Values outside [0, 1] or with F_hi < F_lo (beyond a roundoff of 1e-9) are an error. Prints as JSON the
       mean interval score over all events with the tie / non-tie split (-Infinity when a cell has zero
       probability), all_finite, the counts of -inf and non-finite scores, and mean_finite, the mean over the finite
       scores only (a diagnostic, not the score over all events). --delta is recorded in the output only; the
       caller computes F_hi at tau~ + delta.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from . import score as S


def gaps_of_pkl(path: Path) -> np.ndarray:
    d = pickle.load(open(path, "rb"), encoding="latin-1")
    seqs = []
    for k, v in d.items():
        if k == "dim_process" or not isinstance(v, (list, dict)):
            continue
        seqs += list(v.values()) if isinstance(v, dict) else list(v)
    return np.asarray([e["time_since_last_event"] for s in seqs for e in s[1:]], np.float64)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--pkl", nargs="+", required=True); a.add_argument("--constructed", type=float)
    a.add_argument("--tie-stored", type=float, default=S.TIE_STORED)
    s = sub.add_parser("score")
    s.add_argument("--csv", required=True); s.add_argument("--delta", type=float)
    s.add_argument("--tie-stored", type=float, default=S.TIE_STORED)
    args = ap.parse_args(argv)
    if args.cmd == "audit":
        files = [Path(p) for p in args.pkl]
        g = np.concatenate([gaps_of_pkl(p) for p in files])
        out = S.detect_resolution(g, args.constructed, args.tie_stored)
        out["files"] = [str(p) for p in files]
        print(json.dumps(out, indent=1))
    else:
        rows = list(csv.DictReader(open(args.csv)))
        tau = np.asarray([float(r["tau"]) for r in rows]); lo = np.asarray([float(r["F_lo"]) for r in rows])
        hi = np.asarray([float(r["F_hi"]) for r in rows])
        try:
            ll, tie = S.interval_loglik_from_cdf(tau, lo, hi, args.tie_stored)
        except ValueError as e:
            ap.error(f"{args.csv}: {e}")
        out = S.summarize(ll, tie); out["delta"] = args.delta
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
