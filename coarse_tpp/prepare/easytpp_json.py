"""EasyTPP data sets in their Hugging Face JSON form (train.json, dev.json, test.json; one record per sequence with
the lists time_since_start, time_since_last_event, type_event and dim_process) as per-split pickles.

    python -m coarse_tpp.prepare.easytpp_json --json-dir RAW/easytpp/volcano --name volcano
    python -m coarse_tpp.prepare.easytpp_json --json-dir RAW/easytpp/taxi --name taxi --max-seqs 2000 0 500 --min-len 5

Zero gaps are stored as 1e-6 (the NHP / EasyTPP convention, first event included). --max-seqs keeps the first n
records of train / dev / test (0 = all; Taxi and Earthquake: 2000 0 500), --min-len drops shorter sequences after
that cut (Taxi and Earthquake: 5). Record indices are kept as sequence keys.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import data as D
from .common import SPLITS, floor_zero_gaps, write_split


def convert(json_dir: Path, name: str, root: Path, max_seqs=None, min_len: int = 1):
    out = {}
    for k, sp in enumerate(SPLITS):
        rows = json.loads((Path(json_dir) / f"{sp}.json").read_text())
        seqs, dim = {}, None
        for i, row in enumerate(rows):
            if max_seqs and max_seqs[k] and i >= max_seqs[k]:
                break
            dim = int(row["dim_process"])
            ev = [{"time_since_start": float(a), "time_since_last_event": float(b), "type_event": int(c)}
                  for a, b, c in zip(row["time_since_start"], row["time_since_last_event"], row["type_event"])]
            if len(ev) >= min_len:
                seqs[i] = floor_zero_gaps(ev)
        write_split(root / name, sp, {sp: seqs, "dim_process": dim}, overwrite=True)
        out[sp] = len(seqs)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-seqs", nargs=3, type=int, default=None)
    ap.add_argument("--min-len", type=int, default=1)
    ap.add_argument("--data", default=None)
    a = ap.parse_args(argv)
    root = Path(a.data) if a.data else D.data_dir()
    print(a.name, convert(Path(a.json_dir), a.name, root, a.max_seqs, a.min_len))


if __name__ == "__main__":
    main()
