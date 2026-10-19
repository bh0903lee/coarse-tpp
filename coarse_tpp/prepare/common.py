"""Helpers shared by the data preparation scripts."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

SPLITS = ("train", "dev", "test")
FLOOR = 1e-6                     # stored value of a zero gap in the NHP / EasyTPP-style files


def md5(p: Path) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def items(d: dict, split: str):
    """[(key, sequence)] of a loaded split and whether it was stored as a dict."""
    inner = d[split]
    return (list(inner.items()), True) if isinstance(inner, dict) else (list(enumerate(inner)), False)


def read(ds_dir: Path, split: str) -> dict:
    with open(Path(ds_dir) / f"{split}.pkl", "rb") as fh:
        return pickle.load(fh, encoding="latin-1")


def write_split(ds_dir: Path, split: str, obj: dict, protocol=pickle.HIGHEST_PROTOCOL, overwrite=False):
    """Write {split: sequences, "dim_process": n}; refuses to replace a different existing file unless overwrite."""
    ds_dir = Path(ds_dir); ds_dir.mkdir(parents=True, exist_ok=True)
    blob = pickle.dumps(obj, protocol=protocol)
    dst = ds_dir / f"{split}.pkl"
    if dst.exists() and dst.read_bytes() != blob and not overwrite:
        raise SystemExit(f"refusing to overwrite a different file: {dst}")
    dst.write_bytes(blob)
    return dst


def write_resolution(ds_dir: Path, delta: float, note: str):
    """Record a resolution known by construction (read by coarse_tpp.audit as rule i)."""
    (Path(ds_dir) / "resolution.json").write_text(json.dumps(dict(delta=float(delta), note=note), indent=1))


def floor_zero_gaps(seq):
    """Zero gaps stored as FLOOR (the NHP / EasyTPP convention); other fields unchanged."""
    out = []
    for e in seq:
        ev = dict(e)
        if ev.get("time_since_last_event", 0) == 0:
            ev["time_since_last_event"] = FLOOR
        out.append(ev)
    return out


def summary(ds_dir: Path):
    rows = []
    for sp in SPLITS:
        d = read(ds_dir, sp)
        its, _ = items(d, sp)
        g = np.concatenate([[e["time_since_last_event"] for e in s[1:]] for _, s in its if len(s) > 1])
        rows.append(dict(split=sp, n_seq=len(its), n_targets=int(len(g)), tie_share=float(np.mean(g <= 1.000001e-6)),
                         median_gap=float(np.median(g)), md5=md5(Path(ds_dir) / f"{sp}.pkl")))
    return rows
