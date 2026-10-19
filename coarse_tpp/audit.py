"""Audit of the recording resolution of event-sequence data sets (Section 2, Appendix B). No training.

    python -m coarse_tpp.audit [--data DATA_DIR] [--out results/audit] [--datasets NAME ...]

For each data set (all splits pooled; per-split tie shares reported as well): tie share, smallest positive gap,
median gap, the lattice test over the candidate grids, the hidden-tie flag and the assigned resolution delta with
the rule that applied (coarse_tpp.score.detect_resolution). A data set whose directory contains resolution.json
({"delta": ...}) has a resolution known by construction (rule i). Writes OUT/audit.json and OUT/audit.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import data as D
from .score import detect_resolution


def constructed(ds: str):
    p = D.data_dir() / ds / "resolution.json"
    return float(json.loads(p.read_text())["delta"]) if p.exists() else None


def audit(ds: str) -> dict | None:
    out = {"dataset": ds}
    pooled = []
    for sp in ("train", "dev", "test"):
        try:
            v = D.load(ds, sp)
        except FileNotFoundError:
            return None
        g = np.concatenate([x[1:] for x in v.coarse.iet])
        out[f"n_{sp}"] = int(len(g)); out[f"seqs_{sp}"] = len(v.coarse)
        out[f"tie_share_{sp}"] = float(np.mean(g <= 0.0))
        pooled.append(g)
    # gaps are already read with the stored-tie convention of the data set (coarse_tpp.data.load)
    out.update(detect_resolution(np.concatenate(pooled), constructed=constructed(ds), tie_stored=0.0))
    return out


def fmt_row(r):
    dh = "—" if r.get("delta_hat") is None else f"{r['delta_hat']:.6g} ({r['delta_hat_rel_median']:.2g})"
    hd = f"{r.get('hidden_s1', 0):.3f} / {r.get('hidden_s2', 0):.3f}" + (" **flag**" if r.get("hidden_flag") else "")
    dl = "none" if r["delta"] is None else f"{r['delta']:.6g}"
    return (f"| {r['dataset']} | {r['n']} | {r['tie_share']:.4f} ({r['tie_share_train']:.4f}) | {r.get('min_pos_gap', float('nan')):.6g} | "
            f"{r.get('median_gap', float('nan')):.6g} | {dh} | {hd} | {dl} {r['delta_rule']} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="data directory (default: COARSE_TPP_DATA or ./data)")
    ap.add_argument("--out", default="results/audit")
    ap.add_argument("--datasets", nargs="*", default=None)
    a = ap.parse_args(argv)
    if a.data:
        D.set_data_dir(a.data)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = a.datasets or sorted(p.name for p in D.data_dir().iterdir() if (p / "train.pkl").exists())
    rows = [r for r in (audit(d) for d in names) if r is not None]
    old = {r["dataset"]: r for r in json.loads((out / "audit.json").read_text())} if (out / "audit.json").exists() else {}
    old.update({r["dataset"]: r for r in rows})
    allrows = [old[k] for k in sorted(old)]
    (out / "audit.json").write_text(json.dumps(allrows, indent=1))
    L = ["# Recording-resolution audit", "",
         "Gaps of all splits pooled; ties = recorded zero gaps. delta_hat = largest candidate grid holding >= 99% of the "
         "positive gaps (relative to the median positive gap). Hidden ties: shares of positive gaps in [m, 2m) / [2m, 10m).", "",
         "| data set | gaps | tie share (train) | min pos gap | median gap | delta_hat (rel. median) | hidden ties | delta (rule) |",
         "|---|---:|---:|---:|---:|---|---|---|"]
    L += [fmt_row(r) for r in allrows]
    (out / "audit.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
