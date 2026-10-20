"""Latent-time score B of trained cells on coarsened real data (no training).

    python -m coarse_tpp.latent_score RUN_DIR [--out RUN_DIR/latent_scores.jsonl]

The coarsened data sets keep the original time of every event (time_since_start_fine, coarse_tpp.prepare.coarsen),
so the latent time since the last recorded time, s_i = t_fine_i - t~_{i-1}, is known and B = log q_B(s_i) can be
computed with the readout of each treatment (coarse_tpp.readout). For every finished cell the primary checkpoint
best_C.pt is rebuilt and scored on dev and test; the recomputed test and dev C must reproduce result.json within
TOL, else the cell is flagged (mismatch) and left out of the tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import readout as Ro
from .data import coarse_stats, load, load_fine_s
from .models import build

TOL = 1e-4


def _pad(arrs, idx, L, device):
    out = np.zeros((len(idx), L - 1))
    for b, i in enumerate(idx):
        a = arrs[i][1:]
        out[b, :len(a)] = a
    return torch.tensor(out, dtype=torch.float64, device=device)


@torch.no_grad()
def score_split(model, views, fine, stats, cfg, device, bs=64):
    """Mean per-target C (check) and B, with the tie / non-tie split, over a split."""
    model.eval()
    delta, r = float(cfg["delta"]), Ro.RULE[cfg["treatment"]]
    acc = dict(C=0.0, B=0.0, B_tie=0.0, B_non=0.0, n=0.0, n_tie=0.0)
    sp = views.coarse
    for i0 in range(0, len(sp), bs):
        idx = np.arange(i0, min(i0 + bs, len(sp)))
        b = D.make_batch(sp, idx, device)
        L = b.iet.shape[1]
        m = b.tmask.double()
        tau = b.iet[:, 1:].double()
        tie = Ro.tie_mask(tau, delta).double() * m
        law, tl = model(b, stats)
        c = Ro.C(law, tl, tau, delta, r)
        s = _pad(fine, idx, L, device)
        s = torch.where(m > 0, s, torch.ones_like(s)).clamp(min=1e-10)
        bb = Ro.B(law, tl, s, delta, r)
        c = torch.where(m > 0, c, torch.zeros_like(c)); bb = torch.where(m > 0, bb, torch.zeros_like(bb))
        acc["C"] += float((c * m).sum()); acc["B"] += float((bb * m).sum())
        acc["B_tie"] += float((bb * tie).sum()); acc["B_non"] += float((bb * (m - tie)).sum())
        acc["n"] += float(m.sum()); acc["n_tie"] += float(tie.sum())
    n, nt = acc["n"], acc["n_tie"]
    return dict(C=acc["C"] / n, B=acc["B"] / n, B_tie=acc["B_tie"] / nt if nt else None,
                B_nontie=acc["B_non"] / (n - nt) if n > nt else None, n=int(n), n_tie=int(nt))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run = Path(a.run)
    dst = Path(a.out) if a.out else run / "latent_scores.jsonl"
    done = {json.loads(x)["cell_id"] for x in dst.read_text().splitlines() if x.strip()} if dst.exists() else set()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = {}
    with open(dst, "a") as fh:
        for line in (run / "grid.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            if m["cell_id"] in done:
                continue
            cfg = json.loads(Path(m["config"]).read_text())
            cdir = Path(cfg["out_dir"])
            if not (cdir / "result.json").exists() or not (cdir / "best_C.pt").exists():
                continue
            res = json.loads((cdir / "result.json").read_text())
            if res.get("status") != "ok":
                continue
            ds = cfg["dataset"]
            if ds not in cache:
                tr, dv, te = (load(ds, s) for s in ("train", "dev", "test"))
                cache = {ds: (tr, dv, te, load_fine_s(ds, "dev"), load_fine_s(ds, "test"), coarse_stats(tr, float(cfg["delta"])))}
            tr, dv, te, f_dv, f_te, stats = cache[ds]
            model = build(cfg, stats, tr.coarse.n_types).to(device)
            model.load_state_dict(torch.load(cdir / "best_C.pt", map_location=device, weights_only=True))
            r = dict(cell_id=m["cell_id"], arm=cfg["arm"], dataset=ds, seed=cfg["seed"],
                     config=dict(dataset=ds, encoder=cfg["encoder"], head=cfg["head"], K=cfg["K"], lr_tag=cfg["lr_tag"],
                                 treatment=cfg["treatment"], delta=cfg["delta"]),
                     dev_crit=res["by_selection"]["C"]["dev_criterion"],
                     dev=score_split(model, dv, f_dv, stats, cfg, device), test=score_split(model, te, f_te, stats, cfg, device))
            r["check"] = dict(dev=abs(r["dev"]["C"] - res["dev"]["C"]), test=abs(r["test"]["C"] - res["test"]["C"]))
            r["mismatch"] = bool(max(r["check"].values()) > TOL)
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(f"{m['cell_id']} test B {r['test']['B']:.5f} C check {r['check']['test']:.1e}", flush=True)


if __name__ == "__main__":
    main()
