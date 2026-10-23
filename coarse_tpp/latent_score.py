"""Latent-time score B of trained cells under both readouts (no training).

    python -m coarse_tpp.latent_score RUN_DIR [--out RUN_DIR/latent_scores.jsonl]

The coarsened data sets keep the original time of every event (time_since_start_fine, coarse_tpp.prepare.coarsen),
so the latent time since the last recorded time, s_i = t_fine_i - t~_{i-1}, is known and B = log q_B(s_i) can be
computed with the readout of each treatment (coarse_tpp.readout); the synthetic data sets carry s_i directly. For
every finished cell of the grid whose data set has latent times, the primary checkpoint best_C.pt is rebuilt and
scored on dev and test:
  C       recorded-cell score under the treatment's readout (a check: it must reproduce result.json within TOL,
          else the cell is flagged (mismatch) and left out of the tables)
  B       log q_B(s_i) under the treatment's readout (readout.RULE)
  B_alt   log q_B(s_i) under the alternative readout (readout.ALT: raw <-> smooth, point_mass -> point_mass_raw)
with tie / non-tie splits. Why both: the smooth readout builds the uniform in-cell position of the previous record
into q_B and the raw readout does not, so a comparison of a smooth-type model with a raw-type model on B mixes the
objective and the readout; the report of Supplementary S4 compares the triangular-kernel objective with each other
treatment under the readout that is better for that treatment on the dev split (coarse_tpp.analysis).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import readout as Ro
from .data import coarse_stats, load, load_fine_s, score_mask
from .models import build

TOL = 1e-4
KEYS = ("C", "B", "B_alt")
SCHEMA = 2              # rows with B_alt; rows of version 1.0 (B and C only) are re-scored


def _pad(arrs, idx, L, device):
    out = np.zeros((len(idx), L - 1))
    for b, i in enumerate(idx):
        a = arrs[i][1:]
        out[b, :len(a)] = a
    return torch.tensor(out, dtype=torch.float64, device=device)


def has_latent(ds: str) -> bool:
    return ds.startswith("coarsened_") or ds.startswith("synth_")


def latent_s(views, ds: str, split: str):
    """Per sequence, s_i = t_i - t~_{i-1}: stored with the synthetic views, read from the kept fine times otherwise."""
    if ds.startswith("synth_"):
        if views.lat_s is None:
            raise ValueError(f"{ds}: no latent times")
        return views.lat_s
    return load_fine_s(ds, split)


@torch.no_grad()
def score_split(model, views, fine, stats, cfg, device, bs=64):
    """Mean per-target C (check), B and B_alt, with the tie / non-tie split, over a split (scored targets only)."""
    model.eval()
    delta, r = float(cfg["delta"]), Ro.RULE[cfg["treatment"]]
    acc = {k: [0.0, 0.0, 0.0] for k in KEYS}               # all, tie, non-tie sums
    n = nt = 0.0
    sp = views.coarse
    for i0 in range(0, len(sp), bs):
        idx = np.arange(i0, min(i0 + bs, len(sp)))
        b = D.make_batch(sp, idx, device)
        L = b.iet.shape[1]
        m = (b.tmask * score_mask(views, idx, L, device)).double()
        tau = b.iet[:, 1:].double()
        tie = Ro.tie_mask(tau, delta).double() * m
        law, tl = model(b, stats)
        s = _pad(fine, idx, L, device)
        s = torch.where(m > 0, s, torch.ones_like(s)).clamp(min=1e-10)
        v = dict(C=Ro.C(law, tl, tau, delta, r), B=Ro.B(law, tl, s, delta, r), B_alt=Ro.B(law, tl, s, delta, Ro.ALT[r]))
        for k in KEYS:
            x = torch.where(m > 0, v[k], torch.zeros_like(v[k]))
            acc[k][0] += float((x * m).sum()); acc[k][1] += float((x * tie).sum()); acc[k][2] += float((x * (m - tie)).sum())
        n += float(m.sum()); nt += float(tie.sum())
    out = dict(n=int(round(n)), n_tie=int(round(nt)))
    for k in KEYS:
        out[k] = acc[k][0] / n
        out[f"{k}_tie"] = acc[k][1] / nt if nt else None
        out[f"{k}_nontie"] = acc[k][2] / (n - nt) if n > nt else None
    return out


def current(r: dict) -> bool:
    """A row of the current schema: it carries B_alt on both splits."""
    return r.get("schema") == SCHEMA and "B_alt" in (r.get("test") or {}) and "B_alt" in (r.get("dev") or {})


def fresh_rows(dst: Path) -> dict:
    """{cell_id: row} of the current-schema rows of an existing output file. Rows of an older schema (version 1.0
    wrote B and C only) are not reusable: the file is backed up once to <name>.v1 and rewritten with the current rows
    only, with a notice, so that the missing cells are scored again instead of being skipped."""
    if not dst.exists():
        return {}
    rows = [json.loads(x) for x in dst.read_text().splitlines() if x.strip()]
    keep = {r["cell_id"]: r for r in rows if current(r)}
    stale = [r for r in rows if not current(r)]
    if stale:
        bak = dst.with_name(dst.name + ".v1")
        if not bak.exists():
            bak.write_bytes(dst.read_bytes())
        dst.write_text("".join(json.dumps(r) + "\n" for r in keep.values()))
        print(f"{dst}: {len(stale)} rows of an older schema (no B_alt) moved aside to {bak.name}; their cells are scored again",
              flush=True)
    return keep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run = Path(a.run)
    dst = Path(a.out) if a.out else run / "latent_scores.jsonl"
    done = set(fresh_rows(dst))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = {}
    with open(dst, "a") as fh:
        for line in (run / "grid.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            if m["cell_id"] in done or not has_latent(m["dataset"]):
                continue
            cfg = json.loads(Path(m["config"]).read_text())
            cdir = Path(cfg["out_dir"])
            if not (cdir / "result.json").exists() or not (cdir / "best_C.pt").exists():
                continue
            res = json.loads((cdir / "result.json").read_text())
            if res.get("status") != "ok" or res.get("primary_selection", "C") != "C":
                continue
            ds = cfg["dataset"]
            if ds not in cache:
                tr, dv, te = (load(ds, s) for s in ("train", "dev", "test"))
                cache = {ds: (tr, dv, te, latent_s(dv, ds, "dev"), latent_s(te, ds, "test"), coarse_stats(tr, float(cfg["delta"])))}
            tr, dv, te, f_dv, f_te, stats = cache[ds]
            model = build(cfg, stats, tr.coarse.n_types).to(device)
            model.load_state_dict(torch.load(cdir / "best_C.pt", map_location=device, weights_only=True))
            rule = Ro.RULE[cfg["treatment"]]
            r = dict(schema=SCHEMA, cell_id=m["cell_id"], experiment=cfg.get("experiment"), arm=cfg["arm"], dataset=ds, seed=cfg["seed"],
                     config=dict(dataset=ds, encoder=cfg["encoder"], head=cfg["head"], K=cfg["K"], lr_tag=cfg["lr_tag"],
                                 treatment=cfg["treatment"], delta=cfg["delta"], iw_samples=cfg.get("iw_samples")),
                     rule=rule, alt_rule=Ro.ALT[rule], dev_crit=res["by_selection"]["C"]["dev_criterion"],
                     wall_sec=res.get("wall_sec"),
                     dev=score_split(model, dv, f_dv, stats, cfg, device), test=score_split(model, te, f_te, stats, cfg, device))
            r["check"] = dict(dev=abs(r["dev"]["C"] - res["dev"]["C"]), test=abs(r["test"]["C"] - res["test"]["C"]))
            r["mismatch"] = bool(max(r["check"].values()) > TOL)
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(f"{m['cell_id']} test B {r['test']['B']:.5f} B_alt {r['test']['B_alt']:.5f} C check {r['check']['test']:.1e}"
                  f"{' MISMATCH' if r['mismatch'] else ''}", flush=True)


if __name__ == "__main__":
    main()
