"""Re-scoring of trained mixture-head models under floors and the interval score (Section 5.1). No training.

    python -m coarse_tpp.rescore score RUN_DIR [--selection own] [--audit results/audit/audit.json]
    python -m coarse_tpp.rescore report RUN_DIR [--compare floor_1e-6_sigma_sigmoid floor_1e-6_standard]

score: for every finished cell of RUN_DIR (e.g. the floor_convention experiment) the model is rebuilt, the
checkpoint best_<selection>.pt is loaded, and the per-event mixture is recomputed on dev and test with the inputs the
model was trained on. Scores (mean per target event, with tie / non-tie means): floor_1e-10 / floor_1e-6 / floor_1e-5
(density at max(tau~, eps)) and, with the audited delta of the data set (else the cell's delta, if positive), interval (log P(tau~ <= tau < tau~ + delta)).
Recorded gaps stored as <= 1.000001e-6 are ties. For models trained and selected by the floor 1e-6 log-likelihood
(selection "own" of treatment floor, eps = 1e-6) the recomputed dev score is compared with the stored dev criterion.
Writes RUN_DIR/rescore.jsonl; a later call resumes it and stops if a stored row has another selection or delta.

report: test scores of the same models under the three floors and the interval score (mean over seeds, per data set
and learning rate), the seed-paired comparison of two head parameterizations under each score (learning rate per
head chosen by the dev mean of the same score; one-sided 95% t bounds), and the change of their difference between
floor 1e-10 and floor 1e-5 (two-sided 90% t interval).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy import stats as SS

from . import data as D
from . import lik
from .analysis import audit_rows, fmt
from .data import TIE_STORED, coarse_stats, load, stores_floor
from .models import build

FLOORS = {"floor_1e-10": 1e-10, "floor_1e-6": 1e-6, "floor_1e-5": 1e-5}


@torch.no_grad()
def per_event(model, views, stats, device, bs=64):
    """Concatenated mixture parameters (float64) and stored gaps of all targets of a split, in order."""
    out = {k: [] for k in ("lw", "mu", "ls", "y")}
    sp = views.coarse
    for i0 in range(0, len(sp), bs):
        idx = np.arange(i0, min(i0 + bs, len(sp)))
        b = D.make_batch(sp, idx, device)
        law, _ = model(b, stats)
        m = (b.tmask * D.score_mask(views, idx, b.iet.shape[1], device)).bool()
        out["lw"].append(law.log_w[m].cpu()); out["mu"].append(law.mu[m].cpu())
        out["ls"].append(law.log_sigma[m].cpu()); out["y"].append(b.iet[:, 1:][m].double().cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def scores(lw, mu, ls, y, delta, floor=True):
    """floor: the data set stores ties as TIE_STORED (data.stores_floor); process logs store exact zeros."""
    law = lik.LogNormMix(lw, mu, ls)
    tau = (torch.where(y <= TIE_STORED, torch.zeros_like(y), y) if floor else y).double()
    tie = tau <= 0
    v = {k: lik.event_loglik(law, tau, "floor", eps=e) for k, e in FLOORS.items()}
    if delta:
        v["interval"] = lik.event_loglik(law, tau, "interval", delta=delta)
    out = {"n": int(len(tau)), "n_tie": int(tie.sum())}
    for k, x in v.items():
        out[k] = float(x.mean())
        out[f"{k}_tie"] = float(x[tie].mean()) if tie.any() else None
        out[f"{k}_nontie"] = float(x[~tie].mean()) if (~tie).any() else None
    return out


def score_cell(cfg, ckpt: Path, delta, device, cache=None):
    cache = {} if cache is None else cache
    ds, raw = cfg["dataset"], bool(cfg.get("raw_gaps"))
    if (ds, raw) not in cache:
        tr, dv, te = (load(ds, s, raw=raw) for s in ("train", "dev", "test"))
        cache.clear(); cache[(ds, raw)] = (tr, dv, te, coarse_stats(tr, float(cfg["delta"])))
    tr, dv, te, stats = cache[(ds, raw)]
    model = build(cfg, stats, tr.coarse.n_types).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    out = {}
    for name, v in (("dev", dv), ("test", te)):
        pe = per_event(model, v, stats, device)
        out[name] = scores(pe["lw"], pe["mu"], pe["ls"], pe["y"], delta, floor=stores_floor(ds))
    return out


def score(run, selection="own", audit_path=None):
    run = Path(run)
    audit = audit_rows(audit_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dst = run / "rescore.jsonl"
    done = {}
    if dst.exists():
        for x in dst.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                done[r["cell_id"]] = (r.get("selection"), r.get("delta"))
    cache = {}
    with open(dst, "a") as fh:
        for line in (run / "grid.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            cfg = json.loads(Path(m["config"]).read_text())
            delta = (audit.get(cfg["dataset"]) or {}).get("delta") or (float(cfg["delta"]) if float(cfg["delta"]) > 0 else None)
            if m["cell_id"] in done:
                # resume only with the same selection and delta; never mix rows of different settings in one file
                if done[m["cell_id"]] != (selection, delta):
                    sel0, d0 = done[m["cell_id"]]
                    raise SystemExit(f"{dst} holds {m['cell_id']} with selection {sel0}, delta {d0}; this call asks for "
                                     f"selection {selection}, delta {delta}. Move or remove {dst} to re-score.")
                continue
            cdir = Path(cfg["out_dir"])
            ck = cdir / f"best_{selection}.pt"
            if not (cdir / "result.json").exists() or not ck.exists() or cfg["head"] != "mix":
                continue
            res = json.loads((cdir / "result.json").read_text())
            r = dict(cell_id=m["cell_id"], arm=cfg["arm"], dataset=cfg["dataset"], seed=cfg["seed"], lr_tag=cfg["lr_tag"],
                     selection=selection, delta=delta, **score_cell(cfg, ck, delta, device, cache))
            if selection == "own" and cfg["treatment"] == "floor" and cfg.get("eps") == 1e-6:
                stored = res["by_selection"][selection]["dev_criterion"]          # the dev floor 1e-6 log-likelihood
                r["check"] = dict(stored_dev=stored, abs_diff=abs(r["dev"]["floor_1e-6"] - stored))
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(f"{m['cell_id']}: dev floor_1e-6 {r['dev']['floor_1e-6']:.6f}, test interval {r['test'].get('interval')}", flush=True)


# ------------------------------------------------------------------ report
def load_rows(run):
    return [json.loads(x) for x in (Path(run) / "rescore.jsonl").read_text().splitlines() if x.strip()]


def by_seed(rows, arm, ds, key):
    """{lr: {seed: (dev, test)}} of score `key`."""
    out = {}
    for r in rows:
        if r["arm"] == arm and r["dataset"] == ds and r["dev"].get(key) is not None:
            out.setdefault(r["lr_tag"], {})[r["seed"]] = (r["dev"][key], r["test"][key])
    return out


def best_lr(rows, arm, ds, key):
    lrs = by_seed(rows, arm, ds, key)
    if not lrs:
        return None
    return max(lrs.items(), key=lambda kv: np.mean([v[0] for v in kv[1].values()]))[1]


def compare(rows, a, b, ds, key):
    """Seed-paired test differences a - b, each arm at its own lr (dev mean of the same score)."""
    x, y = best_lr(rows, a, ds, key), best_lr(rows, b, ds, key)
    if not x or not y:
        return None
    return {s: x[s][1] - y[s][1] for s in sorted(set(x) & set(y))}


def t_bounds(d, level=0.95):
    d = np.asarray(d, float); n = len(d)
    m = d.mean()
    if n < 2:
        return m, NAN, NAN
    h = SS.t.ppf(level, n - 1) * d.std(ddof=1) / math.sqrt(n)
    return m, m - h, m + h


NAN = float("nan")


def report(run, compare_arms=None, tied_min=0.01):
    rows = load_rows(run)
    checks = [r["check"]["abs_diff"] for r in rows if r.get("check")]
    L = ["# Re-scoring of trained models", "", f"cells scored: {len(rows)}; max |dev floor_1e-6 - stored dev criterion| over "
         f"{len(checks)} checked cells = {max(checks, default=NAN):.2e}", "",
         "## Test log-likelihood per event of the same models under three floors and the interval score (mean over seeds)", "",
         "| data set | model | lr | ties (test) | floor 1e-10 | floor 1e-6 | floor 1e-5 | interval | spread over floors |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["arm"], r["lr_tag"]), []).append(r)
    spread = []
    for (ds, arm, lr), rs in sorted(groups.items()):
        mean = lambda k: float(np.mean([x["test"][k] for x in rs])) if all(x["test"].get(k) is not None for x in rs) else NAN
        ts = rs[0]["test"]["n_tie"] / rs[0]["test"]["n"]
        f = [mean(k) for k in FLOORS]
        sp = max(f) - min(f)
        if ts >= tied_min:
            spread.append(sp)
        L.append(f"| {ds} | {arm} | {lr} | {ts:.4f} | {f[0]:.4f} | {f[1]:.4f} | {f[2]:.4f} | {mean('interval'):.4f} | {sp:.3f} |")
    if spread:
        L += ["", f"Spread over the three floors on data sets with >= {tied_min:.0%} ties: {min(spread):.2f} to {max(spread):.2f} nats."]
    if compare_arms:
        a, b = compare_arms
        dsets = sorted({r["dataset"] for r in rows})
        L += ["", f"## {a} - {b}: seed-paired test difference under each score (one-sided 95% bounds)", "",
              "| data set | " + " | ".join(list(FLOORS) + ["interval"]) + " | change floor 1e-10 -> 1e-5 (two-sided 90%) |",
              "|---|" + "---|" * (len(FLOORS) + 2)]
        for ds in dsets:
            cols = []
            for k in list(FLOORS) + ["interval"]:
                d = compare(rows, a, b, ds, k)
                if d:
                    m, lo, hi = t_bounds(list(d.values()))
                    cols.append(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
                else:
                    cols.append("—")
            d10, d5 = compare(rows, a, b, ds, "floor_1e-10"), compare(rows, a, b, ds, "floor_1e-5")
            if d10 and d5:
                ch = [d5[s] - d10[s] for s in sorted(set(d10) & set(d5))]
                m, lo, hi = t_bounds(ch)
                cols.append(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]{' outside +-0.01' if lo > 0.01 or hi < -0.01 else ''}")
            else:
                cols.append("—")
            L.append(f"| {ds} | " + " | ".join(cols) + " |")
    (Path(run) / "rescore_report.md").write_text("\n".join(L) + "\n")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("score", "report"))
    ap.add_argument("run")
    ap.add_argument("--selection", default="own")
    ap.add_argument("--audit", default=None)
    ap.add_argument("--compare", nargs=2, default=None, metavar=("ARM_A", "ARM_B"))
    a = ap.parse_args(argv)
    if a.cmd == "score":
        score(a.run, a.selection, a.audit)
    else:
        print("\n".join(report(a.run, a.compare)))


if __name__ == "__main__":
    main()
