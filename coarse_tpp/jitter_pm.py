"""Jitter and point-mass scores of trained models and the analyses of Online Resource 1, Section S5 ("Jitter and
point-mass scores"): the jitter and point-mass rows of Table 6 and the jitter / point-mass columns of Tables S4-S8,
S11 and S13. No training; the saved checkpoints (best_C.pt) are re-scored.

    python -m coarse_tpp.jitter_pm score RUN_DIR [--shard i/n] [--cpu] [--threads N] [--arms ARM ...] [--only CELL ...]
                                     [--limit N] [--force] [--retry-quarantined]
    python -m coarse_tpp.jitter_pm edits RUN_DIR [--cpu] [--force]         (rank_change run: edits of Table S8)
    python -m coarse_tpp.jitter_pm report --out DIR [--rank-change RUN] [--rank-change-seeds RUN] [--wider-encoders RUN]
              [--mixture-components RUN] [--density-heads RUN] [--resplit RUN] [--new-datasets RUN] [--process-logs RUN]
              [--synthetic RUN] [--coarsened-real RUN] [--coarsened-more RUN] [--intensity-scores DIR ...]
              [--intensity-jitter DIR ...] [--intensity-repro DIR ...] [--exceedance DIR] [--audit results/audit/audit.json]
              [--B 2000]

Scores (per scored target, nats; batches, masks and tie rule of coarse_tpp.train.evaluate; a tie is a recorded zero
gap, readout.tie_mask):
  jitter       E_u log q(tau~ + delta u), u ~ U(0, 1), by the N_NODES = 32-node Gauss-Legendre rule on (0, 1)
               (deterministic); q is the model's raw density (law.log_density; for the point-mass treatment its
               continuous part q^c, without log(1 - pi)); the density a uniform-jitter score evaluates.
  point_mass   point-mass scoring at resolution delta: a tie scores the probability of the first cell under the
               treatment's readout, i.e. the tie term of C (for the point-mass treatment pi + (1 - pi) Q(first cell));
               a positive gap scores the density of the model's distribution at the recorded gap,
               log(1 - pi) + log q^c(tau~) for the point-mass treatment and log q(tau~) for every other treatment.
               For a model without an atom this is a constructed score (cell probability on ties, density elsewhere);
               it mixes a probability with a density, so only comparisons within a data set are meaningful.
  C, floor_1e-6  recomputed in the same pass for the reproduction check.
Proposition 1 of the paper gives jitter = C - log(delta) - KL for the treatments read out by the raw density (KL of
the uniform law on the recorded cell from q restricted to that cell); the report uses KL_C = C - log(delta) - jitter.

score: every cell of RUN_DIR with status ok, delta > 0 and best_C.pt (optionally only --arms); the model is rebuilt as
coarse_tpp.train.run does and evaluated on dev and test. Validity: the recomputed C must equal the stored C (dev:
by_selection.C.dev_criterion, test: by_selection.C.test.C) within C_TOL and the recomputed floor 1e-6 the stored one
within FLOOR_TOL (looser: the floor score at a very narrow component moves with float order between devices), with
identical event and tie counts; otherwise the cell is quarantined (kept in the file, left out of the report).
Writes RUN_DIR/jitter_pm/cells/<cell_id>.json (dev and test means with tie / non-tie means) and .npz (test
per-sequence sums n, C, floor_1e-6, jitter, point_mass, for the bootstrap). A cached file is reused while its version,
node count and the SHA-256 of the checkpoint match; a quarantined file is recomputed only with --retry-quarantined or
--force.

edits: the component edits of Table S8 (coarse_tpp.component_edits: location, scale, both; the floor-trained models
of the tied rank_change combinations) scored by both scores; writes RUN_DIR/jitter_pm/edits/<item>.json.

report: each section needs the runs it names and is left out otherwise.
  Learning rate and checkpoint: per (combination, treatment) the learning rate with the best mean validation value
  of the score used (floor 1e-6 and C: as in the paper; jitter and point mass: their own validation means), except
  for the comparison with the latent-time score B (Table 6) and the decomposition, where every score uses the
  learning rate chosen by the validation C, and the predictive accuracy (the models scored by
  coarse_tpp.predictive are those at the validation-C learning rate). Deficit: test C of the interval score's top
  treatment minus test C of the score's top treatment (both at their validation-C learning rate); seeds: seeds on
  which that paired difference is positive; rank: rank of the score's top among the treatments by C. Self: the score
  ranks first the model trained with it (floor 1e-6: floor_1e-6, interval: interval, jitter: jitter_gap, point mass:
  point_mass). Tied combinations: tie share >= TIE_MIN (audit file when given, else the stored test tie share).
  Sections: rankings (rank_change; Table S5 columns), bootstrap (Table S6 columns; the design of
  coarse_tpp.rank_bootstrap), robustness checks (ten seeds, wider encoders, mixture components, density heads with
  the Table S7 columns, new splits, further data sets, process logs), Table 6 (synthetic, coarsened_real,
  coarsened_more), decomposition (KL_C; tie and positive-gap parts of the point-mass score; Table S8 edits),
  intensity-based models (Table S4; coarse_tpp.intensity_jitter), threshold-exceedance Brier on the process logs
  (Table S11; selection over treatment and learning rate by the validation mean of each score, the Brier files and
  bootstrap of coarse_tpp.alarms), and predictive accuracy (Table S13; the files of coarse_tpp.predictive).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sst

from . import analysis as A
from . import data as D
from . import experiments as X
from . import lik
from . import readout as Ro
from . import results as R
from . import synthetic as Syn
from .data import coarse_stats, load, score_mask
from .models import build

N_NODES = 32
C_TOL = 1e-5
FLOOR_TOL = 1e-4
VERSION = 1
TIE_MIN = 0.01
WITHIN = 0.01                    # Table 6: first choice within this many nats of the best B
P_MIN = 0.9                      # bootstrap shares counted in the summary
BOOT_SEED, B_DEFAULT = 20261027, 2000
KEYS = ("C", "floor_1e-6", "jitter", "point_mass")
RANK_ARMS = X.RANK_ARMS          # the eight treatments of rank_change
NEW = ("jitter", "point_mass")
SELF = {"floor_1e-10": "floor_1e-10", "floor_1e-6": "floor_1e-6", "floor_1e-5": "floor_1e-5", "C": "interval",
        "jitter": "jitter_gap", "point_mass": "point_mass"}
LABEL = {"floor_1e-10": "floor 1e-10", "floor_1e-6": "floor 1e-6", "floor_1e-5": "floor 1e-5", "C": "interval",
         "jitter": "jitter", "point_mass": "point mass"}


# ------------------------------------------------------------------ per-event scores (tests/test_jitter_pm.py)
def jitter_loglik(law, tau, delta: float, nodes: int = N_NODES):
    """Per-event jitter score sum_k w_k log q(tau + delta u_k) of a law with `log_density` (float64, shape of tau)."""
    u, w = lik.gauss_legendre(nodes, device=tau.device)
    tau = tau.double()
    out = torch.zeros_like(tau)
    for uk, wk in zip(u, w):
        out = out + wk * law.log_density(tau + delta * uk)
    return out


def point_mass_loglik(law, tl, tau, delta: float, rule: str):
    """Per-event point-mass score: the tie term of C (readout `rule`) at ties, the density of the model's distribution
    at positive gaps (log(1 - pi) + log q for the point_mass readout, log q otherwise)."""
    tau = tau.double()
    tie = Ro.tie_mask(tau, delta)
    c = Ro.C(law, tl, tau, delta, rule)
    dens = law.log_density(torch.where(tie, torch.ones_like(tau), tau))
    if rule == "point_mass":
        dens = dens + F.logsigmoid(-tl.double())
    return torch.where(tie, c, dens)


# ------------------------------------------------------------------ scoring of a run
def ckpt_sha(path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def out_dir_of(run, out=None) -> Path:
    return Path(out) if out else Path(run) / "jitter_pm"


@torch.no_grad()
def evaluate(model, views, stats, cfg, device, bs=64, nodes=N_NODES):
    """Means (with tie / non-tie means) of KEYS over the scored targets of a split, and per-sequence sums."""
    from .train import fwd, rule
    model.eval()
    delta = float(cfg["delta"])
    r = rule(cfg)
    sp = views.coarse
    per = {k: np.zeros(len(sp)) for k in KEYS}
    tie_s = {k: 0.0 for k in KEYS}; non_s = {k: 0.0 for k in KEYS}
    n_seq, nt_seq = np.zeros(len(sp)), np.zeros(len(sp))
    for i0 in range(0, len(sp), bs):
        idx = np.arange(i0, min(i0 + bs, len(sp)))
        b = D.make_batch(sp, idx, device)
        m = (b.tmask * score_mask(views, idx, b.iet.shape[1], device)).double()
        tau = b.iet[:, 1:].double()
        tie = Ro.tie_mask(tau, delta).double() * m
        law, tl, _ = fwd(model, b, stats, cfg)
        v = {"C": Ro.C(law, tl, tau, delta, r), "floor_1e-6": lik.event_loglik(law, tau, "floor", eps=1e-6),
             "jitter": jitter_loglik(law, tau, delta, nodes), "point_mass": point_mass_loglik(law, tl, tau, delta, r)}
        for k, x in v.items():
            x = torch.where(m > 0, x, torch.zeros_like(x))
            per[k][idx] = (x * m).sum(1).cpu().numpy()
            tie_s[k] += float((x * tie).sum()); non_s[k] += float((x * (m - tie)).sum())
        n_seq[idx] = m.sum(1).cpu().numpy(); nt_seq[idx] = tie.sum(1).cpu().numpy()
    n, nt = float(n_seq.sum()), float(nt_seq.sum())
    out = dict(n_events=int(round(n)), n_ties=int(round(nt)), log_delta=math.log(delta))
    for k in KEYS:
        out[k] = float(per[k].sum() / n)
        out[f"{k}_tie"] = tie_s[k] / nt if nt else None
        out[f"{k}_nontie"] = non_s[k] / (n - nt) if n > nt else None
    return out, dict(n=n_seq, **per)


def stored(res, split):
    """(C, floor 1e-6 or None, n_events or None, n_ties or None) stored for best_C.pt on `split`."""
    sel = res["by_selection"]["C"]
    if split == "test":
        t = sel["test"]
        return t["C"], t.get("floor_1e-6"), t.get("n_events"), t.get("n_ties")
    primary = res.get("primary_selection", "C") == "C"
    d = res.get("dev") or {}
    return (sel["dev_criterion"], d.get("floor_1e-6") if primary else None, d.get("n_events") if primary else None,
            d.get("n_ties") if primary else None)


def check_split(s, res, split):
    """Adds the differences from the stored values to `s` and returns whether the split passes."""
    C, f6, n, nt = stored(res, split)
    s["d_C"] = s["C"] - C if C is not None else None
    s["d_floor_1e-6"] = s["floor_1e-6"] - f6 if f6 is not None else None
    ok = s["d_C"] is not None and math.isfinite(s["d_C"]) and abs(s["d_C"]) <= C_TOL
    if s["d_floor_1e-6"] is not None:
        ok &= math.isfinite(s["d_floor_1e-6"]) and abs(s["d_floor_1e-6"]) <= FLOOR_TOL
    if n is not None:
        ok &= s["n_events"] == n and (nt is None or s["n_ties"] == nt)
    return bool(ok)


def scorable(res):
    cfg = res["config"]
    return float(cfg["delta"]) > 0 and (Path(cfg["out_dir"]) / "best_C.pt").exists() and "C" in res.get("by_selection", {})


def run_cells(run, arms=None):
    """Finished cells of a run (status ok) that can be scored."""
    out = []
    for line in (Path(run) / "grid.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        cfg = json.loads(Path(json.loads(line)["config"]).read_text())
        st, _, res = R.load(cfg)
        if st == "ok" and res is not None and (arms is None or res["arm"] in arms) and scorable(res):
            out.append(res)
    return out


def cache_state(path, ckpt=None):
    """'ok', 'quarantined', 'stale' (other checkpoint) or 'invalid' (absent, unreadable, other version)."""
    try:
        r = json.loads(Path(path).read_text())
        if r.get("version") != VERSION or r.get("n_nodes") != N_NODES or not all(sp in r for sp in ("dev", "test")):
            return "invalid"
        if ckpt is not None and r.get("ckpt_sha256") != ckpt_sha(ckpt):
            return "stale"
        if not Path(path).with_suffix(".npz").exists():
            return "invalid"
        return "quarantined" if r.get("quarantined", True) else "ok"
    except Exception:
        return "invalid"


def score_cell(res, device, cache):
    cfg = res["config"]
    ds = cfg["dataset"]
    raw = bool(cfg.get("raw_gaps"))
    key = (ds, raw)
    if key not in cache:
        cache.clear()
        cache[key] = {sp: load(ds, sp, raw=raw) for sp in ("train", "dev", "test")}
    v = cache[key]
    stats = coarse_stats(v["train"], float(cfg["delta"]))
    torch.manual_seed(0)
    model = build(cfg, stats, v["train"].coarse.n_types).to(device)
    ckpt = Path(cfg["out_dir"]) / "best_C.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    rec = dict(version=VERSION, n_nodes=N_NODES, cell_id=cfg["cell_id"], arm=res["arm"], dataset=ds, seed=res["seed"],
               lr_tag=cfg["lr_tag"], delta=float(cfg["delta"]), treatment=cfg["treatment"], checkpoint="best_C.pt",
               ckpt_sha256=ckpt_sha(ckpt), device=str(device))
    ok, per_test = True, None
    for split in ("dev", "test"):
        s, per = evaluate(model, v[split], stats, cfg, device)
        ok &= check_split(s, res, split)
        rec[split] = s
        if split == "test":
            per_test = per
    rec["quarantined"] = not ok
    return rec, per_test


def cmd_score(run, out=None, shard="0/1", cpu=False, threads=4, arms=None, only=None, limit=0, force=False,
              retry_quarantined=False):
    torch.set_num_threads(int(threads))
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    cells = run_cells(run, arms)
    if only:
        cells = [c for c in cells if c["cell_id"] in set(only)]
    cells.sort(key=lambda r: (r["config"]["dataset"], bool(r["config"].get("raw_gaps")), r["cell_id"]))
    i, n = (int(x) for x in shard.split("/"))
    dss = sorted({r["config"]["dataset"] for r in cells})                 # shards by data set: one load per worker
    mine = {d for k, d in enumerate(dss) if k % n == i}
    todo = [r for r in cells if r["config"]["dataset"] in mine]
    if limit:
        todo = todo[:limit]
    od = out_dir_of(run, out) / "cells"
    od.mkdir(parents=True, exist_ok=True)
    cache, worst = {}, 0.0
    for res in todo:
        dst = od / f"{res['cell_id']}.json"
        st = cache_state(dst, Path(res["config"]["out_dir"]) / "best_C.pt")
        if not force and (st == "ok" or (st == "quarantined" and not retry_quarantined)):
            if st == "quarantined":
                print(f"{res['cell_id']}: quarantined cache, not re-scored (use --retry-quarantined or --force)", flush=True)
            continue
        t0 = time.time()
        rec, per = score_cell(res, device, cache)
        rec["wall_sec"] = round(time.time() - t0, 1)
        np.savez(dst.with_suffix(f".tmp{os.getpid()}.npz"), **per)
        os.replace(dst.with_suffix(f".tmp{os.getpid()}.npz"), dst.with_suffix(".npz"))
        tmp = dst.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(R.sanitize(rec), indent=1)); os.replace(tmp, dst)
        d = max(abs(rec[sp][k]) for sp in ("dev", "test") for k in ("d_C", "d_floor_1e-6") if rec[sp][k] is not None)
        worst = max(worst, d)
        print(f"{res['cell_id']}: test jitter {rec['test']['jitter']:.5f} point_mass {rec['test']['point_mass']:.5f} "
              f"max|recomputed - stored| {d:.1e} quarantined={rec['quarantined']} ({rec['wall_sec']}s)", flush=True)
    print(f"scored {len(todo)} cells of shard {shard}; largest difference from the stored scores {worst:.2e}")


# ------------------------------------------------------------------ edits of Table S8 under both scores
@torch.no_grad()
def edit_item(ds, enc, seed, sel, device, cache):
    """Component edits (coarse_tpp.component_edits) of the three floor-trained models of one (unit, seed), scored by
    the jitter and point-mass scores; tie and non-tie sums of the unedited five models and of every edited model."""
    from . import component_edits as CE
    if ds not in cache:
        cache.clear()
        cache[ds] = (load(ds, "train"), load(ds, "test"))
    tr, te = cache[ds]
    P, models, ok = {}, {}, True
    for arm in CE.ARMS:
        res = sel[(ds, enc)][arm][seed]
        cfg = res["config"]
        stats = coarse_stats(tr, float(cfg["delta"]))
        model = build(cfg, stats, tr.coarse.n_types).to(device)
        ck = Path(cfg["out_dir"]) / "best_C.pt"
        model.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
        P[arm] = p = CE.extract(model, te, stats, cfg, device)
        st = res["by_selection"]["C"]["test"]
        e6 = float(CE.floor_score(p["lw"], p["mu"], p["ls"], p["tau"], 1e-6).mean())
        good = abs(e6 - st["floor_1e-6"]) <= CE.REPRO_TOL and len(p["tau"]) == st["n_events"] and int(p["tie"].sum()) == st["n_ties"]
        models[arm] = dict(cell_id=cfg["cell_id"], ckpt_sha256=ckpt_sha(ck), abs_diff=abs(e6 - st["floor_1e-6"]), ok=bool(good))
        ok &= good
    delta = float(sel[(ds, enc)]["floor_1e-6"][seed]["config"]["delta"])

    def sums(lw, mu, ls, tau, tie):
        law = lik.LogNormMix(lw, mu, ls)
        v = {"jitter": jitter_loglik(law, tau, delta), "point_mass": point_mass_loglik(law, None, tau, delta, "raw")}
        return {k: (float(x[tie].sum()), float(x[~tie].sum())) for k, x in v.items()}
    rec = dict(version=VERSION, dataset=ds, encoder=enc, seed=seed, delta=delta, n=int(len(P["floor_1e-6"]["tau"])),
               models=models, quarantined=not ok, base={}, edits={})
    for arm, p in P.items():
        rec["base"][arm] = sums(p["lw"], p["mu"], p["ls"], p["tau"], p["tie"])
    for sm, eps in CE.FLOORS.items():
        p = P[sm]
        rows = p["tie"]
        lw, mu, ls = p["lw"][rows], p["mu"][rows], p["ls"][rows]
        k, _, _ = CE.kstar(lw, mu, ls, eps)
        sig_nt, sig_w, _ = CE.width_values(p, k)
        var = CE.edit_variants(mu, ls, k, sig_nt, sig_w, math.log(delta))
        rec["edits"][sm] = {}
        for name in CE.PRIMARY:
            mu_e, ls_e = var[name]
            s = sums(lw, mu_e, ls_e, p["tau"][rows], torch.ones(int(rows.sum()), dtype=torch.bool))
            rec["edits"][sm][name] = {key: (v[0], rec["base"][sm][key][1]) for key, v in s.items()}
    return rec


def cmd_edits(run, out=None, cpu=False, threads=4, force=False):
    from . import component_edits as CE
    torch.set_num_threads(int(threads))
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    sel = CE.selected(run)
    items = CE.work_items(sel, CE.tied_units(sel))
    od = out_dir_of(run, out) / "edits"
    od.mkdir(parents=True, exist_ok=True)
    cache = {}
    for ds, enc, seed in items:
        dst = od / f"{CE.item_name(ds, enc, seed)}.json"
        exp = {a: ckpt_sha(Path(sel[(ds, enc)][a][seed]["config"]["out_dir"]) / "best_C.pt") for a in CE.ARMS}
        if dst.exists() and not force:
            try:
                r = json.loads(dst.read_text())
                if r.get("version") == VERSION and {a: m["ckpt_sha256"] for a, m in r["models"].items()} == exp:
                    continue
            except Exception:
                pass
        rec = edit_item(ds, enc, seed, sel, device, cache)
        tmp = dst.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(rec)); os.replace(tmp, dst)
        print(f"{dst.stem}: quarantined={rec['quarantined']}", flush=True)


# ------------------------------------------------------------------ values of a cell
class Values:
    """Test and validation values of a result under every score: floors and C from result.json, jitter and point mass
    from the score files of `score` (None when absent, stale or quarantined)."""

    def __init__(self, check_ckpt=True):
        self._files, self.check_ckpt, self.missing, self.quarantined = {}, check_ckpt, set(), set()

    def file(self, res):
        cid = res["cell_id"]
        if cid not in self._files:
            p = out_dir_of(Path(res["config"]["out_dir"]).parent.parent) / "cells" / f"{cid}.json"
            ck = Path(res["config"]["out_dir"]) / "best_C.pt" if self.check_ckpt else None
            st = cache_state(p, ck) if p.exists() else "invalid"
            self._files[cid] = json.loads(p.read_text()) if st == "ok" else None
            if st == "quarantined":
                self.quarantined.add(cid)
            elif st != "ok":
                self.missing.add(cid)
        return self._files[cid]

    def __call__(self, res, score, split):
        if score in NEW:
            j = self.file(res)
            return None if j is None else j[split][score]
        if score == "C":
            return res["by_selection"]["C"]["dev_criterion"] if split == "dev" else res["by_selection"]["C"]["test"]["C"]
        if split == "dev":
            return (res.get("dev") or {}).get(score) if res.get("primary_selection", "C") == "C" else None
        return res["by_selection"]["C"]["test"].get(score)

    def perseq(self, res, score):
        """(n, per-sequence test sums) of `score` (C and floor 1e-6 from the same pass)."""
        p = out_dir_of(Path(res["config"]["out_dir"]).parent.parent) / "cells" / f"{res['cell_id']}.npz"
        with np.load(p) as z:
            return z["n"].astype(float), z[score].astype(float)


def unit_of(res):
    c = res["config"]
    return (c["dataset"], c["encoder"]["type"], c["head"], c["K"])


# ------------------------------------------------------------------ rankings (pure; tests/test_jitter_pm.py)
def units(cells, value, score, lr_score, arms, group=unit_of):
    """{group: {arm: dict(lr, seeds={seed: test value}, mean)}}: per (group, arm) the learning rate with the best mean
    validation value of `lr_score` over seeds (ties: the larger tag), test values of `score` at that learning rate.
    value(res, score, split) -> float or None; cells with a missing value are left out."""
    by = {}
    for r in cells:
        if r["arm"] not in arms:
            continue
        dv, tv = value(r, lr_score, "dev"), value(r, score, "test")
        if dv is None or tv is None or not (math.isfinite(dv) and math.isfinite(tv)):
            continue
        by.setdefault((group(r), r["arm"]), {}).setdefault(r["config"]["lr_tag"], {})[r["seed"]] = (dv, tv)
    out = {}
    for (g, a), lrs in by.items():
        lr, seeds = max(lrs.items(), key=lambda kv: (np.mean([v[0] for v in kv[1].values()]), kv[0]))
        out.setdefault(g, {})[a] = dict(lr=lr, seeds={s: v[1] for s, v in seeds.items()},
                                        mean=float(np.mean([v[1] for v in seeds.values()])))
    return {g: v for g, v in out.items() if all(a in v for a in arms)}


def order(means, arms):
    return sorted(arms, key=lambda a: -means[a])


def kendall(x, y):
    t = sst.kendalltau(x, y)[0]
    return float(t) if t == t else float("nan")


def compare(su, iu, arms, tie_of, self_arm):
    """Per group in both: Kendall's tau of the score's ranking (su) with the interval score's (iu), the two tops, the
    deficit, seeds and rank by C of the score's top, the score's margin and whether it ranks `self_arm` first."""
    rows = []
    for g in sorted(iu):
        if g not in su:
            continue
        a = {m: su[g][m]["mean"] for m in arms}; b = {m: iu[g][m]["mean"] for m in arms}
        oa, ob = order(a, arms), order(b, arms)
        top, topi = oa[0], ob[0]
        seeds = sorted(set(iu[g][top]["seeds"]) & set(iu[g][topi]["seeds"]))
        pos = sum(iu[g][topi]["seeds"][s] - iu[g][top]["seeds"][s] > 0 for s in seeds)
        rows.append(dict(group=list(g), tie_share=tie_of(g), tau=kendall([a[m] for m in arms], [b[m] for m in arms]),
                         top=top, top_interval=topi, changed=top != topi, deficit=b[topi] - b[top],
                         rank_by_interval=ob.index(top) + 1, seeds_positive=pos, n_seeds=len(seeds),
                         self_top=(top == self_arm) if self_arm in arms else None, margin=a[oa[0]] - a[oa[1]],
                         means=a))
    return rows


def summarize(rows, tied_min=TIE_MIN):
    tied = [r for r in rows if r["tie_share"] >= tied_min]
    ch = [r for r in rows if r["changed"]]; cht = [r for r in tied if r["changed"]]
    med = lambda v: float(np.median(v)) if v else None
    has_self = all(r["self_top"] is not None for r in rows)
    return dict(n=len(rows), n_tied=len(tied), changed=len(ch), changed_tied=len(cht),
                tau_median=med([r["tau"] for r in rows]),
                self_top=sum(bool(r["self_top"]) for r in rows) if has_self else None,
                self_top_tied=sum(bool(r["self_top"]) for r in tied) if has_self else None,
                deficit_changed_tied_max=max((r["deficit"] for r in cht), default=None),
                deficit_changed_max=max((r["deficit"] for r in ch), default=None),
                ranks_changed_tied=[r["rank_by_interval"] for r in cht],
                last_by_interval_changed=sum(r["rank_by_interval"] == len(r["means"]) for r in ch))


def score_units(cells, value, arms, floor="floor_1e-6", group=unit_of):
    """{score: units} under the report's learning-rate rules (floor and C: validation C; jitter, point mass: own)."""
    return {floor: units(cells, value, floor, "C", arms, group), "C": units(cells, value, "C", "C", arms, group),
            "jitter": units(cells, value, "jitter", "jitter", arms, group),
            "point_mass": units(cells, value, "point_mass", "point_mass", arms, group)}


# ------------------------------------------------------------------ bootstrap (Table S6 columns)
def bootstrap_score(n, X_, C, self_arm, arms, B=B_DEFAULT, seed=BOOT_SEED):
    """Two-stage bootstrap of coarse_tpp.rank_bootstrap (same generator and draw order) for a score X_ against C:
    p_diff = share of draws whose tops differ, p_self = share whose top by X_ is `self_arm`."""
    from .rank_bootstrap import scores
    top = lambda s: max(s, key=lambda a: (s[a], -arms.index(a)))
    rng = np.random.default_rng(seed)
    M, S = len(n), next(iter(C.values())).shape[0]
    diff = selfp = 0
    for _ in range(B):
        w = np.bincount(rng.integers(0, M, M), minlength=M).astype(float)
        si = rng.integers(0, S, S)
        sx, sc = scores(n, X_, w, si), scores(n, C, w, si)
        tx = top(sx)
        diff += tx != top(sc)
        selfp += tx == self_arm
    return dict(p_diff=diff / B, p_self=selfp / B)


# ------------------------------------------------------------------ Table 6 (pure)
T6_SCORES = ("floor_1e-10", "floor_1e-6", "floor_1e-5", "C", "jitter", "point_mass")


def reference_rows(settings, arms, scores=T6_SCORES):
    """Per setting {arm: {score: mean test value, 'B': mean B}}: Kendall's tau with the ranking by B, the score's top,
    whether it is the top by B, its shortfall in B, whether it is the model trained with the score."""
    out = []
    for key in sorted(settings):
        d = settings[key]
        if not all(a in d and all(d[a].get(s) is not None for s in scores + ("B",)) for a in arms):
            continue
        B = [d[a]["B"] for a in arms]
        row = dict(setting=list(key), top_B=arms[int(np.argmax(B))])
        for sc in scores:
            x = [d[a][sc] for a in arms]
            i = int(np.argmax(x))
            row[sc] = dict(tau=float(sst.kendalltau(x, B)[0]), top=arms[i], shortfall=float(max(B) - B[i]))
        out.append(row)
    return out


def reference_summary(rows, scores=T6_SCORES, arms=X.COARSEN_ARMS):
    out = dict(n=len(rows))
    for sc in scores:
        t, f = [r[sc]["tau"] for r in rows], [r[sc]["shortfall"] for r in rows]
        own = SELF[sc] if SELF[sc] in arms else None
        out[sc] = dict(median_tau=float(np.median(t)) if t else None, top_as_B=sum(r[sc]["top"] == r["top_B"] for r in rows),
                       within=sum(x <= WITHIN for x in f), mean_shortfall=float(np.mean(f)) if f else None,
                       self_top=sum(r[sc]["top"] == own for r in rows) if own else None,
                       tau_below_C=sum(r[sc]["tau"] < r["C"]["tau"] for r in rows))
    return out


def syn_settings(cells, value, scores=T6_SCORES, arms=X.COARSEN_ARMS):
    """{(process, delta): {arm: {score: mean}}} at the validation-C learning rate (coarse_tpp.analysis.select_lr): per
    realization the seed mean, then the mean over realizations (settings whose treatments share the realizations)."""
    acc = {}
    for (ds, enc, head, K, arm), seeds in A.select_lr(cells).items():
        p = Syn.parse(ds)
        if p is None or p[2] == 0 or arm not in arms:
            continue
        m = {}
        for sc in scores + ("B",):
            v = [value(x, sc, "test") if sc != "B" else x["by_selection"]["C"]["test"].get("B") for x in seeds.values()]
            m[sc] = float(np.mean(v)) if v and all(y is not None for y in v) else None
        acc.setdefault((p[0], p[2]), {}).setdefault(arm, {})[p[1]] = m
    out = {}
    for k, v in acc.items():
        if len({tuple(sorted(r)) for r in v.values()}) != 1:
            continue
        out[k] = {a: {sc: (float(np.mean([m[sc] for m in rs.values()])) if all(m[sc] is not None for m in rs.values())
                           else None) for sc in scores + ("B",)} for a, rs in v.items()}
    return out


def coarsened_settings(cells, value, latent, scores=T6_SCORES, arms=X.COARSEN_ARMS, keep=lambda ds: True):
    """{(dataset, encoder): {arm: {score: mean over seeds}}} at the validation-C learning rate; B from `latent`
    (cell_id -> test B; coarse_tpp.latent_score)."""
    out = {}
    for (ds, enc, head, K, arm), seeds in A.select_lr(cells).items():
        if not keep(ds) or arm not in arms or not all(x["cell_id"] in latent for x in seeds.values()):
            continue
        m = {}
        for sc in scores:
            v = [value(x, sc, "test") for x in seeds.values()]
            m[sc] = float(np.mean(v)) if all(y is not None for y in v) else None
        m["B"] = float(np.mean([latent[x["cell_id"]] for x in seeds.values()]))
        out.setdefault((ds, enc), {})[arm] = m
    return out


# ------------------------------------------------------------------ decomposition (pure)
def decomposition(cells, value_file, arms=None, tie_of=None, group=unit_of):
    """Per group at the validation-C learning rate: seed means of C, jitter, point mass, KL_C = C - log(delta) - jitter
    and the tie / positive-gap parts of the point-mass score (n_tie mean_tie / n and n_pos mean_pos / n).
    value_file(res) -> the score file of the cell (test split used)."""
    arms = arms or RANK_ARMS
    acc = {}
    for (ds, enc, head, K, arm), seeds in A.select_lr(cells).items():
        if arm not in arms:
            continue
        rows = []
        for x in seeds.values():
            j = value_file(x)
            if j is None:
                rows = None
                break
            t = j["test"]
            n, nt = t["n_events"], t["n_ties"]
            rows.append(dict(C=t["C"], J=t["jitter"], pm=t["point_mass"], KL_C=t["C"] - t["log_delta"] - t["jitter"],
                             pm_tie=(nt * t["point_mass_tie"] / n) if nt else 0.0,
                             pm_pos=((n - nt) * t["point_mass_nontie"] / n) if n > nt else 0.0))
        if rows:
            acc.setdefault((ds, enc, head, K), {})[arm] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    out = []
    for g, m in sorted(acc.items()):
        if not all(a in m for a in arms):
            continue
        topi = max(arms, key=lambda a: m[a]["C"]); topp = max(arms, key=lambda a: m[a]["pm"])
        out.append(dict(group=list(g), tie_share=tie_of(g) if tie_of else None, top_interval=topi,
                        top_jitter=max(arms, key=lambda a: m[a]["J"]), top_point_mass=topp,
                        KL_C_smallest=min(arms, key=lambda a: m[a]["KL_C"]),
                        pm_d_tie=m[topp]["pm_tie"] - m[topi]["pm_tie"], pm_d_pos=m[topp]["pm_pos"] - m[topi]["pm_pos"],
                        means=m))
    return out


def edits_summary(recs):
    """Seed means of the five models and of the edited floor-trained models; per score, the number of (unit, floor
    model) pairs whose model (unedited or edited) ranks first among the five."""
    from . import component_edits as CE
    by = {}
    for r in recs:
        by.setdefault((r["dataset"], r["encoder"]), []).append(r)
    units_, first = {}, {s: {e: 0 for e in ("none",) + CE.PRIMARY} for s in NEW}
    for u, rs in sorted(by.items()):
        S = lambda src, key: float(np.mean([(src(r)[key][0] + src(r)[key][1]) / r["n"] for r in rs]))
        base = {a: {k: S(lambda r, a=a: r["base"][a], k) for k in NEW} for a in CE.ARMS}
        row = {}
        for sc in NEW:
            row[sc] = {}
            for sm in CE.FLOORS:
                other = max(base[a][sc] for a in CE.ARMS if a != sm)
                vals = {"none": base[sm][sc]}
                vals.update({e: S(lambda r, sm=sm, e=e: r["edits"][sm][e], sc) for e in CE.PRIMARY})
                row[sc][sm] = {e: dict(score=v, margin=v - other, first=v > other) for e, v in vals.items()}
                for e, v in vals.items():
                    first[sc][e] += v > other
        units_[f"{u[0]}|{u[1]}"] = dict(seeds=sorted(r["seed"] for r in rs), scores=row)
    return dict(units=units_, first=first, n_pairs=len(units_) * len(CE.FLOORS))


# ------------------------------------------------------------------ selections over (treatment, lr) (pure)
def choose(cands, score_of, keys, seeds):
    """Candidate key with the highest mean over `seeds` of score_of(result); None if a value is missing (ties: sorted
    key order, as coarse_tpp.alarms.choose)."""
    means = {}
    for k in keys:
        v = [score_of(cands[k][s]) for s in seeds] if k in cands and all(s in cands[k] for s in seeds) else [None]
        if any(x is None or not math.isfinite(x) for x in v):
            return None, means
        means[k] = float(np.mean(v))
    return max(sorted(means), key=lambda k: means[k]), means


def brier_comparison(load, key_score, key_interval, boot):
    """Difference interval selection - selection by the score of the test Brier loss (seed means of per-event means
    over the horizons) with the case bootstrap `boot` (coarse_tpp.alarms.boot_diff). load(key) -> (n, [per-sequence
    Brier sums over horizons, averaged, per seed]) or None."""
    if key_score == key_interval:
        return dict(same=True)
    a, b = load(key_score), load(key_interval)
    if a is None or b is None:
        return None
    n, A_ = a
    _, B_ = b
    pt, lo, hi = boot(A_, B_, n)
    bs = float(np.mean([x.sum() / n.sum() for x in A_])); bi = float(np.mean([x.sum() / n.sum() for x in B_]))
    return dict(same=False, diff=float(pt), boot95=[float(lo), float(hi)], brier_score=bs, brier_interval=bi,
                rel=(bs - bi) / bi)


# ------------------------------------------------------------------ predictive accuracy (pure)
def predictive_rows(groups, arms=None, metrics=("twcrps", "mae_median")):
    """groups {(dataset, encoder): {arm: {seed: {score...: test value, metric...: value}}}} -> per group the top
    treatment of each score, the seed-paired difference (interval score's choice - the score's choice) of each metric in
    percent of the score's choice with the one-sided 95% upper bound, and Kendall's tau of the score's ranking with
    the ranking by -twCRPS."""
    from .predictive import kendall as ktau, t_upper
    arms = arms or RANK_ARMS
    out = {}
    for g, A_ in sorted(groups.items()):
        mean = lambda a, k: float(np.mean([x[k] for x in A_[a].values()]))
        top = {s: max(arms, key=lambda a: mean(a, s)) for s in ("floor_1e-6", "C") + NEW}
        row = dict(top=top)
        for s in ("floor_1e-6",) + NEW:
            ts, ti = top[s], top["C"]
            r = dict(changed=ts != ti)
            for k in metrics:
                seeds = sorted(set(A_[ts]) & set(A_[ti]))
                d = [A_[ti][x][k] - A_[ts][x][k] for x in seeds]
                m, up = t_upper(d) if ts != ti else (0.0, 0.0)
                base = mean(ts, k)
                r[k] = dict(pct=100 * m / base, up95_pct=100 * up / base, d=m)
            r["tau_twcrps"] = ktau([mean(a, s) for a in arms], [-mean(a, "twcrps") for a in arms])
            row[s] = r
        row["C"] = dict(tau_twcrps=ktau([mean(a, "C") for a in arms], [-mean(a, "twcrps") for a in arms]))
        out[g] = row
    return out


# ------------------------------------------------------------------ report
def f(x, d=3, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def tie_fn(audit, cells):
    """Tie share of a group: the audit's share of the data set when available, else the mean stored test share."""
    ts = {}
    for r in cells:
        ts.setdefault(r["config"]["dataset"], []).append(r["by_selection"]["C"]["test"].get("tie_share", 0.0))
    return lambda g: audit[g[0]]["tie_share"] if g[0] in audit else float(np.mean(ts.get(g[0], [0.0])))


def ranking_block(cells, value, arms, tie_of, floor="floor_1e-6", group=unit_of):
    U = score_units(cells, value, arms, floor, group)
    out = {sc: compare(U[sc], U["C"], arms, tie_of, SELF[sc]) for sc in U}
    return U, out, {sc: summarize(rows) for sc, rows in out.items()}


def ranking_lines(title, summ, floor):
    L = [f"### {title}", "", "| score | combinations (tied) | top differs from interval (tied) | self first (tied) | "
         "median tau with interval | largest deficit where changed | changed tied: rank by interval |",
         "|---|---|---|---|---:|---:|---|"]
    for sc in (floor, "C") + NEW:
        s = summ[sc]
        st = "—" if s["self_top"] is None else f"{s['self_top']} ({s['self_top_tied']})"
        L.append(f"| {LABEL[sc]} | {s['n']} ({s['n_tied']}) | {s['changed']} ({s['changed_tied']}) | {st} | "
                 f"{f(s['tau_median'], 2)} | {f(s['deficit_changed_max'], 4, False)} | {s['ranks_changed_tied']} |")
    return L + [""]


def gname(g):
    return " ".join(str(x) for x in g if x not in ("mix", 8))


def section_rank_change(run, value, audit, B):
    cells = A.load_cells(Path(run))
    tie_of = tie_fn(audit, cells)
    arms = RANK_ARMS
    U, rows, summ = ranking_block(cells, value, arms, tie_of)
    L = ["## Rankings of the eight treatments (rank_change; Table S5 columns)", ""] + ranking_lines("Summary", summ, "floor_1e-6")
    L += ["Combinations with tied data: top treatment, deficit, seeds and rank by the interval score (Table S5).", "",
          "| combination | ties | interval top | floor 1e-6 top | jitter top: deficit, seeds, rank | point-mass top: deficit, seeds, rank |",
          "|---|---:|---|---|---|---|"]
    by = {sc: {tuple(r["group"]): r for r in rows[sc]} for sc in rows}
    cell = lambda r: (f"{r['top']} (same)" if not r["changed"] else
                      f"{r['top']}, {r['deficit']:.4f}, {r['seeds_positive']}/{r['n_seeds']}, {r['rank_by_interval']}")
    for g in sorted(by["C"], key=lambda g: -tie_of(g)):
        if tie_of(g) < TIE_MIN:
            continue
        L.append(f"| {gname(g)} | {tie_of(g):.1%} | {by['C'][g]['top']} | {by['floor_1e-6'][g]['top']} | {cell(by['jitter'][g])} | "
                 f"{cell(by['point_mass'][g])} |")
    # bootstrap (Table S6 columns)
    L += ["", "### Bootstrap (Table S6 columns)", "",
          f"{B} two-stage draws (seed {BOOT_SEED}) of test sequences and seeds, the design of coarse_tpp.rank_bootstrap; "
          "p_diff: share of draws whose tops by the score and by C differ; p_self: share whose top by the score is the model "
          "trained with it. Learning rates as above; C from perseq_C.npz (the stored per-sequence sums).", "",
          "| combination | ties | jitter p_diff | jitter p_self | point mass p_diff | point mass p_self |", "|---|---:|---:|---:|---:|---:|"]
    boot, bad = {}, []
    lookup = {(unit_of(r), r["arm"], r["config"]["lr_tag"], r["seed"]): r for r in cells}
    for g in sorted(U["C"], key=lambda g: -tie_of(g)):
        seeds = sorted(U["C"][g]["interval"]["seeds"])
        try:
            Cc, n_ref = {}, None
            for a in arms:
                ys = []
                for s in seeds:
                    r = lookup[(g, a, U["C"][g][a]["lr"], s)]
                    with np.load(Path(r["config"]["out_dir"]) / "perseq_C.npz") as z:
                        n, c = z["n"].astype(float), z["C"].astype(float)
                    n_ref = n if n_ref is None else n_ref
                    if not np.array_equal(n, n_ref):
                        raise ValueError("per-sequence target counts differ")
                    ys.append(c)
                Cc[a] = np.stack(ys)
            row = {}
            for sc in NEW:
                Xs = {}
                for a in arms:
                    ys = []
                    for s in seeds:
                        r = lookup[(g, a, U[sc][g][a]["lr"], s)]
                        if value.file(r) is None:
                            raise ValueError(f"{r['cell_id']}: no valid score file")
                        n, y = value.perseq(r, sc)
                        if not np.array_equal(n, n_ref):
                            raise ValueError("per-sequence target counts differ from perseq_C.npz")
                        ys.append(y)
                    Xs[a] = np.stack(ys)
                row[sc] = bootstrap_score(n_ref, Xs, Cc, SELF[sc], list(arms), B)
        except (KeyError, ValueError, OSError) as e:
            bad.append(f"{gname(g)}: {e}")
            continue
        boot[g] = row
        L.append(f"| {gname(g)} | {tie_of(g):.1%} | {row['jitter']['p_diff']:.3f} | {row['jitter']['p_self']:.3f} | "
                 f"{row['point_mass']['p_diff']:.3f} | {row['point_mass']['p_self']:.3f} |")
    tied = [g for g in boot if tie_of(g) >= TIE_MIN]
    bsum = {sc: dict(changed_tied=sum(by[sc][g]["changed"] for g in tied),
                     changed_tied_p_diff_ge=sum(by[sc][g]["changed"] and boot[g][sc]["p_diff"] >= P_MIN for g in tied),
                     tied_p_self_ge=sum(boot[g][sc]["p_self"] >= P_MIN for g in tied), n_tied=len(tied)) for sc in NEW}
    L += [""] + [f"- {LABEL[sc]}: tied combinations whose top changed {v['changed_tied']}, of which p_diff >= {P_MIN} in "
                 f"{v['changed_tied_p_diff_ge']}; p_self >= {P_MIN} in {v['tied_p_self_ge']} of {v['n_tied']} tied." for sc, v in bsum.items()]
    if bad:
        L += ["", "Left out of the bootstrap: " + "; ".join(bad)]
    summary = dict(summary=summ, rows={sc: rows[sc] for sc in rows}, bootstrap={"|".join(map(str, g)): v for g, v in boot.items()},
                   bootstrap_summary=bsum, bootstrap_left_out=bad)
    return L + [""], summary, (cells, U, tie_of)


ROBUST = (("rank_change_seeds", "ten seeds", X.RANK_ARMS, "floor_1e-6"),
          ("wider_encoders", "width 128", X.WIDE_ARMS, "floor_1e-10"),
          ("mixture_components", "1-16 components", X.WIDE_ARMS, "floor_1e-10"),
          ("density_heads", "Weibull-mixture and spline-flow heads", X.HEADS_ARMS, "floor_1e-6"),
          ("resplit", "new splits", X.RESPLIT_ARMS, "floor_1e-6"),
          ("new_datasets", "further data sets", X.RESPLIT_ARMS, "floor_1e-6"),
          ("process_logs", "process logs", X.PROCESS_ARMS, "floor_1e-6"))


def section_robust(runs, value, audit, rank_change=None):
    L = ["## Robustness checks with the jitter and point-mass scores", "",
         "Floor of each check as in the paper (1e-10 for width 128 and the number of components). Self (jitter): the "
         "jitter score ranks the gap-jitter model first.", ""]
    out, tot = {}, dict(n=0, n_tied=0, self=0, self_tied=0)
    for name, title, arms, floor in ROBUST:
        if name not in runs:
            continue
        cells = A.load_cells(Path(runs[name]))
        if name == "rank_change_seeds" and rank_change:
            cells = A.load_cells(Path(rank_change)) + cells
        elif name == "density_heads":
            cells = [r for r in cells if r["config"]["head"] in X.DENSITY_HEADS]
        tie_of = tie_fn(audit, cells)
        U, rows, summ = ranking_block(cells, value, arms, tie_of, floor)
        L += ranking_lines(f"{title} ({name})", summ, floor)
        out[name] = dict(summary=summ, rows=rows)
        s = summ["jitter"]
        tot["n"] += s["n"]; tot["n_tied"] += s["n_tied"]; tot["self"] += s["self_top"]; tot["self_tied"] += s["self_top_tied"]
    L += [f"Over these checks the jitter score ranks the gap-jitter model first in {tot['self_tied']} of {tot['n_tied']} tied "
          f"combinations ({tot['self']} of {tot['n']} in all).", ""]
    out["jitter_self_total"] = tot
    # Table S7 columns: best by jitter and by point mass, density heads and the log-normal mixture (four treatments)
    if "density_heads" in runs:
        cells = [r for r in A.load_cells(Path(runs["density_heads"])) if r["config"]["head"] in X.DENSITY_HEADS]
        if rank_change:
            cells += [r for r in A.load_cells(Path(rank_change)) if r["config"]["encoder"]["type"] == "gru"
                      and r["config"]["dataset"] in X.HEADS_DATASETS]
        heads = lambda r: (r["config"]["dataset"], r["config"]["head"])
        U = score_units(cells, value, X.HEADS_ARMS, "floor_1e-6", heads)
        L += ["### Table S7 columns: best treatment by each score (GRU, four treatments)", "",
              "| data set | head | floor 1e-6 | interval | jitter | point mass |", "|---|---|---|---|---|---|"]
        t7 = {}
        for g in sorted(U["C"]):
            tops = {sc: order({a: U[sc][g][a]["mean"] for a in X.HEADS_ARMS}, X.HEADS_ARMS)[0] for sc in U if g in U[sc]}
            t7["|".join(g)] = tops
            same = lambda sc: " (same)" if tops.get(sc) == tops["C"] else ""
            L.append(f"| {g[0]} | {g[1]} | {tops.get('floor_1e-6')} | {tops['C']} | {tops.get('jitter')}{same('jitter')} | "
                     f"{tops.get('point_mass')}{same('point_mass')} |")
        out["table_s7"] = t7
        L.append("")
    return L, out


def section_table6(runs, value, audit):
    L = ["## Table 6 rows: rankings against the latent-time score B", "",
         "Six treatments; every score at the learning rate chosen by the validation C (as in Table 6). Within: the "
         f"score's first choice within {WITHIN} nats of the best B; self: the score ranks the model trained with it first; "
         "tau below C: settings where the score's tau is below the interval score's.", "",
         "| data (settings) | score | median tau | top as by B | within | mean deficit in B | self | tau below C |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    blocks, out = [], {}
    if "synthetic" in runs:
        ss = syn_settings(A.load_cells(Path(runs["synthetic"])), value)
        main = set(Syn.main_settings())
        blocks += [("synthetic, pre-specified", {k: v for k, v in ss.items() if k in main}), ("synthetic, all", ss)]
    for name in ("coarsened_real", "coarsened_more"):
        if name in runs:
            p = Path(runs[name]) / "latent_scores.jsonl"
            if not p.exists():
                L.append(f"| {name} | latent-time scores missing (run coarse_tpp.latent_score) | | | | | | |")
                continue
            rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            lat = {r["cell_id"]: r["test"]["B"] for r in rows if not r.get("mismatch") and A.latent_identity_current(r)}
            blocks.append((name, coarsened_settings(A.load_cells(Path(runs[name])), value, lat)))
    for label, settings in blocks:
        rows = reference_rows(settings, X.COARSEN_ARMS)
        sm = reference_summary(rows)
        out[label] = dict(summary=sm, rows=rows)
        for sc in T6_SCORES:
            x = sm[sc]
            L.append(f"| {label} ({sm['n']}) | {LABEL[sc]} | {f(x['median_tau'], 2)} | {x['top_as_B']} | {x['within']} | "
                     f"{f(x['mean_shortfall'], 3, False)} | {'—' if x['self_top'] is None else x['self_top']} | "
                     f"{'—' if sc == 'C' else x['tau_below_C']} |")
    return L + [""], out


def section_decomposition(rc, value, run, audit):
    cells, U, tie_of = rc
    dec = decomposition(cells, value.file, tie_of=tie_of)
    tied = [r for r in dec if r["tie_share"] >= TIE_MIN]
    pmc = [r for r in tied if r["top_point_mass"] != r["top_interval"]]
    out = dict(rows=dec, n_tied=len(tied), jitter_gap_smallest_KL_C=sum(r["KL_C_smallest"] == "jitter_gap" for r in tied),
               pm_changed_tied=len(pmc), pm_positive_part_dominates=sum(abs(r["pm_d_pos"]) > abs(r["pm_d_tie"]) for r in pmc))
    L = ["## Decomposition (rank_change, validation-C learning rate)", "",
         "KL_C = C - log(delta) - jitter (Proposition 1 for the raw-density treatments); point mass = tie part + positive-gap "
         "part (n_tie / n x tie mean + n_pos / n x positive-gap mean).", "",
         "| combination | ties | smallest KL_C | interval top | point-mass top | point-mass difference: tie part / positive-gap part |",
         "|---|---:|---|---|---|---|"]
    for r in sorted(tied, key=lambda r: -r["tie_share"]):
        L.append(f"| {gname(r['group'])} | {r['tie_share']:.1%} | {r['KL_C_smallest']} | {r['top_interval']} | {r['top_point_mass']} | "
                 + ("same" if r["top_point_mass"] == r["top_interval"] else f"{r['pm_d_tie']:+.4f} / {r['pm_d_pos']:+.4f}") + " |")
    L += ["", f"- The gap-jitter model has the smallest KL_C in {out['jitter_gap_smallest_KL_C']} of {len(tied)} tied combinations.",
          f"- Where the point-mass top differs from the interval top ({len(pmc)} tied combinations), the positive-gap part "
          f"carries the larger share of the difference in {out['pm_positive_part_dominates']}."]
    ed = Path(run) / "jitter_pm" / "edits"
    recs = []
    if ed.exists():
        for p in sorted(ed.glob("*.json")):
            r = json.loads(p.read_text())
            if r.get("version") == VERSION and not r.get("quarantined"):
                recs.append(r)
    if recs:
        es = edits_summary(recs)
        out["edits"] = es
        L += ["", f"- Table S8 edits ({es['n_pairs']} floor-trained models): models ranked first among the five by the jitter score: "
              + ", ".join(f"{e} {v}" for e, v in es["first"]["jitter"].items()) + "; by the point-mass score: "
              + ", ".join(f"{e} {v}" for e, v in es["first"]["point_mass"].items()) + "."]
    else:
        L += ["", "- Table S8 edits: run `python -m coarse_tpp.jitter_pm edits RUN_DIR` to add them."]
    return L + [""], out


def section_brier(run, value):
    from . import alarms as AL
    cells, _ = AL.load_results(run)
    cells = [r for r in cells if r["arm"] in AL.BASE_ARMS]
    cands = AL.candidates(cells)
    out_dir = Path(run) / "alarms"
    keys = AL.expected_keys(False)
    rules = {"floor_1e-6": lambda r: value(r, "floor_1e-6", "dev"), "C": lambda r: value(r, "C", "dev"),
             "jitter": lambda r: value(r, "jitter", "dev"), "point_mass": lambda r: value(r, "point_mass", "dev")}

    def loader(u):
        def load_(key):
            zs = []
            for s in AL.PLANNED_SEEDS:
                r = cands[u][key][s]
                if AL.verification(out_dir, r["cell_id"], "C", r) != "verified":
                    return None
                zs.append(dict(np.load(out_dir / "cells" / f"{r['cell_id']}__C.npz")))
            return zs[0]["n"], [z["brier"].mean(1) for z in zs]
        return load_
    L = ["## Threshold-exceedance Brier on the process logs (Table S11 columns)", "",
         "Selection over (treatment, learning rate) by the validation seed mean of each score (best_C.pt); test Brier "
         "averaged over the four horizons and five seeds (coarse_tpp.alarms); difference = interval selection - selection by "
         f"the score (x 1e3) with the case-bootstrap 95% interval ({AL.N_BOOT} resamples, seed {AL.BOOT_SEED}); relative = "
         "(score selection - interval selection) / interval selection.", "",
         "| log | encoder | interval | floor 1e-6 | jitter | jitter: diff [95%], relative | point mass | point mass: diff [95%], relative |",
         "|---|---|---|---|---|---|---|---|"]
    out = {}
    for u in sorted(cands, key=lambda x: (x[0] not in AL.PRIMARY_DS, x)):
        ch = {n: choose(cands[u], fn, keys, AL.PLANNED_SEEDS)[0] for n, fn in rules.items()}
        if any(v is None for v in ch.values()):
            L.append(f"| {u[0]} | {u[1]} | pending | | | | | |")
            continue
        comp = {sc: brier_comparison(loader(u), ch[sc], ch["C"], lambda a, b, n: AL.boot_diff(a, b, n)) for sc in ("floor_1e-6",) + NEW}
        out[f"{u[0]}|{u[1]}"] = dict(choice={k: list(v) for k, v in ch.items()}, comparison=comp)
        lab = lambda k: f"{k[0]} {k[1]}"
        cc = lambda c: ("same" if c and c["same"] else "pending" if c is None else
                        f"{1e3 * c['diff']:+.1f} [{1e3 * c['boot95'][0]:+.1f}, {1e3 * c['boot95'][1]:+.1f}], {100 * c['rel']:+.2f}%")
        L.append(f"| {u[0]} | {u[1]} | {lab(ch['C'])} | {lab(ch['floor_1e-6'])} | {lab(ch['jitter'])} | {cc(comp['jitter'])} | "
                 f"{lab(ch['point_mass'])} | {cc(comp['point_mass'])} |")
    rng = {}
    for sc in ("floor_1e-6",) + NEW:
        rel = [v["comparison"][sc]["rel"] for v in out.values() if v["comparison"][sc] and not v["comparison"][sc]["same"]]
        rng[sc] = dict(n=len(rel), min=min(rel) if rel else None, max=max(rel) if rel else None)
    L += [""] + [f"- {LABEL[sc]}: {v['n']} selections differ from the interval selection"
                 + (f"; relative difference {100 * v['min']:+.2f}% to {100 * v['max']:+.2f}%." if v["n"] else ".")
                 for sc, v in rng.items()]
    return L + [""], dict(units=out, ranges=rng)


def section_predictive(rc, value, run):
    cells, U, tie_of = rc
    sel = {}
    for (ds, enc, head, K, arm), seeds in A.select_lr(cells).items():
        sel.setdefault((ds, enc), {})[arm] = seeds
    groups, missing = {}, []
    for g, arms in sel.items():
        if tie_of(g) < TIE_MIN or g[0] in X.RANK_CONTROLS or set(arms) != set(RANK_ARMS):
            continue
        G = {}
        for arm, seeds in arms.items():
            for s, r in seeds.items():
                p = Path(run) / "predictive" / "cells" / f"{r['cell_id']}.json"
                pr = json.loads(p.read_text()) if p.exists() else None
                ident = R.checkpoint_identity(r["config"], r, "C")
                vals = {sc: value(r, sc, "test") for sc in ("floor_1e-6", "C") + NEW}
                if pr is None or pr.get("quarantined") or ident is None or pr.get("identity") != ident or \
                        any(v is None for v in vals.values()):
                    missing.append(r["cell_id"]); continue
                G.setdefault(arm, {})[s] = dict(vals, twcrps=pr["twcrps"], mae_median=pr["mae_median"])
        groups[g] = G
    complete = {g: G for g, G in groups.items() if set(G) == set(RANK_ARMS)
                and len({tuple(sorted(v)) for v in G.values()}) == 1}
    pr = predictive_rows(complete)
    L = ["## Predictive accuracy of the top treatment of each score (Table S13 columns)", "",
         "Tied combinations, models at the validation-C learning rate (those of coarse_tpp.predictive). Difference: interval "
         "score's choice minus the score's choice, in percent of the score's choice [one-sided 95% upper bound]; tau: Kendall's "
         "tau between the score's ranking and the ranking by twCRPS.", "",
         "| combination | interval top | floor 1e-6: twCRPS, MAE / tau | jitter: top, twCRPS, MAE / tau | point mass: top, twCRPS, MAE / tau | interval tau |",
         "|---|---|---|---|---|---:|"]
    cell = lambda r, s: (f"{r['top'][s]} (same) / {f(r[s]['tau_twcrps'], 2)}" if not r[s]["changed"] else
                         f"{r['top'][s]}, {r[s]['twcrps']['pct']:+.2f}% [{r[s]['twcrps']['up95_pct']:+.2f}%], "
                         f"{r[s]['mae_median']['pct']:+.2f}% [{r[s]['mae_median']['up95_pct']:+.2f}%] / {f(r[s]['tau_twcrps'], 2)}")
    for g, r in pr.items():
        L.append(f"| {g[0]} {g[1]} | {r['top']['C']} | {cell(r, 'floor_1e-6')} | {cell(r, 'jitter')} | {cell(r, 'point_mass')} | "
                 f"{f(r['C']['tau_twcrps'], 2)} |")
    summ = {}
    for s in ("floor_1e-6",) + NEW:
        ch = [r[s] for r in pr.values() if r[s]["changed"]]
        summ[s] = dict(n_changed=len(ch), twcrps_min=min((x["twcrps"]["pct"] for x in ch), default=None),
                       twcrps_max=max((x["twcrps"]["pct"] for x in ch), default=None),
                       mae_min=min((x["mae_median"]["pct"] for x in ch), default=None),
                       mae_max=max((x["mae_median"]["pct"] for x in ch), default=None),
                       tau_median=float(np.median([r[s]["tau_twcrps"] for r in pr.values()])) if pr else None)
    summ["C"] = dict(tau_median=float(np.median([r["C"]["tau_twcrps"] for r in pr.values()])) if pr else None)
    L += [""] + [f"- {LABEL[s]}: {v['n_changed']} changed"
                 + (f"; twCRPS {v['twcrps_min']:+.2f}% to {v['twcrps_max']:+.2f}%, MAE {v['mae_min']:+.2f}% to {v['mae_max']:+.2f}%"
                    if v["n_changed"] else "") + f"; median tau with twCRPS {f(v['tau_median'], 2)}."
                 for s, v in summ.items() if s != "C"]
    L.append(f"- interval: median tau with twCRPS {f(summ['C']['tau_median'], 2)}.")
    if missing:
        L += ["", f"Pending (no current predictive or score file): {len(missing)} cells, e.g. {', '.join(missing[:5])}."]
    return L + [""], dict(rows={"|".join(g): v for g, v in pr.items()}, summary=summ, pending=len(missing))


def cmd_report(a):
    audit = A.audit_rows(a.audit)
    value = Values()
    runs = {k: getattr(a, k) for k in ("rank_change_seeds", "wider_encoders", "mixture_components", "density_heads", "resplit",
                                       "new_datasets", "process_logs", "synthetic", "coarsened_real", "coarsened_more")
            if getattr(a, k)}
    L = ["# Jitter and point-mass scores (Online Resource 1, Section S5; no training)", "",
         f"Jitter: {N_NODES}-node Gauss-Legendre E_u log q(tau~ + delta u); point mass: tie term of C at ties, density of the "
         "model's distribution at positive gaps (log(1 - pi) + log q for the point-mass treatment). Learning rates: floor 1e-6 "
         "and C by the validation C, jitter and point mass by their own validation means, unless a section states otherwise.", ""]
    S = {}
    rc = None
    if a.rank_change:
        Lr, S["rank_change"], rc = section_rank_change(a.rank_change, value, audit, a.B)
        L += Lr
    if runs.keys() & {n for n, *_ in ROBUST}:
        Lr, S["robustness"] = section_robust(runs, value, audit, a.rank_change)
        L += Lr
    if runs.keys() & {"synthetic", "coarsened_real", "coarsened_more"}:
        Lr, S["table6"] = section_table6(runs, value, audit)
        L += Lr
    if rc:
        Lr, S["decomposition"] = section_decomposition(rc, value, a.rank_change, audit)
        L += Lr
    if a.intensity_scores and a.intensity_jitter:
        from . import intensity_jitter as IJ
        Lr, S["intensity"] = IJ.section(a.intensity_scores, a.intensity_jitter, exceedance=a.exceedance,
                                        repro_dirs=a.intensity_repro or ())
        L += Lr
    if "process_logs" in runs and (Path(runs["process_logs"]) / "alarms").exists():
        Lr, S["brier"] = section_brier(runs["process_logs"], value)
        L += Lr
    if rc and (Path(a.rank_change) / "predictive" / "cells").exists():
        Lr, S["predictive"] = section_predictive(rc, value, a.rank_change)
        L += Lr
    if value.missing or value.quarantined:
        L += ["## Cells without a valid score file", "",
              f"Missing or stale: {len(value.missing)}; quarantined: {len(value.quarantined)}. These cells are left out of "
              "the comparisons that need them (a combination is complete only with every treatment)."]
    S["cells_without_scores"] = dict(missing=sorted(value.missing), quarantined=sorted(value.quarantined))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "summary.json").write_text(json.dumps(jsonable(S), indent=1))
    return L


def jsonable(x):
    if isinstance(x, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="score the checkpoints of a run")
    s.add_argument("run")
    s.add_argument("--shard", default="0/1", help="i/n: every n-th data set starting at i")
    s.add_argument("--cpu", action="store_true")
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--arms", nargs="*", default=None, help="score only these treatments")
    s.add_argument("--only", nargs="*", default=None, help="score only these cell ids")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--force", action="store_true")
    s.add_argument("--retry-quarantined", action="store_true")
    e = sub.add_parser("edits", help="component edits of Table S8 under both scores (rank_change run)")
    e.add_argument("run")
    e.add_argument("--cpu", action="store_true")
    e.add_argument("--threads", type=int, default=4)
    e.add_argument("--force", action="store_true")
    r = sub.add_parser("report", help="tables and summary")
    r.add_argument("--out", required=True)
    for k in ("rank-change", "rank-change-seeds", "wider-encoders", "mixture-components", "density-heads", "resplit",
              "new-datasets", "process-logs", "synthetic", "coarsened-real", "coarsened-more"):
        r.add_argument(f"--{k}", default=None, help=f"{k.replace('-', '_')} run directory (scored with `score`)")
    r.add_argument("--intensity-scores", nargs="*", default=None, help="output directories of coarse_tpp.rescore_retrained score")
    r.add_argument("--intensity-jitter", nargs="*", default=None, help="output directories of coarse_tpp.intensity_jitter score")
    r.add_argument("--intensity-repro", nargs="*", default=None, help="further directories with reproduction references of the "
                   "retrained EasyTPP models (e.g. the repro/ directory of the archive)")
    r.add_argument("--exceedance", default=None, help="report directory of coarse_tpp.intensity_exceedance")
    r.add_argument("--audit", default=None, help="audit.json (tie shares; default results/audit/audit.json)")
    r.add_argument("--B", type=int, default=B_DEFAULT, help="bootstrap draws")
    a = ap.parse_args(argv)
    if a.cmd == "score":
        cmd_score(a.run, None, a.shard, a.cpu, a.threads, a.arms, a.only, a.limit, a.force, a.retry_quarantined)
    elif a.cmd == "edits":
        cmd_edits(a.run, None, a.cpu, a.threads, a.force)
    else:
        print("\n".join(cmd_report(a)))


if __name__ == "__main__":
    main()
