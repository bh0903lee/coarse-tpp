"""Jitter and point-mass scores of the retrained EasyTPP models (Online Resource 1, Section S4 and the lower part of
Table S5). No training: the checkpoints of coarse_tpp.easytpp_train (or of the trained-models archive) are re-scored.

    python -m coarse_tpp.intensity_jitter score  --experiment {time_only,further} --ckpt RUNS --data PREPARED \
                                                 --scores RESCORE_OUT --out OUT [--datasets ..] [--models ..] [--seeds ..] [--cpu]
    python -m coarse_tpp.intensity_jitter report --scores OUT_TIME_ONLY OUT_FURTHER --jitter JIT_TIME_ONLY JIT_FURTHER \
                                                 --out DIR [--exceedance EXC_DIR] [--repro-dir ARCHIVE/repro]

RESCORE_OUT is the output directory of `coarse_tpp.rescore_retrained score` for the same experiment (or `import`
plus `score` for the runs the import skips): its score files give the delta, tie threshold and integration rule of
each run, its floor and interval scores (the reproduction gate below) and the inputs it was computed from.

Scores per target event (natural log; tau~ = recorded gap, a stored gap <= the data set's tie threshold is 0):
  jitter      J = sum_j w_j log q(tau~ + delta u_j), (u_j, w_j) the 32-node Gauss-Legendre rule on [0, 1], i.e.
              E_u log q(tau~ + delta u), u ~ U(0, 1). For an intensity model q(s) = lambda(s) exp(-Lambda(0, s)), so
              J = sum_j w_j [log lambda(tau~ + delta u_j) - Lambda(tau~, tau~ + delta u_j)] - Lambda(0, tau~), where
              lambda is the total intensity with EasyTPP's eps per mark (as in the floor score), Lambda(0, tau~) is the
              compensator of the interval score (graded rule of coarse_tpp.intensity_eval; S2P2 by the same stratified
              Monte Carlo draws), and Lambda(tau~, tau~ + delta u_j) is integrated by a composite rule whose panel
              breaks contain every u_j (the graded breaks with half the nodes per panel of the run's rule, at least 8;
              S2P2: the run's `cell_panels` uniform cell panels, two nodes each), so that every partial integral is a
              sum of whole panels. IntensityFree: the closed-form log-normal mixture density at tau~ + delta u_j;
              FullyNN: log(sum_k dPhi_k/ds + K eps) - Phi(s), the convention of its floor score.
  point mass  for a model without an atom: on a tie the interval score's tie term (probability of the first cell),
              on a positive gap the log density at the recorded gap; computed from the stored scores as
              (n_tie interval_tie + n_pos floor_1e-10_nontie) / n (every untied gap exceeds 1e-10, so the floor
              1e-10 score of an untied event is the density at the recorded gap). No re-scoring.
Reproduction gate: the same pass recomputes the floor scores and the interval score; a run is quarantined when its
dev or test means differ from the rescore_retrained score file by more than GATE_TOL (interval) or GATE_TOL_FLOOR
(floor 1e-6: a floor at a narrow density peak, as in IntensityFree, is the most sensitive to float order), or its
event / tie counts differ. Score on the device the score file was computed on: S2P2's stratified draws come from the
device's generator.

Cache: OUT/<data set>__<model>__s<seed>.json is reused only when its version, the score file's cache key (inputs,
rule, delta, tie threshold) and the jitter settings all match.

Report (DIR/report.md, DIR/summary.json): per data set (Email-EU, Retweet subsample, CollegeMsg from time_only;
Volcano, Sepsis, BPIC12 from further), the architecture chosen by the highest validation seed mean of each score
(floor 1e-6, interval, point mass, jitter) among the eight architectures, and the test S2P2 - NHP seed-paired mean
difference with the number of seeds on which S2P2 is higher. Eligibility as in the existing reports: quarantined runs
(including the reproduction check of `further`) are left out; time_only (as coarse_tpp.intensity_exceedance) ranks an
architecture only with a finite value on all five seeds; further uses the seed means over the valid runs
(coarse_tpp.rescore_retrained). With --exceedance (the output directory of `coarse_tpp.intensity_exceedance score`,
which must hold every selected architecture: pass --models to that command), the test Brier loss (mean over the four
horizons and the five seeds) of the architecture each score selects, its difference from the interval selection and
the number of seeds on which it is lower.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import yaml

from . import easytpp as E
from . import intensity_eval as IE
from . import rescore_retrained as RR
from .easytpp_train import EXPERIMENTS, experiment_of
from .rescore_intensity import batches, make_batch

VERSION = 1
N_NODES = 32
GATE_TOL = 1e-5
GATE_TOL_FLOOR = 1e-4
MODELS = ("RMTPP", "NHP", "SAHP", "AttNHP", "THP", "S2P2", "FullyNN", "IntensityFree")
SCORES = ("floor", "interval", "point_mass", "jitter")
KEY = {"floor": "floor_1e-6", "interval": "interval", "point_mass": "point_mass", "jitter": "jitter"}
LABEL = {"floor": "floor 1e-6", "interval": "interval", "point_mass": "point mass", "jitter": "jitter"}
F6, IV = RR.F6, RR.IV


# ------------------------------------------------------------------ quadrature (pure)
@lru_cache(maxsize=None)
def jitter_nodes(n: int = N_NODES):
    """Gauss-Legendre nodes u_j in (0, 1) and weights w_j summing to 1."""
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1) / 2, w / 2


@lru_cache(maxsize=None)
def merged_rule(breaks: tuple, n: int, nodes: int = N_NODES):
    """Composite Gauss-Legendre rule on [0, 1] with the panel breaks `breaks` plus every jitter node u_j, n nodes per
    panel. Returns (u, W, A) with (A f)_j = int_0^{u_j} f exactly as a sum of whole panels."""
    uj, _ = jitter_nodes(nodes)
    b = np.unique(np.concatenate([np.asarray(breaks, float), uj, [0.0, 1.0]]))
    x, w = np.polynomial.legendre.leggauss(n)
    us, ws, ends = [], [], []
    for a, c in zip(b[:-1], b[1:]):
        us.append(a + (c - a) * (x + 1) / 2); ws.append((c - a) * w / 2); ends.append(np.full(n, c))
    u, W, e = np.concatenate(us), np.concatenate(ws), np.concatenate(ends)
    A = (e[None, :] <= uj[:, None] * (1 + 1e-15)) * W[None, :]
    return u, W, A


def jitter_rule(rule: IE.Rule, ds: str, model_id: str):
    """The merged rule of one model under `rule` (module docstring)."""
    if rule.stratified(model_id):
        p = rule.cell_panels
        return merged_rule(tuple(np.arange(p + 1) / p), 2)
    return merged_rule(IE.GRADED_BREAKS, max(rule.nodes_per_panel(ds, model_id) // 2, 8))


def point_mass(split: dict):
    """Point-mass score of a model without an atom from its stored split means (module docstring); None if missing."""
    if not split:
        return None
    n, nt = split.get("n"), split.get("n_tie")
    a, b = split.get(f"{IV}_tie"), split.get("floor_1e-10_nontie")
    if not n:
        return None
    if nt == 0:
        return b if RR.fin(b) else None
    if nt == n:
        return a if RR.fin(a) else None
    return (nt * a + (n - nt) * b) / n if RR.fin(a) and RR.fin(b) else None


def _t(a, device):
    return torch.as_tensor(np.asarray(a), dtype=torch.float64, device=device)


def jitter_from_intensity(lam_j, lam_m, base, delta, A):
    """J per event from the total intensity at the jitter points lam_j [..., NJ], on the merged rule lam_m [..., G],
    the compensator base = Lambda(0, tau~) and the merged rule's cumulative matrix A [NJ, G]."""
    _, wj = jitter_nodes(lam_j.shape[-1])
    Lj = delta * torch.einsum("...g,jg->...j", lam_m, _t(A, lam_m.device))
    return (_t(wj, lam_j.device) * (torch.log(lam_j) - Lj)).sum(-1) - base


# ------------------------------------------------------------------ per-batch scores
def scores(ev: IE.Intensity, delta, rule: IE.Rule, ds: str, tie_thr: float, device):
    """Floor scores, interval score and jitter score of every target of a batch (one set of intensity evaluations)."""
    model, mid, batch = ev.model, ev.mid, ev.batch
    tau = IE._targets(batch, tie_thr)
    uj, wj = jitter_nodes(N_NODES)
    uj_t, wj_t = _t(uj, device), _t(wj, device)
    xj = tau[..., None] + delta * uj_t
    if mid in ("IntensityFree", "FullyNN"):
        out = IE.time_scores(ev, delta, rule, ds, tie_thr, device)
        if mid == "IntensityFree":
            p = IE.ifl_params(model, batch)
            m, s = float(model.mean_log_inter_time), float(model.std_log_inter_time)
            lq = IE.ifl_logpdf(tuple(t[..., None, :] for t in p), xj, m, s)
        else:
            parts = []
            for i in range(0, xj.shape[-1], 64):
                Phi, lam = IE.fullynn_phi(model, batch, xj[..., i:i + 64])
                parts.append(torch.log(lam) - Phi)
            lq = torch.cat(parts, -1)
        out["jitter"] = (wj_t * lq).sum(-1)
        return out
    um, _, A = jitter_rule(rule, ds, mid)
    xs = {k: torch.clamp(tau, min=e) for k, e in IE.FLOORS.items()}
    ug, wg = IE.graded_rule(rule.nodes_per_panel(ds, mid)); ug, wg = ug.to(device), wg.to(device)
    out = {}
    if rule.stratified(mid):
        M = rule.strata
        gen = IE.strat_generator(rule, batch, device)
        U = (torch.arange(M, device=device, dtype=torch.float64)
             + torch.rand(*tau.shape, M, device=device, dtype=torch.float64, generator=gen)) / M
        lam = ev.total(tau[..., None] * U)
        base = tau * lam.mean(-1)
        del lam, U
        uc, wc = IE.dense_cell_rule(rule.cell_panels); uc, wc = uc.to(device), wc.to(device)
        ranges = [(tau, xs[k]) for k in IE.FLOORS] + [(tau, tau + delta)]
        rls = [(ug, wg)] * len(IE.FLOORS) + [(uc, wc)]
    else:
        ranges = [(torch.zeros_like(tau), xs[k]) for k in IE.FLOORS] + [(torch.zeros_like(tau), tau), (tau, tau + delta)]
        rls = [(ug, wg)] * len(ranges)
    pts = [a[..., None] + (b - a)[..., None] * u for (a, b), (u, _) in zip(ranges, rls)]
    pts += [torch.stack([xs[k] for k in IE.FLOORS], -1), xj, tau[..., None] + delta * _t(um, device)]
    lam = ev.total(torch.cat(pts, -1))
    Lam, off = [], 0
    for (a, b), (u, w) in zip(ranges, rls):
        Lam.append(((lam[..., off:off + len(u)] * w).sum(-1)) * (b - a)); off += len(u)
    nf = len(IE.FLOORS)
    lam_pt = lam[..., off:off + nf]; off += nf
    lam_j = lam[..., off:off + N_NODES]; off += N_NODES
    lam_m = lam[..., off:]
    if not rule.stratified(mid):
        base = Lam[nf]
    for j, k in enumerate(IE.FLOORS):
        out[k] = torch.log(lam_pt[..., j]) - (Lam[j] if not rule.stratified(mid) else base + Lam[j])
    y = Lam[-1].clamp(min=1e-300)
    out["interval"] = -base + torch.log(-torch.expm1(-y))
    out["jitter"] = jitter_from_intensity(lam_j, lam_m, base, delta, A)
    return IE._finish(out, tau, batch)


def score_split(model, model_id, seqs, pad_id, delta, rule: IE.Rule, ds: str, tie_thr: float, device):
    """Means over the target events of the floor, interval and jitter scores, with tie / non-tie means."""
    acc, n, n_tie = {}, 0, 0
    with torch.no_grad():
        for bs in batches(seqs):
            b = make_batch(bs, pad_id, device)
            o = scores(IE.Intensity(model, model_id, b), delta, rule, ds, tie_thr, device)
            m = o["mask"]; tie = o["tie"] & m
            n += int(m.sum()); n_tie += int(tie.sum())
            for k in [k for k in o if k not in ("tie", "mask", "phi0")]:
                v = o[k]
                a = acc.setdefault(k, [0.0, 0.0, 0.0, 0])
                a[0] += float(v[m].sum()); a[1] += float(v[tie].sum()); a[2] += float(v[m & ~o["tie"]].sum())
                a[3] += int((~torch.isfinite(v[m])).sum())
    out = {"n": n, "n_tie": n_tie}
    for k, (s, st, sn, nbad) in acc.items():
        out[k] = s / n if n else None
        out[f"{k}_tie"] = st / n_tie if n_tie else None
        out[f"{k}_nontie"] = sn / (n - n_tie) if n > n_tie else None
        out[f"{k}_nonfinite"] = nbad
    return out


def gate(mine: dict, stored: dict):
    """(ok, differences) of the recomputed floor 1e-6 and interval means and counts against a stored split."""
    d = {}
    ok = mine.get("n") == stored.get("n") and mine.get("n_tie") == stored.get("n_tie")
    for k, tol in ((IV, GATE_TOL), (F6, GATE_TOL_FLOOR)):
        a, b = mine.get(k), stored.get(k)
        d[k] = abs(a - b) if RR.fin(a) and RR.fin(b) else (0.0 if a == b else float("inf"))
        ok &= d[k] <= tol
    return bool(ok), d


# ------------------------------------------------------------------ score command
def score(experiment, ckpt, data_dir, scores_out, out, datasets=None, models=None, seeds=E.SEEDS, cpu=False, force=False):
    experiment_of(experiment)
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for ds in datasets or EXPERIMENTS[experiment]["datasets"]:
        pkl = Path(data_dir) / f"{E.EASYTPP_ID[ds]}.pkl"
        for model_id in models or MODELS:
            for seed in seeds:
                ref_p = Path(scores_out) / "scores" / f"{ds}__{model_id}__s{seed}.json"
                if not ref_p.exists():
                    continue
                ref = json.loads(ref_p.read_text())
                if ref.get("experiment") != experiment or not ref.get("dev") or not ref.get("test"):
                    continue
                loc, res, ident = RR.run_inputs(ckpt, ds, model_id, seed, data_dir)
                if not ident["checkpoint"] or not ident["config"]:
                    continue
                rd = ref.get("rule") or {}
                rule = IE.Rule(per_panel=int(rd.get("per_panel", 32)), strata=int(rd.get("strata", 0)),
                               strat_seed=int(rd.get("strat_seed", 0)), cell_panels=int(rd.get("cell_panels", 1500)),
                               overrides=())
                key = dict(version=VERSION, nodes=N_NODES, score_file=ref.get("cache_key"),
                           inputs={k: v for k, v in ident.items() if k.endswith("sha256")})
                dst = out / f"{ds}__{model_id}__s{seed}.json"
                if dst.exists() and not force and json.loads(dst.read_text()).get("cache_key") == key:
                    continue
                t0 = time.time()
                cfg = yaml.safe_load(Path(ident["config"]).read_text())
                model = IE.build_model(model_id, cfg, device, pkl)
                model.load_state_dict(torch.load(ident["checkpoint"], map_location=device, weights_only=True))
                pad = cfg["model_config"]["event_pad_index"]
                r = dict(cache_key=key, experiment=experiment, dataset=ds, model=model_id, seed=seed, delta=ref["delta"],
                         tie_threshold=ref["tie_threshold"], rule=rd, inputs=ident, score_file=str(ref_p.resolve()),
                         device_scored=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                         device_score_file=ref.get("device_scored"), gate={}, quarantined=False)
                for sp in ("dev", "test"):
                    seqs, _ = E.load_split(pkl, sp)
                    s = score_split(model, model_id, seqs, pad, ref["delta"], rule, ds, ref["tie_threshold"], device)
                    ok, d = gate(s, ref[sp])
                    r["gate"][sp] = dict(ok=ok, **{f"abs_diff_{k}": v for k, v in d.items()})
                    r["quarantined"] |= not ok
                    r[sp] = {k: s.get(k) for k in ("n", "n_tie", "jitter", "jitter_tie", "jitter_nontie", "jitter_nonfinite",
                                                   F6, IV)}
                r["wall_sec"] = round(time.time() - t0, 1)
                tmp = dst.with_suffix(f".tmp{os.getpid()}")
                tmp.write_text(json.dumps(r, indent=1)); os.replace(tmp, dst)
                print(f"{ds} {model_id} s{seed}: test jitter {r['test']['jitter']:.5f}; gate "
                      f"{ {sp: {k: float(f'{v:.1e}') for k, v in g.items() if k != 'ok'} for sp, g in r['gate'].items()} }"
                      f"{' QUARANTINED' if r['quarantined'] else ''} ({r['wall_sec']} s)", flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()


# ------------------------------------------------------------------ report
def load(scores_dirs, jitter_dirs, repro_dirs=()):
    """Cells of every rescore_retrained output directory (time_only and further) with the point-mass score added to
    each split and the jitter score from the jitter files of the same run and inputs (else None)."""
    jit = {}
    for d in jitter_dirs:
        for p in sorted(Path(d).glob("*__s*.json")):
            j = json.loads(p.read_text())
            if j.get("cache_key", {}).get("version") == VERSION:
                jit[(j["experiment"], j["dataset"], j["model"], j["seed"])] = j
    cells = []
    for d in scores_dirs:
        for exp in ("time_only", "further"):
            for c in RR.load_cells(exp, d, repro_dirs):
                if c["model"] not in MODELS:
                    continue
                j = jit.get((exp, c["dataset"], c["model"], c["seed"]))
                ok = (j is not None and not j.get("quarantined")
                      and j["cache_key"].get("score_file") == c.get("cache_key")
                      and all(j["inputs"].get(k) == (c.get("inputs") or {}).get(k)
                              for k in ("checkpoint_sha256", "config_sha256")))
                for sp in ("dev", "test"):
                    if c.get(sp):
                        c[sp]["point_mass"] = point_mass(c[sp])
                        c[sp]["jitter"] = j[sp]["jitter"] if ok else None
                c["jitter_status"] = "ok" if ok else ("missing" if j is None else "quarantined or stale")
                cells.append(c)
    return cells


def select(cells, ds, key, require_all):
    """(model, {model: validation seed mean}): highest mean over the valid runs; require_all: only architectures with
    a finite value on all five seeds."""
    means = {}
    for m in MODELS:
        v = {c["seed"]: c["dev"].get(key) for c in cells if c["dataset"] == ds and c["model"] == m and RR.valid(c)}
        if require_all:
            if set(v) == set(E.SEEDS) and all(RR.fin(x) for x in v.values()):
                means[m] = float(np.mean(list(v.values())))
        else:
            x = [y for y in v.values() if RR.fin(y)]
            if x:
                means[m] = float(np.mean(x))
    best = max(sorted(means), key=lambda m: means[m]) if means else None
    return best, means


def s2p2_minus_nhp(cells, ds, key):
    """Test S2P2 - NHP per seed (valid runs, finite under `key`), mean and number of seeds with S2P2 higher."""
    P = RR.per_seed(cells, ds, "test", key)
    a, b = P.get("S2P2", {}), P.get("NHP", {})
    d = {s: a[s] - b[s] for s in E.SEEDS if RR.fin(a.get(s)) and RR.fin(b.get(s))}
    return dict(mean=float(np.mean(list(d.values()))) if d else None, per_seed=d, n_pos=sum(v > 0 for v in d.values()), n=len(d))


def brier(exc_dir, ds, model):
    """{seed: test Brier loss (mean over the horizons)} from the files of coarse_tpp.intensity_exceedance score."""
    out = {}
    for s in E.SEEDS:
        p = Path(exc_dir) / f"{ds}__{model}__s{s}.json"
        if p.exists():
            j = json.loads(p.read_text())
            arr = j.get("arrays") or {}
            if j.get("complete") and arr.get("path") and Path(arr["path"]).is_file() and E.sha256(arr["path"]) == arr.get("sha256"):
                out[s] = j["brier_mean"]
    return out


def section(scores_dirs, jitter_dirs, exceedance=None, repro_dirs=()):
    """Markdown lines and summary of the report (module docstring). summary = {data set: dict(experiment,
    test_tie_share, selected {score: model}, dev_means {score: {model: mean}}, s2p2_minus_nhp {score: dict(mean,
    per_seed, n_pos, n)}, brier {score: dict(model, brier, diff_vs_interval, rel_vs_interval, seeds_lower)} (time_only
    data sets, with exceedance)), missing_jitter: [runs used without a jitter value]}; scores are floor, interval,
    point_mass, jitter."""
    return summarize(load(scores_dirs, jitter_dirs, repro_dirs), exceedance)


def summarize(cells, exceedance=None):
    """section() on loaded cells (dicts with experiment, dataset, model, seed, status, quarantined and dev / test split
    means carrying floor_1e-6, interval, point_mass, jitter, n, n_tie)."""
    L = ["## Retrained EasyTPP models: choice of architecture by each score (Online Resource 1, Table S5, lower part)", "",
         "Architecture with the highest validation seed mean among the eight (time_only: finite on all five seeds; further: "
         "valid runs). S2P2 - NHP: test seed-paired mean difference (nats per event), in brackets the number of seeds on "
         "which S2P2 is higher. Point mass: interval tie term on ties, density at the recorded gap otherwise. Jitter: "
         f"E_u log q(tau~ + delta u), {N_NODES}-node Gauss-Legendre.", "",
         "| data set (test ties) | " + " | ".join(LABEL[s] for s in SCORES) + " | "
         + " | ".join(f"S2P2 - NHP {LABEL[s]}" for s in SCORES) + " |", "|---|" + "---|" * 4 + "---:|" * 4]
    summ = {}
    order = [(exp, ds) for exp in ("time_only", "further") for ds in EXPERIMENTS[exp]["datasets"]]
    for exp, ds in order:
        cs = [c for c in cells if c["dataset"] == ds and c["experiment"] == exp]
        if not cs:
            continue
        sel = {s: select(cs, ds, KEY[s], exp == "time_only") for s in SCORES}
        dif = {s: s2p2_minus_nhp(cs, ds, KEY[s]) for s in SCORES}
        tie = RR.tie_share(cs, ds)
        D = dict(experiment=exp, test_tie_share=tie, selected={s: v[0] for s, v in sel.items()},
                 dev_means={s: v[1] for s, v in sel.items()}, s2p2_minus_nhp=dif)
        f3 = lambda x: "—" if x is None else f"{x:+.3f}"
        L.append(f"| {E.NAMES[ds]} ({100 * tie:.2f}%) | " + " | ".join(str(D["selected"][s]) for s in SCORES) + " | "
                 + " | ".join(f"{f3(dif[s]['mean'])} ({dif[s]['n_pos']}/{dif[s]['n']})" for s in SCORES) + " |")
        if exceedance and exp == "time_only":
            B = {}
            bi = brier(exceedance, ds, D["selected"]["interval"])
            for s in SCORES:
                m = D["selected"][s]
                b = brier(exceedance, ds, m)
                if len(b) < len(E.SEEDS) or len(bi) < len(E.SEEDS):
                    B[s] = dict(model=m, brier=None, note="not scored by coarse_tpp.intensity_exceedance for every seed")
                    continue
                mb, mi = float(np.mean(list(b.values()))), float(np.mean(list(bi.values())))
                B[s] = dict(model=m, brier=mb, diff_vs_interval=mb - mi, rel_vs_interval=(mb - mi) / mi,
                            seeds_lower=sum(b[k] < bi[k] for k in E.SEEDS))
            D["brier"] = B
        summ[ds] = D
    if exceedance:
        L += ["", "Test Brier loss (mean over the four horizons and five seeds) of the architecture each score selects; "
              "difference from the interval selection (relative), seeds on which it is lower:", ""]
        for ds, D in summ.items():
            for s, b in (D.get("brier") or {}).items():
                L.append(f"- {E.NAMES[ds]}, {LABEL[s]}: {b['model']} " + (
                    "pending (" + b["note"] + ")" if b.get("brier") is None else
                    f"{b['brier']:.4f}, {b['diff_vs_interval']:+.4f} ({100 * b['rel_vs_interval']:+.1f}%), "
                    f"{b['seeds_lower']}/{len(E.SEEDS)}"))
    miss = sorted(f"{c['dataset']} {c['model']} s{c['seed']} ({c['jitter_status']})" for c in cells
                  if RR.valid(c) and c.get("jitter_status", "ok") != "ok")
    L += ["", "Valid runs without a jitter value (run `score`): " + (", ".join(miss) if miss else "none") + "."]
    return L, dict(per_dataset=summ, missing_jitter=miss)


def report(scores_dirs, jitter_dirs, out, exceedance=None, repro_dirs=()):
    L, summ = section(scores_dirs, jitter_dirs, exceedance, repro_dirs)
    L = ["# Jitter and point-mass scores of the retrained EasyTPP models", ""] + L
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("score", "report"))
    ap.add_argument("--experiment", choices=("time_only", "further"), help="score: the experiment")
    ap.add_argument("--ckpt", help="score: run root (or the archive directory of the experiment)")
    ap.add_argument("--data", help="score: directory of the prepared EasyTPP pickles")
    ap.add_argument("--scores", nargs="+", required=True, help="output directories of coarse_tpp.rescore_retrained "
                    "(score: the one of --experiment)")
    ap.add_argument("--jitter", nargs="*", default=[], help="report: output directories of `score`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exceedance", default=None, help="report: output directory of coarse_tpp.intensity_exceedance score")
    ap.add_argument("--repro-dir", nargs="*", default=[], help="report: further directories with reproduction reference "
                    "files (e.g. the repro/ directory of an archive)")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--models", nargs="*", choices=MODELS)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(E.SEEDS))
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "score":
        if not (a.experiment and a.ckpt and a.data) or len(a.scores) != 1:
            ap.error("score needs --experiment, --ckpt, --data and one --scores directory")
        score(a.experiment, a.ckpt, a.data, a.scores[0], a.out, a.datasets, a.models, tuple(a.seeds), a.cpu, a.force)
    else:
        print("\n".join(report(a.scores, a.jitter, a.out, a.exceedance, a.repro_dir)))


if __name__ == "__main__":
    main()
