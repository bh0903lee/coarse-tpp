"""Threshold-exceedance forecasts of the retrained EasyTPP models (time-only experiment; Section S4, paragraph on
threshold-exceedance forecasts, last part): the architecture a practitioner would pick by the floor score and by the
interval score, compared by the Brier loss of P(next event not recorded within h). No training.

    python -m coarse_tpp.intensity_exceedance score  --scores OUT_TIME_ONLY --ckpt RUNS --data PREPARED --out OUT
    python -m coarse_tpp.intensity_exceedance report --scores OUT_TIME_ONLY --ckpt RUNS --out OUT

Selection per data set (Email-EU, Retweet subsample, CollegeMsg): the architecture with the highest mean dev floor 1e-6
score and the one with the highest mean dev interval score, means over the five seeds (diverged seeds included); an
architecture without a finite dev value on every seed is not rankable. The dev scores are those of
`coarse_tpp.rescore_retrained score --experiment time_only` (OUT_TIME_ONLY/scores).

Forecasts on the test split, for every seed of the chosen architectures: p_h = P(recorded gap > h) = P(s >= h + delta)
at h = 0 and at the 50%, 75% and 90% quantiles of the positive train gaps rounded down to the grid
(coarse_tpp.alarms.H_KEYS, QUANTS), outcomes from the recorded gaps in double precision; Brier loss, logarithmic
score, AUC and costs as coarse_tpp.alarms.sla_metrics. Survival P(s >= x) = exp(-Lambda(0, x)):
  intensity models  Lambda by the graded composite Gauss-Legendre rule (coarse_tpp.intensity_eval) with 128 nodes per
                    panel over the nested ranges [0, x_1], [x_1, x_2], ...; also with 32 nodes (convergence check,
                    reported as max |p_32 - p_128|);
  S2P2              stratified Monte Carlo, `strata` (65,536) equal strata of each [0, x_j] with one uniform point
                    each, the generator seeded per batch as for its scores (strat seed 0);
  IntensityFree     closed form (log-normal mixture with EasyTPP's affine of the log gap);
  FullyNN           excluded (exp(-Phi) with EasyTPP's Phi, whose Phi(0) is not subtracted, is not a normalized
                    survival); --allow-fullynn scores it under that convention.
The report gives the Brier loss averaged over the four horizons and the seeds for both choices, the difference
(interval choice minus floor choice) with a 95% case-level bootstrap interval (coarse_tpp.alarms.boot_diff: 2,000
resamples of the test sequences, the same draws for both models and every seed; conditional on the integration draws),
and the per-seed differences. A flag (not a gate) compares the mean log(1 - p_0) over test ties with the stored
test interval score of the ties.

Cache: OUT/<data set>__<model>__s<seed>.json and .npz are reused only for the same version, method, stratified
settings, horizons, delta and checkpoint and configuration SHA-256; the report rejects caches that no longer match.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from . import alarms as A
from . import easytpp as E
from . import intensity_eval as IE
from .analysis import audit_rows
from .rescore_intensity import MAX_TOKENS, make_batch
from .rescore_retrained import load_cells, F6, IV

VERSION = 1
DATASETS = ("email_eu", "retweet_sub", "collegemsg")
STRATA = 65536
REPRO_TOL = 1e-4


def primary_method(model_id):
    if model_id in ("IntensityFree", "FullyNN"):
        return "closed"
    return "strat" if model_id == "S2P2" else "graded128"


def methods_for(model_id, check=True):
    prim = primary_method(model_id)
    if prim == "closed":
        return ["closed"]
    return sorted({"graded32", "graded128", prim}) if check else [prim]


# ------------------------------------------------------------------ selection
def select(cells, ds):
    """{'floor': (model, mean), 'interval': (model, mean)} by the dev means over all five seeds, and all means."""
    by = {}
    for c in cells:
        if c["dataset"] == ds:
            by.setdefault(c["model"], {})[c["seed"]] = c
    means = {}
    for m, seeds in by.items():
        for key, k in (("floor", F6), ("interval", IV)):
            v = [(s.get("dev") or {}).get(k) for s in seeds.values()]
            ok = len(v) == len(E.SEEDS) and all(x is not None and math.isfinite(x) for x in v)
            means.setdefault(key, {})[m] = float(np.mean(v)) if ok else None
    out = {}
    for key in ("floor", "interval"):
        r = {m: v for m, v in means.get(key, {}).items() if v is not None}
        best = max(sorted(r), key=lambda m: r[m]) if r else None
        out[key] = (best, r.get(best))
    return out, means


def h_grid(pkl, ds, delta):
    """{key: (kh, h, raw quantile)} from the positive train gaps (sequences of length >= 2)."""
    seqs, _ = E.load_split(pkl, "train")
    tg = np.concatenate([s[1][1:] for s in seqs])
    tg = np.where(tg <= E.tie_threshold(ds), 0.0, tg)
    pos = tg[tg > 0]
    out = {"h0": (0, 0.0, 0.0)}
    for k, q in A.QUANTS.items():
        v = float(np.quantile(pos, q)); kh = int(math.floor(v / delta + 1e-6))
        out[k] = (kh, kh * delta, v)
    return out


def index_batches(seqs):
    """coarse_tpp.rescore_intensity.batches over sequence indices (same length-sorted packing)."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    i = 0
    while i < len(order):
        n = len(seqs[order[i]][0]); b = max(1, MAX_TOKENS // n)
        j = min(len(order), i + b)
        while j > i + 1 and len(seqs[order[j - 1]][0]) * (j - i) > MAX_TOKENS:
            j -= 1
        yield order[i:j]
        i = j


# ------------------------------------------------------------------ survival
def Lambda_graded(ev, xs, nodes, device):
    """Lambda(0, x_j) [B, N-1, H] over nested ranges [x_{j-1}, x_j] (x_{-1} = 0) with the graded rule."""
    u, w = IE.graded_rule(nodes); u, w = u.to(device), w.to(device)
    lo = torch.zeros_like(xs[..., 0])
    ranges = []
    for j in range(xs.shape[-1]):
        ranges.append((lo, xs[..., j])); lo = xs[..., j]
    lam = ev.total(torch.cat([a[..., None] + (b - a)[..., None] * u for a, b in ranges], -1))
    n = len(u)
    parts = [(lam[..., r * n:(r + 1) * n] * w).sum(-1) * (b - a) for r, (a, b) in enumerate(ranges)]
    return torch.cumsum(torch.stack(parts, -1), -1)


def Lambda_strat(ev, xs, rule: IE.Rule, device):
    """Lambda(0, x_j) [B, N-1, H] by stratified Monte Carlo with rule.strata strata of each [0, x_j] (blocks of 8,192
    strata; one generator per batch, seeded as for the scores)."""
    M = rule.strata
    gen = IE.strat_generator(rule, ev.batch, device)
    out = []
    for j in range(xs.shape[-1]):
        x = xs[..., j]
        acc = torch.zeros_like(x)
        for c0 in range(0, M, 8192):
            m = min(8192, M - c0)
            U = (torch.arange(c0, c0 + m, device=device, dtype=torch.float64)
                 + torch.rand(*x.shape, m, device=device, dtype=torch.float64, generator=gen)) / M
            acc += ev.total(x[..., None] * U).sum(-1)
        out.append(x * acc / M)
    return torch.stack(out, -1)


def survival(ev, xs, method, rule, device):
    """p [B, N-1, H] = P(s >= x) at xs."""
    if ev.mid == "IntensityFree":
        p = IE.ifl_params(ev.model, ev.batch)
        return torch.exp(IE.ifl_log_survival(p, xs, float(ev.model.mean_log_inter_time), float(ev.model.std_log_inter_time)))
    if ev.mid == "FullyNN":
        Phi, _ = IE.fullynn_phi(ev.model, ev.batch, xs)
        return torch.exp(-Phi)
    if method == "strat":
        return torch.exp(-Lambda_strat(ev, xs, rule, device))
    return torch.exp(-Lambda_graded(ev, xs, int(method.replace("graded", "")), device))


@torch.no_grad()
def probs(model, mid, seqs, pad, kh, delta, methods, rule, ds, device):
    """{method: p (N, H)}, tau (N,) float64 recorded gaps (0 for ties), sequence index (N,)."""
    acc = {m: [] for m in methods}
    taus, sq = [], []
    thr = E.tie_threshold(ds)
    for idx in index_batches(seqs):
        b = make_batch([seqs[i] for i in idx], pad, device)
        ev = IE.Intensity(model, mid, b)
        ts, dts, ks, non_pad, attn = b
        tgt = dts[:, 1:].double()
        tau = torch.where(tgt <= thr, torch.zeros_like(tgt), tgt)
        msk = non_pad[:, 1:]
        xs = torch.tensor([(k + 1) * delta for k in kh], dtype=torch.float64, device=device).expand(*tau.shape, len(kh)).contiguous()
        for m in methods:
            acc[m].append(survival(ev, xs, m, rule, device)[msk].cpu())
        t64 = torch.zeros_like(tau)
        si = torch.zeros(tau.shape, dtype=torch.long)
        for r, i in enumerate(idx):
            g = np.where(seqs[i][1][1:] <= thr, 0.0, seqs[i][1][1:])
            t64[r, :len(g)] = torch.as_tensor(g, dtype=torch.float64, device=device); si[r] = i
        taus.append(t64[msk].cpu()); sq.append(si[msk.cpu()])
    return {m: torch.cat(v).double().numpy() for m, v in acc.items()}, torch.cat(taus).numpy(), torch.cat(sq).numpy()


# ------------------------------------------------------------------ runs
def identity(ds, mid, rule, kh, delta, loc):
    out = dict(version=VERSION, primary=primary_method(mid), kh=list(kh), delta=delta,
               checkpoint_sha256=E.sha256(loc["checkpoint"]), config_sha256=E.sha256(loc["config"]))
    if out["primary"] == "strat":
        out["strat"] = dict(strata=rule.strata, strat_seed=rule.strat_seed)
    return out


def score_run(ds, mid, seed, delta, kh, rule, ckpt, pkl, scores_out, out_dir, device, max_seqs=None, check=True):
    loc = E.locate_run(ckpt, ds, mid, seed)
    if loc["checkpoint"] is None:
        return dict(dataset=ds, model=mid, seed=seed, missing=True)
    cfg = yaml.safe_load(loc["config"].read_text())
    model = IE.build_model(mid, cfg, device, pkl)
    model.load_state_dict(torch.load(loc["checkpoint"], map_location=device, weights_only=True))
    seqs, _ = E.load_split(pkl, "test")
    complete = max_seqs is None or max_seqs >= len(seqs)
    seqs = seqs if complete else seqs[:max_seqs]
    prim = primary_method(mid)
    methods = methods_for(mid, check)
    t0 = time.time()
    P, tau, sq = probs(model, mid, seqs, cfg["model_config"]["event_pad_index"], kh, delta, methods, rule, ds, device)
    y, resid = A.outcomes(tau, kh, delta)
    p = np.clip(P[prim], 0.0, 1.0)
    m, brier, logs, cost = A.sla_metrics(p, y)
    rec = dict(identity=identity(ds, mid, rule, kh, delta, loc), dataset=ds, model=mid, seed=seed, primary=prim,
               methods=methods, complete=complete, n_events=int(len(tau)), grid_residual=resid, **m)
    if "graded32" in P and "graded128" in P:
        d = np.abs(P["graded32"] - P["graded128"])
        rec["conv_32_128"] = dict(max_abs=float(d.max()), mean_abs=float(d.mean()))
    if "strat" in P and "graded128" in P:
        d = np.abs(P["strat"] - P["graded128"])
        rec["conv_strat_128"] = dict(max_abs=float(d.max()), mean_abs=float(d.mean()))
    tie = tau <= 0
    cell = Path(scores_out) / "scores" / f"{ds}__{mid}__s{seed}.json"
    if tie.any() and cell.exists() and complete:
        stored = json.loads(cell.read_text())["test"].get(f"{IV}_tie")
        mine = float(np.mean(np.log(np.clip(1 - P[prim][tie, 0], 1e-300, None))))
        rec["tie_check"] = dict(mine=mine, stored=stored, absdiff=abs(mine - stored) if stored is not None else None,
                                flagged=bool(stored is None or abs(mine - stored) > REPRO_TOL))
    n_seq = len(seqs)
    np.savez_compressed(out_dir / f"{ds}__{mid}__s{seed}.npz", n=A.seq_sums(sq, n_seq, np.ones(len(sq))),
                        brier=A.seq_sums(sq, n_seq, brier), logs=A.seq_sums(sq, n_seq, logs), cost=A.seq_sums(sq, n_seq, cost))
    rec["wall_sec"] = round(time.time() - t0, 1)
    return rec


def cache_ok(j, ds, mid, seed, rule, kh, delta, ckpt):
    loc = E.locate_run(ckpt, ds, mid, seed)
    return loc["checkpoint"] is not None and j.get("identity") == identity(ds, mid, rule, kh, delta, loc)


def _setup(scores_out, ds, audit, pkl):
    cells = [c for c in load_cells("time_only", scores_out) if not any("stale" in p for p in c["problems"])]
    delta = (audit_rows(audit).get(ds) or {}).get("delta")
    hg = h_grid(pkl, ds, delta)
    return cells, delta, hg, [hg[k][0] for k in A.H_KEYS]


def cmd_score(a):
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    rule = IE.Rule(strata=a.strata, strat_seed=a.strat_seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for ds in a.datasets or DATASETS:
        pkl = Path(a.data) / f"{E.EASYTPP_ID[ds]}.pkl"
        cells, delta, hg, kh = _setup(a.scores, ds, a.audit, pkl)
        sel, means = select(cells, ds)
        (out / f"{ds}__selection.json").write_text(json.dumps(dict(
            selection=sel, dev_means=means, delta=delta, h_grid={k: dict(kh=v[0], h=v[1], q_raw=v[2]) for k, v in hg.items()}), indent=1))
        models = a.models or sorted({m for m, _ in sel.values() if m})
        if "FullyNN" in models and not a.allow_fullynn:
            print("FullyNN excluded (exp(-Phi) is not a normalized survival; --allow-fullynn scores it)", flush=True)
            models = [m for m in models if m != "FullyNN"]
        print(f"{ds}: floor -> {sel['floor']}, interval -> {sel['interval']}; h {[hg[k][1] for k in A.H_KEYS]}", flush=True)
        for mid in models:
            for seed in a.seeds:
                p = out / f"{ds}__{mid}__s{seed}.json"
                if p.exists() and not a.force:
                    j = json.loads(p.read_text())
                    if (j.get("complete") or a.max_seqs is not None) and cache_ok(j, ds, mid, seed, rule, kh, delta, a.ckpt):
                        continue
                rec = score_run(ds, mid, seed, delta, kh, rule, a.ckpt, pkl, a.scores, out, device, a.max_seqs, not a.no_check)
                p.write_text(json.dumps(rec, indent=1, default=str))
                print(f"  {mid} s{seed}: brier_mean {rec.get('brier_mean')} conv32/128 {rec.get('conv_32_128')} "
                      f"tie check {rec.get('tie_check')} ({rec.get('wall_sec')} s)", flush=True)


def cmd_report(a):
    out = Path(a.out)
    rule = IE.Rule(strata=a.strata, strat_seed=a.strat_seed)
    L = ["# Threshold-exceedance forecasts of the retrained EasyTPP models (time-only objective)", "",
         "Architecture chosen by the mean dev floor 1e-6 score and by the mean dev interval score over five seeds; test "
         "Brier loss averaged over the horizons h0, q50, q75, q90 and the seeds; diff = interval choice minus floor "
         f"choice with a 95% case-level bootstrap interval. S2P2 survival by stratified Monte Carlo ({rule.strata} "
         f"strata, seed {rule.strat_seed}); other intensity models graded rule, 128 nodes per panel.", "",
         "| data set | floor choice (dev mean) | interval choice (dev mean) | seeds | Brier floor | Brier interval | "
         "diff [95% bootstrap] | per-seed diffs | max \\|p32 - p128\\| | tie flags | complete |",
         "|---|---|---|---:|---:|---:|---|---|---|---|---|"]
    summ = []
    for ds in DATASETS:
        sp = out / f"{ds}__selection.json"
        if not sp.exists():
            continue
        S = json.loads(sp.read_text())
        kh = [S["h_grid"][k]["kh"] for k in A.H_KEYS]
        (mf, vf), (mi, vi) = S["selection"]["floor"], S["selection"]["interval"]

        def runs(mid):
            js, zs = {}, {}
            for p in sorted(out.glob(f"{ds}__{mid}__s*.json")):
                j = json.loads(p.read_text())
                if j.get("missing"):
                    continue
                if not cache_ok(j, ds, mid, j["seed"], rule, kh, S["delta"], a.ckpt):
                    raise ValueError(f"{p.name}: computed with other settings or inputs than requested; re-run score")
                js[j["seed"]] = j; zs[j["seed"]] = dict(np.load(p.with_suffix(".npz")))
            return js, zs
        if "FullyNN" in (mf, mi):
            L.append(f"| {E.NAMES[ds]} | {mf} ({vf:+.4f}) | {mi} ({vi:+.4f}) | — | FullyNN chosen: not scored | | | | | | |")
            continue
        jf, zf = runs(mf); ji, zi = runs(mi)
        seeds = sorted(set(jf) & set(ji))
        if not seeds:
            L.append(f"| {E.NAMES[ds]} | {mf} ({vf:+.4f}) | {mi} ({vi:+.4f}) | 0 | pending | | | | | | |"); continue
        n = zf[seeds[0]]["n"]
        bF = [jf[s]["brier_mean"] for s in seeds]; bI = [ji[s]["brier_mean"] for s in seeds]
        if mf == mi:
            dtxt, pt, lo, hi = "same model", 0.0, 0.0, 0.0
        elif not all(np.array_equal(z["n"], n) for z in [zf[s] for s in seeds] + [zi[s] for s in seeds]):
            dtxt, pt, lo, hi = "not comparable (different test subsets)", float("nan"), float("nan"), float("nan")
        else:
            pt, lo, hi = A.boot_diff([zf[s]["brier"].mean(1) for s in seeds], [zi[s]["brier"].mean(1) for s in seeds], n)
            pt, lo, hi = float(pt), float(lo), float(hi)
            dtxt = f"{pt:+.5f} [{lo:+.5f}, {hi:+.5f}]"
        conv = max([j["conv_32_128"]["max_abs"] for j in list(jf.values()) + list(ji.values()) if "conv_32_128" in j] or [float("nan")])
        flags = sorted({f"{j['model']} s{j['seed']}" for j in list(jf.values()) + list(ji.values()) if (j.get("tie_check") or {}).get("flagged")})
        compl = all(j["complete"] for j in list(jf.values()) + list(ji.values()))
        L.append(f"| {E.NAMES[ds]} | {mf} ({vf:+.4f}) | {mi} ({vi:+.4f}) | {len(seeds)} | {np.mean(bF):.5f} | {np.mean(bI):.5f} | "
                 f"{dtxt} | {', '.join(f'{b - f:+.4f}' for f, b in zip(bF, bI))} | {conv:.2e} | {', '.join(flags) or 'none'} | {compl} |")
        per_h = {h: dict(floor=float(np.mean([jf[s]["brier"][h] for s in seeds])),
                         interval=float(np.mean([ji[s]["brier"][h] for s in seeds]))) for h in A.H_KEYS}
        summ.append(dict(dataset=ds, floor=mf, interval=mi, seeds=seeds, brier_floor=float(np.mean(bF)),
                         brier_interval=float(np.mean(bI)), diff=pt, boot95=[lo, hi],
                         per_seed_diff={s: b - f for s, f, b in zip(seeds, bF, bI)}, brier_by_h=per_h, h_grid=S["h_grid"],
                         complete=compl))
    L += ["", "Brier loss per horizon (floor choice / interval choice):", ""]
    for s in summ:
        L.append(f"- {E.NAMES[s['dataset']]}: " + "; ".join(
            f"{h} (h = {s['h_grid'][h]['h']:g}) {s['brier_by_h'][h]['floor']:.4f} / {s['brier_by_h'][h]['interval']:.4f}" for h in A.H_KEYS))
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("score", "report"))
    ap.add_argument("--scores", required=True, help="output directory of rescore_retrained (time_only)")
    ap.add_argument("--ckpt", required=True, help="run root of the time-only experiment")
    ap.add_argument("--data", help="directory of the prepared EasyTPP pickles (score)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit", default=None)
    ap.add_argument("--datasets", nargs="*", choices=DATASETS)
    ap.add_argument("--models", nargs="*", help="score these architectures instead of the chosen ones")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(E.SEEDS))
    ap.add_argument("--strata", type=int, default=STRATA)
    ap.add_argument("--strat-seed", type=int, default=0)
    ap.add_argument("--max-seqs", type=int, default=None, help="smoke test: first n test sequences (marked incomplete)")
    ap.add_argument("--no-check", action="store_true", help="skip the 32-node convergence check")
    ap.add_argument("--allow-fullynn", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "score":
        if not a.data:
            ap.error("--data is required for score")
        cmd_score(a)
    else:
        print("\n".join(cmd_report(a)))


if __name__ == "__main__":
    main()
