"""Scoring, reproduction check and reports of the retrained EasyTPP models (Section 5.2; Supplementary Tables S3-S5 and
the S2P2 - NHP differences of Fig. S5). Models: coarse_tpp.easytpp_train (experiments time_only, joint, further).
Scores: coarse_tpp.intensity_eval (floors, interval score; joint scores of time and mark for `joint`).

    python -m coarse_tpp.rescore_retrained score  --experiment time_only --ckpt RUNS --data PREPARED --out OUT [--audit A]
    python -m coarse_tpp.rescore_retrained repro  --experiment joint --ckpt RUNS --data PREPARED --out OUT [--repeats 50]
    python -m coarse_tpp.rescore_retrained report --experiment joint --out OUT

RUNS is the run root of coarse_tpp.easytpp_train (<data set>/<model>/s<seed>/ with result.json and EasyTPP's run
directory; the run roots of the trained-models archive have the same layout), PREPARED the directory of the EasyTPP
pickles (`easytpp_train prepare`), delta the audited resolution of the data set (results/audit/audit.json, or --audit).
Options --models, --seeds, --datasets restrict a call; --per-panel, --strata, --strat-seed, --cell-panels change the
integration (coarse_tpp.intensity_eval.Rule; defaults are the paper's: 32 nodes per panel, 128 for AttNHP on Email-EU,
S2P2 by stratified Monte Carlo with 16,384 strata, seed 0 and 1,500 cell panels).

Cache: OUT/scores/<data set>__<model>__s<seed>.json is reused only when its version, delta, tie threshold, integration
settings of that model and the SHA-256 of the checkpoint, its EasyTPP configuration, its result.json and the data file
all match; the report re-checks the hashes and lists every run whose inputs changed as stale (left out).

Problems (a run with any is quarantined: kept in the files, left out of tables, orders and selections): an epoch
budget other than 100, a checkpoint changed after training, an objective that does not match the experiment (mark
term on / off), a data file other than the one the run trained on (when result.json records its hash), no checkpoint.

Reproduction check (experiments joint and further): `repro` re-runs EasyTPP's own evaluation of the checkpoint (its
runner, data loader and validation pass with the training setting, 10 Monte Carlo samples per gap) `repeats` times
(torch seeds 1000 + r) on dev and test. A run passes when |our score - mean of the repeats| <= 0.02 + 3 standard errors
of that mean on both splits, for the score in the training objective's terms: the joint score for `joint`, the time
score otherwise, at eps = 1e-5 for IntensityFree (its training clamps gaps at 1e-5), 1e-6 where ties are stored as
1e-6, 1e-10 on BPIC12 (ties stored as 1e-7 h). A run without a passing check is quarantined. EasyTPP's Monte Carlo
draws come from the device's generator, so the reference differs slightly between devices (within its standard error).
The time-only five-seed table (Table S3) uses every run with a checkpoint, as in the paper; its report gives the check
against the single estimate logged at training time for information.

Divergence: a test floor 1e-6 score below -100 nats per event or non-finite. Tables give mean +- SD over the valid
runs that did not diverge (over the diverged ones, marked *, when all diverged).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from . import easytpp as E
from . import intensity_eval as IE
from .analysis import audit_rows
from .easytpp_train import EXPERIMENTS, EPOCHS, experiment_of, make_config

VERSION = 1
REPRO_TOL = 0.02
MC_SE_MULT = 3
DIVERGED = -100.0
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
F6, IV = "floor_1e-6", "interval"


# ------------------------------------------------------------------ helpers
def fin(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def stats(vals):
    v = [x for x in vals if fin(x)]
    n = len(v)
    if n == 0:
        return dict(n=0, mean=None, sd=None, min=None, max=None)
    return dict(n=n, mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if n > 1 else None,
                min=float(min(v)), max=float(max(v)))


def rank(vals):
    ok = [(m, v) for m, v in vals.items() if fin(v)]
    return [m for m, _ in sorted(ok, key=lambda kv: -kv[1])]


def kendall_tau(order_a, order_b):
    """Kendall tau-a between two rankings of the same items (lists, best first)."""
    items = [x for x in order_a if x in order_b]
    pa = {m: i for i, m in enumerate(order_a)}; pb = {m: i for i, m in enumerate(order_b)}
    c = d = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            s = (pa[items[i]] - pa[items[j]]) * (pb[items[i]] - pb[items[j]])
            c += s > 0; d += s < 0
    n = len(items) * (len(items) - 1) / 2
    return (c - d) / n if n else None


def fmt(x, d=3):
    return "—" if not fin(x) else (f"{x:+.{d}f}" if abs(x) < 1e4 else f"{x:+.3e}")


def objective_key(model_id: str, ds: str) -> str:
    if model_id == "IntensityFree":
        return "floor_1e-5"
    return F6 if abs(E.tie_threshold(ds) - E.D.TIE_STORED) < 1e-15 else "floor_1e-10"


def file_hash(p):
    return E.sha256(p) if p and Path(p).is_file() else None


def run_inputs(ckpt, ds, model, seed, data_dir):
    """(run location, result dict, identity of the inputs: paths and SHA-256 of checkpoint, configuration, training
    record and data file)."""
    loc = E.locate_run(ckpt, ds, model, seed)
    ck, cfg, rp = loc["checkpoint"], loc["config"], loc["result_path"]
    pkl = Path(data_dir) / f"{E.EASYTPP_ID[ds]}.pkl"
    ident = dict(checkpoint=str(ck.resolve()) if ck else None, checkpoint_sha256=file_hash(ck),
                 config=str(cfg.resolve()) if cfg else None, config_sha256=file_hash(cfg),
                 result=str(rp.resolve()) if rp else None, result_sha256=file_hash(rp),
                 data=str(pkl.resolve()), data_sha256=file_hash(pkl))
    return loc, loc["result"], ident


def run_problems(experiment, res, ident, data_pkl):
    """Problems of a run independent of its scores (module docstring)."""
    out = []
    if res.get("pilot") or res.get("epochs") != EPOCHS or res.get("max_seqs"):
        out.append(f"not a full run (requires {EPOCHS} requested epochs on the full data)")
    if res.get("checkpoint_sha256") and ident["checkpoint_sha256"] and res["checkpoint_sha256"] != ident["checkpoint_sha256"]:
        out.append("checkpoint changed after training")
    want = EXPERIMENTS[experiment]["mark_loss"]
    if want and res.get("mark_loss") is not True:
        out.append("trained without the mark term")
    if not want and res.get("mark_loss"):
        out.append("trained with the mark term")
    if ident["data_sha256"] is None:
        out.append("data file missing")
    elif res.get("data_sha256") and res["data_sha256"] != ident["data_sha256"]:
        out.append("data file differs from the one the run trained on")
    elif res.get("data_md5"):
        import hashlib
        if hashlib.md5(Path(data_pkl).read_bytes()).hexdigest() != res["data_md5"]:
            out.append("data file differs from the one the run trained on")
    if ident["checkpoint_sha256"] is None:
        out.append("no checkpoint")
    return out


# ------------------------------------------------------------------ score
def score(experiment, ckpt, data_dir, out, datasets=None, models=None, seeds=E.SEEDS, rule=IE.Rule(), audit=None,
          force=False):
    ex = experiment_of(experiment)
    joint = ex["mark_loss"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(out); (out / "scores").mkdir(parents=True, exist_ok=True)
    deltas = audit_rows(audit)
    for ds in datasets or ex["datasets"]:
        delta = (deltas.get(ds) or {}).get("delta")
        if not delta and not any(E.locate_run(ckpt, ds, m, s)["result"] is not None for m in models or E.MODELS for s in seeds):
            continue                                                # nothing trained on this data set
        if not delta:
            raise ValueError(f"{ds}: no audited resolution (run coarse_tpp.audit or pass --audit)")
        thr = E.tie_threshold(ds)
        pkl = Path(data_dir) / f"{E.EASYTPP_ID[ds]}.pkl"
        for model_id in models or E.MODELS:
            for seed in seeds:
                loc, res, ident = run_inputs(ckpt, ds, model_id, seed, data_dir)
                if res is None:
                    continue                                        # not trained or not finished
                dst = out / "scores" / f"{ds}__{model_id}__s{seed}.json"
                key = dict(version=VERSION, experiment=experiment, delta=delta, tie_threshold=thr,
                           rule=rule.identity(ds, model_id), inputs={k: v for k, v in ident.items() if k.endswith("sha256")})
                if dst.exists() and not force:
                    old = json.loads(dst.read_text())
                    if old.get("cache_key") == key:
                        continue
                r = dict(cache_key=key, inputs=ident, experiment=experiment, dataset=ds, model=model_id, seed=seed,
                         delta=delta, tie_threshold=thr, rule=rule.identity(ds, model_id), status=res.get("status"),
                         error=res.get("error"), epochs_run=res.get("epochs_run"), device_trained=res.get("device"),
                         device_scored=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
                problems = run_problems(experiment, res, ident, pkl)
                rr = res.get("result") if isinstance(res.get("result"), dict) else {}
                r["best_valid_epoch"] = rr.get("best_valid_epoch")
                bm = rr.get("best_metrics") or {}
                if ident["checkpoint"] and ident["config"] and ident["data_sha256"]:
                    cfg = yaml.safe_load(Path(ident["config"]).read_text())
                    model = IE.build_model(model_id, cfg, device, pkl)
                    model.load_state_dict(torch.load(ident["checkpoint"], map_location=device, weights_only=True))
                    r["n_params"] = sum(p.numel() for p in model.parameters())
                    pad = cfg["model_config"]["event_pad_index"]
                    for sp in ("dev", "test"):
                        seqs, _ = E.load_split(pkl, sp)
                        r[sp] = IE.score_split(model, model_id, seqs, pad, delta, rule, ds, thr, device, joint=joint)
                    k = objective_key(model_id, ds)
                    if bm.get("valid") and bm.get("test"):
                        f = lambda a, b: abs(a - b) if fin(a) and fin(b) else None
                        rep = dict(key=k, primary="joint" if joint else "time")
                        pairs = [("time", k, "time_ll")]
                        if joint:
                            pairs = [("joint", f"joint_{k}", "loglike"), ("mark", f"mark_{k}", "mark_ll")] + pairs
                        for name, ours, theirs in pairs:
                            for sp, lsp in (("dev", "valid"), ("test", "test")):
                                rep[f"{name}_{sp}"] = f(r[sp].get(ours), bm[lsp].get(theirs))
                        r["logged"] = {sp: bm[lsp] for sp, lsp in (("dev", "valid"), ("test", "test"))}
                        r["repro_logged"] = rep
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                r["problems"] = problems
                tmp = dst.with_suffix(f".tmp{os.getpid()}")
                tmp.write_text(json.dumps(r, indent=1)); os.replace(tmp, dst)
                t = r.get("test", {})
                msg = f"{ds} {model_id} s{seed} [{r['status']}]: test {F6} {t.get(F6)} {IV} {t.get(IV)}"
                if joint:
                    msg += f" joint_{F6} {t.get('joint_' + F6)} joint_{IV} {t.get('joint_' + IV)}"
                print(msg + f" problems {problems or 'none'}", flush=True)


# ------------------------------------------------------------------ reproduction reference
def repro(experiment, ckpt, data_dir, out, datasets=None, models=None, seeds=E.SEEDS, repeats=50):
    """EasyTPP's own Monte Carlo evaluation of each checkpoint, `repeats` times on dev and test (module docstring)."""
    ex = experiment_of(experiment)
    from .easytpp_train import set_mark_loss
    set_mark_loss(ex["mark_loss"])
    os.environ["EASYTPP_WORKERS"] = "0"
    E.activate("train")
    from easy_tpp.config_factory import Config
    from easy_tpp.runner import Runner
    from easy_tpp.utils import RunnerPhase
    out = Path(out); (out / "repro").mkdir(parents=True, exist_ok=True)
    for ds in datasets or ex["datasets"]:
        pkl = (Path(data_dir) / f"{E.EASYTPP_ID[ds]}.pkl").resolve()
        for model_id in models or E.MODELS:
            for seed in seeds:
                loc, res, ident = run_inputs(ckpt, ds, model_id, seed, data_dir)
                if res is None or ident["checkpoint"] is None:
                    continue
                dst = out / "repro" / f"{ds}__{model_id}__s{seed}.json"
                if dst.exists():
                    old = json.loads(dst.read_text())
                    if old.get("checkpoint_sha256") == ident["checkpoint_sha256"] and old.get("repeats") == repeats:
                        continue
                work = out / "repro" / "work" / f"{ds}__{model_id}__s{seed}"
                work.mkdir(parents=True, exist_ok=True)
                # the run's training configuration (equal to its config_in.yaml up to paths), on the current data file
                cfg = make_config(ds, model_id, seed, EPOCHS, work.resolve(), pkl, int(E.read_file(pkl)["dim_process"]),
                                  0 if torch.cuda.is_available() else -1)
                exp = f"{model_id}_{E.EASYTPP_ID[ds]}_train"
                cfg_path = work / "config.yaml"; cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    runner = Runner.build_from_config(Config.build_from_yaml_file(str(cfg_path), experiment_id=exp))
                model = runner.model_wrapper.model
                model.load_state_dict(torch.load(ident["checkpoint"], map_location=runner.model_wrapper.device, weights_only=True))
                k = objective_key(model_id, ds)
                logged_key = "loglike" if ex["mark_loss"] else "time_ll"
                rec = dict(experiment=experiment, dataset=ds, model=model_id, seed=seed,
                           checkpoint_sha256=ident["checkpoint_sha256"], repeats=repeats,
                           G=model.loss_integral_num_sample_per_step, ours_key=f"joint_{k}" if ex["mark_loss"] else k,
                           logged_key=logged_key,
                           device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
                loaders = {"dev": runner._data_loader.valid_loader(), "test": runner._data_loader.test_loader()}
                for sp, loader in loaders.items():
                    vals = []
                    for r in range(repeats):
                        torch.manual_seed(1000 + r)
                        with contextlib.redirect_stdout(io.StringIO()):
                            m = runner.run_one_epoch(loader, RunnerPhase.VALIDATE)
                        vals.append(float(m[logged_key]))
                    vals = np.asarray(vals)
                    logged = (((res.get("result") or {}).get("best_metrics") or {}).get("valid" if sp == "dev" else "test") or {}).get(logged_key)
                    sd = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
                    rec[sp] = dict(mc_mean=float(vals.mean()), mc_sd_single=sd, mc_se_mean=sd / math.sqrt(len(vals)),
                                   mc_min=float(vals.min()), mc_max=float(vals.max()), logged=logged,
                                   z_logged=(logged - float(vals.mean())) / sd if fin(logged) and sd > 0 else None)
                dst.write_text(json.dumps(rec, indent=1))
                print(f"repro {ds} {model_id} s{seed}: dev MC mean {rec['dev']['mc_mean']:.4f}, test {rec['test']['mc_mean']:.4f}",
                      flush=True)


# ------------------------------------------------------------------ report
def load_cells(experiment, out):
    """Score files of `out` with their problems: those recorded at scoring, stale inputs, and for joint / further the
    reproduction check."""
    out = Path(out)
    cells = []
    for p in sorted((out / "scores").glob("*.json")):
        c = json.loads(p.read_text())
        if c.get("experiment") != experiment:
            continue
        problems = list(c.get("problems") or [])
        if c.get("cache_key", {}).get("version") != VERSION:
            problems.append("stale score version; run score before report")
        inp = c.get("inputs") or {}
        for name in ("checkpoint", "config", "result", "data"):
            if inp.get(name) and inp.get(f"{name}_sha256") and file_hash(inp[name]) != inp[f"{name}_sha256"]:
                problems.append(f"stale input: {name} changed since scoring")
        if experiment != "time_only" and c.get("test"):
            problems = mc_gate(c, out, problems)
        c["problems"], c["quarantined"] = problems, bool(problems)
        cells.append(c)
    return cells


def mc_gate(c, out, problems):
    f = Path(out) / "repro" / f"{c['dataset']}__{c['model']}__s{c['seed']}.json"
    if not f.exists():
        c["repro"] = dict(ok=False)
        return problems + ["reproduction check missing (run repro)"]
    d = json.loads(f.read_text())
    if d.get("checkpoint_sha256") != (c.get("inputs") or {}).get("checkpoint_sha256"):
        c["repro"] = dict(ok=False)
        return problems + ["reproduction check is not bound to this checkpoint (run repro)"]
    rep, ok = dict(key=d["ours_key"], repeats=d["repeats"]), True
    for sp in ("dev", "test"):
        ours = c[sp].get(d["ours_key"])
        if not fin(ours):
            c["repro"] = dict(ok=False)
            return problems + ["reproduction check: score missing"]
        diff, se = abs(ours - d[sp]["mc_mean"]), d[sp]["mc_se_mean"]
        rep[f"{sp}_diff"], rep[f"{sp}_tol"] = diff, REPRO_TOL + MC_SE_MULT * se
        ok &= math.isfinite(diff) and diff <= REPRO_TOL + MC_SE_MULT * se
    rep["ok"] = bool(ok)
    c["repro"] = rep
    if not ok:
        problems.append(f"reproduction failed (dev {rep['dev_diff']:.4f} tol {rep['dev_tol']:.4f}, "
                        f"test {rep['test_diff']:.4f} tol {rep['test_tol']:.4f})")
    return problems


def valid(c):
    return bool(c.get("test")) and not c.get("quarantined")


def diverged(c):
    return not fin(c["test"].get(F6)) or c["test"][F6] < DIVERGED


def seed_means(cells, ds, split, key, models=E.MODELS):
    out = {}
    for m in models:
        s = stats([c[split].get(key) for c in cells if c["dataset"] == ds and c["model"] == m and valid(c)])
        if s["n"]:
            out[m] = s["mean"]
    return out


def per_seed(cells, ds, split, key):
    d = {}
    for c in cells:
        if c["dataset"] == ds and valid(c):
            d.setdefault(c["model"], {})[c["seed"]] = c[split].get(key)
    return d


def top(vals):
    o = rank(vals)
    return o[0] if o else None


def flipped_pairs(cells, ds, key_a, key_b, min_seeds=4):
    """Pairs (x, y) whose seed-paired sign is consistent on >= min_seeds seeds under both scores and opposite."""
    A, B = per_seed(cells, ds, "test", key_a), per_seed(cells, ds, "test", key_b)
    ms = sorted(set(A) & set(B), key=E.MODELS.index)
    out = []
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            x, y = ms[i], ms[j]
            seeds = [s for s in E.SEEDS if all(fin(D_.get(m, {}).get(s)) for D_ in (A, B) for m in (x, y))]
            sa = [np.sign(A[x][s] - A[y][s]) for s in seeds]; sb = [np.sign(B[x][s] - B[y][s]) for s in seeds]
            da = 1 if sum(v > 0 for v in sa) >= min_seeds else -1 if sum(v < 0 for v in sa) >= min_seeds else 0
            db = 1 if sum(v > 0 for v in sb) >= min_seeds else -1 if sum(v < 0 for v in sb) >= min_seeds else 0
            if da and db and da != db:
                out.append((x, y, da, db, len(seeds)))
    return out


def model_summary(cells, ds, keys):
    """Per model: runs, status-ok runs, valid runs, diverged valid runs, and test means of keys over the valid runs that
    did not diverge (`table`: over the diverged ones when all diverged, flagged `all_diverged`)."""
    out = {}
    for m in E.MODELS:
        cs = [c for c in cells if c["dataset"] == ds and c["model"] == m]
        if not cs:
            continue
        ok = [c for c in cs if valid(c)]
        div = [c for c in ok if diverged(c)]
        nd = [c for c in ok if c not in div]
        src = nd or div
        out[m] = dict(runs=len(cs), status_ok=sum(c["status"] == "ok" for c in cs), valid=len(ok), diverged=len(div),
                      diverged_seeds=sorted(c["seed"] for c in div), all_diverged=bool(ok) and not nd,
                      table={k: stats([c["test"].get(k) for c in src]) for k in keys},
                      quarantined={c["seed"]: c["problems"] for c in cs if c.get("quarantined")})
    return out


def ms(st, d=2):
    if not st or not st["n"]:
        return "—"
    if st["sd"] is None:
        return f"{st['mean']:+.{d}f}"
    return f"{st['mean']:+.{d}f} ± {st['sd']:.{d}f}"


def table_rows(summary, keys, labels):
    L = ["| model | " + " | ".join(labels) + " | runs used | diverged |", "|---|" + "---:|" * len(keys) + "---|---|"]
    for m, r in summary.items():
        if r["valid"] == 0:
            why = "training failed on every seed" if r["status_ok"] == 0 else "excluded on every seed (failed or not reproduced)"
            L.append(f"| {m} | {why} |" + " |" * (len(keys) - 1) + f" {r['valid']}/{r['runs']} | — |"); continue
        star = "*" if r["all_diverged"] else ""
        L.append(f"| {m} | " + " | ".join(ms(r["table"][k]) + star for k in keys) + f" | {r['valid']}/{r['runs']} | "
                 f"{r['diverged']}/{r['valid']} |")
    return L


def s2p2_minus_nhp(cells, ds, key_floor, key_int):
    """Per seed S2P2 - NHP under the floor and the interval key (test), on the seeds with finite values under both."""
    A, B = per_seed(cells, ds, "test", key_floor), per_seed(cells, ds, "test", key_int)
    seeds = [s for s in E.SEEDS if all(fin(D_.get(m, {}).get(s)) for D_ in (A, B) for m in ("S2P2", "NHP"))]
    f = {s: A["S2P2"][s] - A["NHP"][s] for s in seeds}
    g = {s: B["S2P2"][s] - B["NHP"][s] for s in seeds}
    return dict(seeds=seeds, floor=f, interval=g, floor_mean=float(np.mean(list(f.values()))) if f else None,
                interval_mean=float(np.mean(list(g.values()))) if g else None)


def diff_lines(name, d):
    return [f"- {name}: floor " + ", ".join(f"s{s} {v:+.4f}" for s, v in d["floor"].items())
            + f" (mean {fmt(d['floor_mean'], 4)}); interval " + ", ".join(f"s{s} {v:+.4f}" for s, v in d["interval"].items())
            + f" (mean {fmt(d['interval_mean'], 4)})"]


def problems_lines(cells):
    bad = [c for c in cells if c["status"] != "ok" or c.get("quarantined")]
    if not bad:
        return ["Training failures and quarantined runs: none.", ""]
    return ["Training failures (status != ok) and quarantined runs:", ""] + [
        f"- {c['dataset']} {c['model']} s{c['seed']}: status {c['status']}"
        + (f" ({(c.get('error') or '')[:100]})" if c.get("error") else "")
        + (f"; quarantined: {'; '.join(c['problems'])}" if c.get("quarantined") else "") for c in bad] + [""]


def share(x):
    return "—" if x is None else f"{x:.4f}"


def tie_share(cells, ds):
    c = next((c for c in cells if c["dataset"] == ds and c.get("test")), None)
    return c["test"]["n_tie"] / c["test"]["n"] if c else None


def seed42_inside(cells, csv_path):
    """Seed-42 models of Supplementary Table S2 (trained-models archive, intensity_models/scores.csv): whether their test
    floor 1e-6 score lies within the range of the five retrained seeds (all runs with a checkpoint)."""
    import csv
    rows = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["dataset"] in EXPERIMENTS["time_only"]["datasets"] and r.get("test_eps6"):
                rows[(r["dataset"], r["model"])] = float(r["test_eps6"])
    out = {}
    for (ds, m), v in sorted(rows.items()):
        new = [c["test"][F6] for c in cells if c["dataset"] == ds and c["model"] == m and valid(c) and fin(c["test"].get(F6))]
        if new:
            out[f"{ds}/{m}"] = dict(seed42=v, min=min(new), max=max(new), inside=min(new) <= v <= max(new))
    return out


def report_time_only(cells, out, seed42=None):
    keys = ("floor_1e-10", F6, "floor_1e-5", IV)
    L = ["# Retrained EasyTPP models, time-only objective, seeds 0-4 (Supplementary Table S3)", "",
         "Test split, mean per target event (nats). Mean ± SD over the runs with a checkpoint that did not diverge "
         "(test floor 1e-6 score below -100 or non-finite); * = every run diverged, mean over the diverged runs. "
         "Orders and Kendall's tau use these means.", ""] + problems_lines(cells)
    summ = {}
    for ds in EXPERIMENTS["time_only"]["datasets"]:
        if not any(c["dataset"] == ds for c in cells):
            continue
        S = model_summary(cells, ds, keys + (f"{F6}_tie", f"{IV}_tie"))
        tie = tie_share(cells, ds)
        L += [f"## {E.NAMES[ds]} (test tie share {share(tie)})", ""]
        L += table_rows(S, keys + (f"{F6}_tie", f"{IV}_tie"),
                        ("floor 1e-10", "floor 1e-6", "floor 1e-5", "interval", "ties: floor 1e-6", "ties: interval"))
        means = {k: {m: r["table"][k]["mean"] for m, r in S.items() if r["valid"]} for k in keys}
        orders = {k: rank(means[k]) for k in keys}
        spread = {m: max(abs(c["test"]["floor_1e-5"] - c["test"]["floor_1e-10"]) for c in cells
                         if c["dataset"] == ds and c["model"] == m and valid(c) and not diverged(c))
                  for m, r in S.items() if r["valid"] > r["diverged"]}
        tau = dict(floor_1e6_vs_interval=kendall_tau(orders[F6], orders[IV]),
                   floor_1e10_vs_interval=kendall_tau(orders["floor_1e-10"], orders[IV]),
                   floor_1e10_vs_floor_1e5=kendall_tau(orders["floor_1e-10"], orders["floor_1e-5"]))
        d = s2p2_minus_nhp(cells, ds, F6, IV)
        L += ["", f"- order by floor 1e-6: {' > '.join(orders[F6])}", f"- order by interval: {' > '.join(orders[IV])}",
              f"- order by floor 1e-10: {' > '.join(orders['floor_1e-10'])}",
              f"- Kendall tau: floor 1e-6 vs interval {fmt(tau['floor_1e6_vs_interval'], 2)}; floor 1e-10 vs interval "
              f"{fmt(tau['floor_1e10_vs_interval'], 2)}; floor 1e-10 vs floor 1e-5 {fmt(tau['floor_1e10_vs_floor_1e5'], 2)}",
              "- largest |floor 1e-5 - floor 1e-10| per run (non-diverged): "
              + ", ".join(f"{m} {v:.4f}" for m, v in spread.items())] + diff_lines("S2P2 - NHP per seed", d) + [""]
        summ[ds] = dict(models=S, orders=orders, kendall_tau=tau, floor_spread_max=spread, s2p2_minus_nhp=d, tie_share_test=tie)
    if seed42:
        ins = seed42_inside(cells, seed42)
        summ["seed42_inside"] = ins
        L += ["## Seed-42 models (Supplementary Table S2) within the range of the five seeds (test floor 1e-6)", "",
              f"{sum(v['inside'] for v in ins.values())} of {len(ins)}: "
              + ", ".join(f"{k} {'inside' if v['inside'] else 'outside'} ({v['seed42']:+.3f} vs {v['min']:+.3f}..{v['max']:+.3f})"
                          for k, v in ins.items()), ""]
    return L, summ


def report_joint(cells, out):
    ds = "retweet_sub"
    keys = (F6, IV, "mark_rec", f"joint_{F6}", f"joint_{IV}", f"joint_{IV}2")
    L = ["# Retrained EasyTPP models with the mark term, Retweet subsample (Supplementary Table S4)", "",
         "Test split, mean per target event (nats); mean ± SD over the runs used (reproduction check passed, not "
         "quarantined) that did not diverge. joint floor 1e-6 = floor score plus the mark log-likelihood at the floored "
         "gap; joint interval = log-probability of the recorded cell and mark; mark_rec = mark log-likelihood at the "
         "recorded time; joint_interval2 = interval + mark_rec (approximation).", ""] + problems_lines(cells)
    S = model_summary(cells, ds, keys)
    L += table_rows(S, keys, ("floor 1e-6", "interval", "mark term", "joint floor 1e-6", "joint interval", "joint interval 2"))
    cc = [abs(c["test"].get("cellcheck")) for c in cells if valid(c) and fin(c["test"].get("cellcheck"))]
    okeys = (f"joint_{F6}", f"joint_{IV}", f"joint_{IV}2", F6, IV)
    orders = {k: rank(seed_means(cells, ds, "test", k)) for k in okeys}
    no_fnn = lambda o: [m for m in o if m != "FullyNN"]
    taus = {f"{a} vs {b}": (kendall_tau(orders[a], orders[b]), kendall_tau(no_fnn(orders[a]), no_fnn(orders[b])))
            for a, b in ((f"joint_{F6}", f"joint_{IV}"), (F6, IV))}
    sel = {k: top(seed_means(cells, ds, "dev", k)) for k in okeys}
    d = s2p2_minus_nhp(cells, ds, f"joint_{F6}", f"joint_{IV}")
    L += ["", f"cell quadrature check: max |mean cellcheck| (test, runs used) {fmt(max(cc) if cc else None, 6)}", "",
          "Orders by seed-mean test score (runs used):"] + [f"- {k}: {' > '.join(o)}" for k, o in orders.items()] + \
         ["", "Kendall tau (all architectures / without FullyNN):"] + [f"- {k}: {fmt(a, 2)} / {fmt(b, 2)}" for k, (a, b) in taus.items()] + \
         ["", "Validation-selected architecture (highest seed-mean dev score): "
          + ", ".join(f"{k} {v}" for k, v in sel.items()), ""] + diff_lines("S2P2 - NHP per seed (joint scores)", d) + [""]
    return L, dict(retweet_sub=dict(models=S, orders=orders, kendall_tau=taus, selected_dev=sel, s2p2_minus_nhp=d,
                                    cellcheck_max=max(cc) if cc else None))


def report_further(cells, out):
    keys = ("floor_1e-10", F6, "floor_1e-5", IV)
    L = ["# Retrained EasyTPP models on Volcano, Sepsis and BPIC12, time-only objective (Supplementary Table S5)", "",
         "Test split, mean per target event (nats); mean ± SD over the runs used (reproduction check passed, not "
         "quarantined) that did not diverge.", ""] + problems_lines(cells)
    summ = {}
    for ds in EXPERIMENTS["further"]["datasets"]:
        if not any(c["dataset"] == ds for c in cells):
            continue
        S = model_summary(cells, ds, keys)
        tie = tie_share(cells, ds)
        L += [f"## {E.NAMES[ds]} (test tie share {share(tie)})", ""]
        L += table_rows(S, keys, ("floor 1e-10", "floor 1e-6", "floor 1e-5", "interval"))
        o6, og = rank(seed_means(cells, ds, "test", F6)), rank(seed_means(cells, ds, "test", IV))
        flips = flipped_pairs(cells, ds, F6, IV)
        sel = {k: top(seed_means(cells, ds, "dev", k)) for k in keys}
        d = s2p2_minus_nhp(cells, ds, F6, IV)
        L += ["", f"- order by floor 1e-6: {' > '.join(o6)}", f"- order by interval: {' > '.join(og)}",
              f"- Kendall tau floor 1e-6 vs interval {fmt(kendall_tau(o6, og), 2)}",
              f"- reversed pairs (sign consistent on >= 4 seeds under both scores): "
              + ("; ".join(f"{x}/{y}" for x, y, *_ in flips) or "none"),
              "- validation-selected architecture: " + ", ".join(f"{k} {v}" for k, v in sel.items())]
        L += diff_lines("S2P2 - NHP per seed", d) + [""]
        summ[ds] = dict(models=S, order_floor_1e6=o6, order_interval=og, kendall_tau=kendall_tau(o6, og), flips=flips,
                        selected_dev=sel, selection_changes=sel[F6] != sel[IV], s2p2_minus_nhp=d, tie_share_test=tie)
    return L, summ


def report(experiment, out, seed42=None):
    experiment_of(experiment)
    out = Path(out)
    cells = load_cells(experiment, out)
    if experiment == "time_only":
        L, summ = report_time_only(cells, out, seed42)
    else:
        L, summ = {"joint": report_joint, "further": report_further}[experiment](cells, out)
    rules = sorted({json.dumps(c.get("rule"), sort_keys=True) for c in cells})
    L += [f"Runs: {len(cells)}; integration settings: {', '.join(rules)}"]
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "summary.json").write_text(json.dumps(dict(experiment=experiment, per_dataset=summ), indent=1, default=str))
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("score", "repro", "report"))
    ap.add_argument("--experiment", required=True, choices=tuple(EXPERIMENTS))
    ap.add_argument("--ckpt", help="run root")
    ap.add_argument("--data", help="directory of the prepared EasyTPP pickles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--models", nargs="*", choices=E.MODELS)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(E.SEEDS))
    ap.add_argument("--audit", default=None)
    ap.add_argument("--per-panel", type=int, default=32)
    ap.add_argument("--strata", type=int, default=16384)
    ap.add_argument("--strat-seed", type=int, default=0)
    ap.add_argument("--cell-panels", type=int, default=1500)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed42", default=None, help="report, time_only: intensity_models/scores.csv of the trained-models "
                    "archive (seed-42 models within the range of the five seeds)")
    a = ap.parse_args(argv)
    if a.cmd in ("score", "repro") and not (a.ckpt and a.data):
        ap.error("--ckpt and --data are required")
    if a.cmd == "score":
        rule = IE.Rule(per_panel=a.per_panel, strata=a.strata, strat_seed=a.strat_seed, cell_panels=a.cell_panels)
        score(a.experiment, a.ckpt, a.data, a.out, a.datasets, a.models, tuple(a.seeds), rule, a.audit, a.force)
    elif a.cmd == "repro":
        repro(a.experiment, a.ckpt, a.data, a.out, a.datasets, a.models, tuple(a.seeds), a.repeats)
    else:
        print("\n".join(report(a.experiment, a.out, a.seed42)))


if __name__ == "__main__":
    main()
