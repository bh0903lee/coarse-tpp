"""Cell grids of the experiments of the paper. Usage:

    python -m coarse_tpp.experiments EXPERIMENT --out runs/EXPERIMENT [--audit results/audit/audit.json]

writes RUN/grid.jsonl (one line per cell) and RUN/configs/<cell_id>.json; every cell trains with
coarse_tpp.train into RUN/cells/<cell_id>/. An existing config file is never changed. The resolution delta of a
recorded data set is read from the audit table (coarse_tpp.audit).

Experiments (paper section in brackets):
  synthetic               seven processes x 5 realizations x delta in {0, .01, .03, .1, .3, 1}, GRU, seven
                          treatments (delta = 0: latent)                                            [6.1, S1, S2]
  rank_change             mixture head, nine data sets, GRU and Transformer, eight treatments       [5.3, 6.2, S2]
  intensity_heads         RMTPP and THP heads, six data sets, five treatments                       [6.2, S1, S2]
  process_logs            seven business process logs, marks as inputs, GRU and Transformer         [5.3, 6.2, S2]
  delta_misspecification  interval likelihood (and uniform jitter) trained with delta_model = m x delta  [8, S3]
  iw_jitter               multi-sample jitter, S in {1, 4, 16, 64}                                  [6.2, S3]
  iw_jitter_large         SC and SP at delta = 1, S in {64, 256, 1024, 4096}                        [8, S3]
  inbin_smoothing         triangular kernel and two interval / jitter mixtures                       [8, S3]
  inbin_smoothing_seeds   triangular vs interval on the tied real data sets, 40 new seeds            [8, S3]
  marks                   mark head with the time treatments, and a control trained on marks alone  [8, S3]
  coarsened_real          ContTime, Taxi, StackOverflow coarsened to delta = r x median gap        [6.1, S3]
  resplit                 rank_change (GRU, five treatments) on new random splits                  [5.3, 6.2, S2]
  new_datasets            MemeTracker, MOOC and Wikipedia (control)                                [5.3, 6.2, S2]
  floor_convention        GRU + mixture models trained and selected under the floor 1e-6 convention, two
                          parameterizations of the mixture head, 21 data sets; models of the kind that
                          coarse_tpp.rescore re-scores, an approximation of the trained models of the
                          paper (see README, "Reproduction scope")                                  [5.1, S2]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import synthetic as Syn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "results" / "audit" / "audit.json"
LRS = {"lr3e4": 3e-4, "lr1e3": 1e-3}
OPT = dict(batch_size=32, max_epochs=400, patience=20, grad_clip=1.0)
GRU = {"type": "gru", "d_model": 64}
TRF = {"type": "transformer", "d_model": 64, "n_layers": 2, "n_heads": 4, "d_ff": 128, "dropout": 0.0}
REAL_SEEDS = (0, 1, 2, 3, 4)

# arm -> (treatment, extra config)
FLOOR_ARMS = {"floor_1e-10": 1e-10, "floor_1e-6": 1e-6, "floor_1e-5": 1e-5}


def arm_spec(arm: str):
    if arm in FLOOR_ARMS:
        return "floor", dict(eps=FLOOR_ARMS[arm])
    if arm.startswith("iw_jitter_S"):
        return "iw_jitter", dict(iw_samples=int(arm[len("iw_jitter_S"):]))
    if arm.startswith("interval_jitter_mix_"):
        return "interval_jitter_mix", dict(mix_lambda=float(arm[len("interval_jitter_mix_"):]))
    if arm == "mark_only":
        return "floor", dict(eps=1e-10, mark_head=True, mark_only=True)
    if arm in ("latent", "interval", "triangular", "point_mass", "jitter_gap", "jitter_time", "jitter_time_gauss"):
        return arm, {}
    raise ValueError(arm)


# ------------------------------------------------------------------ data sets
SYN_PROCS = ("SP", "SR", "SC", "H1", "H2", "NSP", "NSR")
SYN_JUDGED = ("SP", "SR", "SC", "H1", "H2")
SYN_SEEDS = (0, 1)
SYN_ARMS = ("floor_1e-10", "floor_1e-6", "jitter_gap", "jitter_time", "point_mass", "interval", "triangular")
SYN_NON_INTERVAL = ("floor_1e-10", "floor_1e-6", "jitter_gap", "jitter_time", "point_mass")

RANK_DATASETS = ("retweet", "email_eu", "volcano", "stackoverflow", "collegemsg", "conttime_q02", "conttime_q10",
                 "taxi", "synthetic_exp")
RANK_CONTROLS = ("taxi", "synthetic_exp")
RANK_ARMS = ("floor_1e-10", "floor_1e-6", "floor_1e-5", "jitter_gap", "jitter_time", "jitter_time_gauss",
             "point_mass", "interval")
SAME_DENSITY_ARMS = ("floor_1e-10", "floor_1e-6", "floor_1e-5", "jitter_gap", "interval")

INTENSITY_DATASETS = ("email_eu", "volcano", "stackoverflow", "conttime_q10", "retweet_sub", "taxi")
INTENSITY_HEADS = ("rmtpp", "thp")
INTENSITY_ARMS = ("floor_1e-10", "floor_1e-6", "jitter_gap", "jitter_time", "interval")

PROCESS_LOGS = ("bpi_bpic12", "bpi_bpic13i", "bpi_bpic20dd", "bpi_bpic20ptc", "bpi_bpic20rfp", "bpi_helpdesk",
                "bpi_sepsis")
PROCESS_ARMS = ("floor_1e-10", "floor_1e-6", "jitter_gap", "jitter_time", "point_mass", "interval")

TIED_REAL = ("email_eu", "volcano", "conttime_q10", "retweet_sub", "stackoverflow")
MISSPEC_MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)
MISSPEC_ARMS = ("interval", "jitter_gap")
MISSPEC_SYN_REALIZATIONS = (1, 2, 3)

IW_S = (1, 4, 16, 64)
IW_ARMS = tuple(f"iw_jitter_S{S}" for S in IW_S) + ("interval", "jitter_gap")
IW_LARGE_SETTINGS = (("SC", 1.0), ("SP", 1.0))
IW_LARGE_S = (64, 256, 1024, 4096)
IW_LARGE_ARMS = tuple(f"iw_jitter_S{S}" for S in IW_LARGE_S) + ("interval", "jitter_gap")

SMOOTH_ARMS = ("triangular", "interval_jitter_mix_0.1", "interval_jitter_mix_0.5", "interval", "jitter_gap")
SMOOTH_SYN_SEEDS = (4, 5)
SMOOTH_SEEDS_ARMS = ("triangular", "interval")
SMOOTH_NEW_SEEDS = tuple(range(100, 140))

MARK_DATASETS = ("stackoverflow", "retweet_sub", "conttime_q10", "taxi", "bpi_bpic12", "bpi_sepsis", "bpi_helpdesk")
MARK_ARMS = ("floor_1e-10", "floor_1e-6", "jitter_gap", "point_mass", "interval")

COARSEN_BASES = ("conttime", "taxi", "stackoverflow")
COARSEN_R = (0.03, 0.1, 0.3, 1.0)
COARSEN_ARMS = ("floor_1e-10", "floor_1e-6", "jitter_gap", "jitter_time", "point_mass", "interval")

RESPLIT_PARENTS = ("retweet", "email_eu", "volcano", "stackoverflow", "collegemsg", "conttime_q02", "conttime_q10",
                   "taxi")
RESPLIT_ARMS = ("floor_1e-10", "floor_1e-6", "jitter_gap", "point_mass", "interval")
NEW_DATASETS = ("memetracker", "mooc", "wikipedia")
NEW_CONTROLS = ("wikipedia",)


FLOOR_CONV_DATASETS = ("amazon", "bookorder", "collegemsg", "conttime", "conttime_q02", "conttime_q10", "earthquake",
                       "email_eu", "mimic", "retweet", "retweet_sub", "stackoverflow", "stackoverflow_sub", "taobao",
                       "taxi", "volcano", "synthetic_corr_pareto_01", "synthetic_corr_pareto_02", "synthetic_exp",
                       "synthetic_powerlaw_01", "synthetic_powerlaw_02")
FLOOR_CONV_PARAMS = ("standard", "sigma_sigmoid")
FLOOR_CONV_SEEDS = tuple(range(10))


def rcode(r):
    return f"{r:g}".replace(".", "p")


def coarsened_name(base, r):
    return f"coarsened_{base}_r{rcode(r)}"


COARSENED = tuple(coarsened_name(b, r) for b in COARSEN_BASES for r in COARSEN_R)
RESPLIT = tuple(f"{p}_resplit" for p in RESPLIT_PARENTS)
RESPLIT_CONTROLS = ("taxi_resplit",)


# ------------------------------------------------------------------ cells
def cell(experiment, run, ds, arm, seed, lr_tag, delta, encoder=GRU, head="mix", K=8, dmult=1.0, tag=None):
    treatment, extra = arm_spec(arm)
    name = tag or arm
    enc_tag = {"gru": "gru", "transformer": "trf"}[encoder["type"]]
    cid = f"{experiment}__{name}__{head}_{enc_tag}_K{K}__{ds}__{lr_tag}__s{seed}"
    c = dict(cell_id=cid, experiment=experiment, arm=name, dataset=ds, seed=seed, encoder=dict(encoder), head=head,
             K=K, treatment=treatment, eps=extra.pop("eps", None), delta=float(delta),
             delta_model=float(delta) * dmult, optim=dict(OPT, lr=LRS[lr_tag]), lr_tag=lr_tag, threads=2,
             out_dir=str((Path(run) / "cells" / cid).resolve()))
    c.update(extra)
    return c


def audit_table(path=None) -> dict:
    p = Path(path or DEFAULT_AUDIT)
    if not p.exists():
        raise SystemExit(f"no audit table at {p} (run scripts/audit.py first)")
    return {r["dataset"]: r for r in json.loads(p.read_text())}


_ONLY = None          # optional data-set filter of the command line (quick tests)


def _keep(ds) -> bool:
    return _ONLY is None or ds in _ONLY


def delta_of(audit: dict, ds: str) -> float:
    if ds not in audit or not audit[ds].get("delta"):
        raise SystemExit(f"{ds}: no resolution in the audit table")
    return float(audit[ds]["delta"])


def _real(experiment, run, audit, datasets, arms, encoders=(GRU,), heads=("mix",), marks=False, seeds=REAL_SEEDS,
          lrs=tuple(LRS), **kw):
    out = []
    for ds in filter(_keep, datasets):
        d = delta_of(audit, ds)
        for enc in encoders:
            enc = dict(enc, marks=True) if marks else enc
            for h in heads:
                for a in arms:
                    if h != "mix" and a == "point_mass":
                        continue
                    out += [cell(experiment, run, ds, a, s, t, d, encoder=enc, head=h, **kw) for t in lrs for s in seeds]
    return out


def _synthetic(experiment, run, settings, arms, seeds, realizations=Syn.REALIZATIONS, **kw):
    out = []
    for proc, d in settings:
        for r in realizations:
            ds = Syn.name(proc, r, d)
            if _keep(ds):
                out += [cell(experiment, run, ds, a, s, t, d, **kw) for a in arms for t in LRS for s in seeds]
    return out


def cells_synthetic(run, audit=None):
    out = []
    for proc in SYN_PROCS:
        for r in Syn.REALIZATIONS:
            for d in Syn.DELTAS:
                ds = Syn.name(proc, r, d)
                for a in (("latent",) if d == 0 else SYN_ARMS):
                    out += [cell("synthetic", run, ds, a, s, t, d) for t in LRS for s in SYN_SEEDS]
    return out


def cells_rank_change(run, audit):
    return _real("rank_change", run, audit, RANK_DATASETS, RANK_ARMS, (GRU, TRF))


def cells_intensity_heads(run, audit):
    return _real("intensity_heads", run, audit, INTENSITY_DATASETS, INTENSITY_ARMS, heads=INTENSITY_HEADS)


def cells_process_logs(run, audit):
    return _real("process_logs", run, audit, PROCESS_LOGS, PROCESS_ARMS, (TRF, GRU), marks=True)


def misspec_tag(arm, mult):
    return f"{arm}_dx{mult:g}"


def cells_delta_misspecification(run, audit):
    out = []
    for proc, d in Syn.main_settings():
        for r in MISSPEC_SYN_REALIZATIONS:
            ds = Syn.name(proc, r, d)
            if not _keep(ds):
                continue
            out += [cell("delta_misspecification", run, ds, "interval", 0, t, d, dmult=0.5, tag=misspec_tag("interval", 0.5))
                    for t in LRS]
    for ds in filter(_keep, TIED_REAL):
        d = delta_of(audit, ds)
        for a in MISSPEC_ARMS:
            for m in MISSPEC_MULTS:
                out += [cell("delta_misspecification", run, ds, a, s, t, d, dmult=m, tag=misspec_tag(a, m))
                        for t in LRS for s in REAL_SEEDS]
    return out


def cells_iw_jitter(run, audit):
    return (_synthetic("iw_jitter", run, Syn.main_settings(), IW_ARMS, SYN_SEEDS)
            + _real("iw_jitter", run, audit, TIED_REAL, IW_ARMS))


def cells_iw_jitter_large(run, audit=None):
    return _synthetic("iw_jitter_large", run, IW_LARGE_SETTINGS, IW_LARGE_ARMS, SYN_SEEDS)


def cells_inbin_smoothing(run, audit):
    return (_synthetic("inbin_smoothing", run, Syn.main_settings(), SMOOTH_ARMS, SMOOTH_SYN_SEEDS)
            + _real("inbin_smoothing", run, audit, TIED_REAL, SMOOTH_ARMS))


def cells_inbin_smoothing_seeds(run, audit):
    return _real("inbin_smoothing_seeds", run, audit, TIED_REAL, SMOOTH_SEEDS_ARMS, seeds=SMOOTH_NEW_SEEDS)


def cells_marks(run, audit):
    out = []
    enc = dict(GRU, marks=True)
    for ds in filter(_keep, MARK_DATASETS):
        d = delta_of(audit, ds)
        for a in MARK_ARMS + ("mark_only",):
            for t in LRS:
                for s in REAL_SEEDS:
                    c = cell("marks", run, ds, a, s, t, d, encoder=enc)
                    c["mark_head"] = True
                    out.append(c)
    return out


def cells_coarsened_real(run, audit):
    return _real("coarsened_real", run, audit, COARSENED, COARSEN_ARMS, (GRU, TRF))


def cells_resplit(run, audit):
    return _real("resplit", run, audit, RESPLIT, RESPLIT_ARMS)


def cells_new_datasets(run, audit):
    return _real("new_datasets", run, audit, NEW_DATASETS, RESPLIT_ARMS)


def cells_floor_convention(run, audit=None):
    """Models of the eps convention: trained with floor 1e-6 on the stored gaps (ties stored as 1e-6, also as inputs),
    early stopping and checkpoint by the dev floor log-likelihood ("own"), delta = 0 for the statistics."""
    out = []
    for ds in FLOOR_CONV_DATASETS:
        for p in FLOOR_CONV_PARAMS:
            for t in LRS:
                for s in FLOOR_CONV_SEEDS:
                    c = cell("floor_convention", run, ds, "floor_1e-6", s, t, 0.0, tag=f"floor_1e-6_{p}")
                    c.update(mix_param=p, primary_selection="own", raw_gaps=True)
                    if p != "standard":
                        c["sig_max"] = 4.0
                    out.append(c)
    return out


BUILDERS = {"synthetic": cells_synthetic, "rank_change": cells_rank_change, "intensity_heads": cells_intensity_heads,
            "process_logs": cells_process_logs, "delta_misspecification": cells_delta_misspecification,
            "iw_jitter": cells_iw_jitter, "iw_jitter_large": cells_iw_jitter_large,
            "inbin_smoothing": cells_inbin_smoothing, "inbin_smoothing_seeds": cells_inbin_smoothing_seeds,
            "marks": cells_marks, "coarsened_real": cells_coarsened_real, "resplit": cells_resplit,
            "new_datasets": cells_new_datasets, "floor_convention": cells_floor_convention}
NEEDS_AUDIT = set(BUILDERS) - {"synthetic", "iw_jitter_large", "floor_convention"}


def write(experiment: str, run: Path, audit_path=None, cells=None):
    """Write the grid of `experiment` (or the given cell list) into `run`."""
    run = Path(run).resolve()
    audit = audit_table(audit_path) if experiment in NEEDS_AUDIT and cells is None else None
    cs = cells if cells is not None else BUILDERS[experiment](run, audit)
    ids = [c["cell_id"] for c in cs]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate cell ids")
    (run / "configs").mkdir(parents=True, exist_ok=True)
    lines = []
    for c in cs:
        p = run / "configs" / f"{c['cell_id']}.json"
        txt = json.dumps(c, indent=1, sort_keys=True)
        if p.exists() and p.read_text() != txt:
            raise SystemExit(f"a cell config would change: {p}")
        if not p.exists():
            p.write_text(txt)
        lines.append(json.dumps(dict(cell_id=c["cell_id"], arm=c["arm"], dataset=c["dataset"], seed=c["seed"],
                                     config=str(p))))
    (run / "grid.jsonl").write_text("\n".join(lines) + "\n")
    (run / "meta.json").write_text(json.dumps(dict(experiment=experiment, n_cells=len(cs)), indent=1))
    return cs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=sorted(BUILDERS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit", default=None, help="audit table (default results/audit/audit.json)")
    ap.add_argument("--datasets", nargs="*", default=None, help="keep only the cells of these data sets")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="keep only the cells of these seeds")
    ap.add_argument("--max-epochs", type=int, default=None, help="override the epoch budget (quick tests only)")
    a = ap.parse_args(argv)
    global _ONLY
    _ONLY = set(a.datasets) if a.datasets else None
    cells = None
    if a.datasets or a.seeds or a.max_epochs:
        run = Path(a.out).resolve()
        audit = audit_table(a.audit) if a.experiment in NEEDS_AUDIT else None
        cells = [c for c in BUILDERS[a.experiment](run, audit) if (not a.datasets or c["dataset"] in a.datasets)
                 and (not a.seeds or c["seed"] in a.seeds)]
        if a.max_epochs:
            for c in cells:
                c["optim"]["max_epochs"] = a.max_epochs
    cs = write(a.experiment, Path(a.out), a.audit, cells=cells)
    print(f"{a.experiment}: {len(cs)} cells -> {a.out}")


if __name__ == "__main__":
    main()
