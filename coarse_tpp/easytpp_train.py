"""Training of the EasyTPP models of Section 5.2 and Supplementary Tables S3-S5 with the vendored, patched EasyTPP
(third_party/easy_tpp_train, PATCHES.md). Three experiments, nine architectures, seeds 0-4:

  time_only   Email-EU, Retweet subsample, CollegeMsg; time-only objective (Table S3, Fig. S5, the threshold-
              exceedance comparison of Section S4)
  joint       Retweet subsample; the mark term in the objective (EASYTPP_MARK_LOSS=1; Table S4, Fig. S5)
  further     Volcano, Sepsis, BPIC12; time-only objective, activities of the process logs as marks (Table S5, Fig. S5)

    python -m coarse_tpp.easytpp_train prepare --out runs/easytpp/data [--datasets ...]
    python -m coarse_tpp.easytpp_train train --experiment time_only --datasets email_eu --models NHP S2P2 \
        --seeds 0 1 2 3 4 --data runs/easytpp/data --out runs/easytpp/time_only
    python -m coarse_tpp.easytpp_train config --experiment joint --dataset retweet_sub --model NHP --seed 0 \
        --data runs/easytpp/data --out runs/easytpp/joint          (writes the configuration only)

Settings (all architectures): hidden size 64, Adam with learning rate 1e-3, cosine decay after a 1% warm-up, batch
size 32, 100 epochs, no shuffling, 10 Monte Carlo samples per gap for the compensator, the checkpoint EasyTPP saves at
its best validation log-likelihood under the training objective. Model-specific settings: SPECS. The trainer seed is
the run's seed. AttNHP on CollegeMsg was trained in four row chunks per batch (EASYTPP_MICROBATCH, the same gradient as
for the whole batch; memory only); this is the default for that run (--microbatch overrides it).

Run directory <out>/<data set>/<model>/s<seed>/: config_in.yaml (the configuration EasyTPP reads; it equals that of
the trained-models archive up to the paths), EasyTPP's own <identifier>/<model>/train_<time>/{models/saved_model.pt,
log, *_output.yaml}, and result.json, written last: status 'ok' | 'nan_stop' (EasyTPP stopped on a NaN epoch loss;
fewer epochs than requested) | 'error' (exception, traceback recorded); the best checkpoint saved before a failure is
kept. result.json records the experiment, the mark-loss switch as the loss code saw it, the data file and its SHA-256,
the device and the checkpoint SHA-256. Finished runs (result.json present) are skipped unless --force.

The paper's models were trained on single GPUs (NVIDIA H100 NVL or RTX A6000) with one CPU thread; retraining on
other hardware gives different weights. On CPU, the trainer uses gpu = -1.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import yaml

from . import data as D
from . import easytpp as E

COMMON = dict(hidden_size=64, time_emb_size=16, num_layers=2, num_heads=2, dropout_rate=0.0, use_ln=False,
              loss_integral_num_sample_per_step=10, use_mc_samples=True)
SPECS = {
    "RMTPP": {}, "NHP": {}, "SAHP": {}, "AttNHP": {}, "THP": {},
    "IntensityFree": dict(model_specs=dict(num_mix_components=3)),
    "FullyNN": dict(model_specs=dict(num_mlp_layers=3, proper_marked_intensities=False)),
    "ODETPP": dict(model_specs=dict(ode_num_sample_per_step=2)),
    "S2P2": dict(num_layers=4, model_specs=dict(P=16, dropout_rate=0.1, act_func="gelu", for_loop=True, pre_norm=False,
                                                post_norm=True, int_forward_variant=False, int_backward_variant=True,
                                                relative_time=True)),
}
TRAINER = dict(batch_size=32, max_epoch=100, shuffle=False, optimizer="adam", learning_rate=1e-3, valid_freq=1,
               use_tfb=False, metrics=["rmse", "acc"], gpu=0, lr_scheduler=True, weight_decay=0.0, warmup_pct=0.01,
               lr_decay_style="cosine")
EPOCHS = 100
EXPERIMENTS = {
    "time_only": dict(datasets=("email_eu", "retweet_sub", "collegemsg"), mark_loss=False),
    "joint": dict(datasets=("retweet_sub",), mark_loss=True),
    "further": dict(datasets=("volcano", "bpi_sepsis", "bpi_bpic12"), mark_loss=False),
}
MICROBATCH = {("collegemsg", "AttNHP"): 4}


def experiment_of(name: str) -> dict:
    if name not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {name}; one of {tuple(EXPERIMENTS)}")
    return EXPERIMENTS[name]


def data_file(data_dir, ds: str) -> Path:
    return Path(data_dir) / f"{E.EASYTPP_ID[ds]}.pkl"


def make_config(ds: str, model: str, seed: int, epochs: int, rdir: Path, pkl: Path, num_types: int, gpu: int = 0) -> dict:
    """The EasyTPP runner configuration of one run (the layout the paper's runs used)."""
    eid = E.EASYTPP_ID[ds]
    mc = dict(COMMON); mc.update(SPECS[model])
    return {"pipeline_config_id": "runner_config",
            "data": {eid: dict(data_format="pkl", train_dir=str(pkl), valid_dir=str(pkl), test_dir=str(pkl),
                               data_specs=dict(num_event_types=num_types, pad_token_id=num_types, padding_side="right"))},
            f"{model}_{eid}_train": dict(
                base_config=dict(stage="train", backend="torch", dataset_id=eid, runner_id="std_tpp", model_id=model,
                                 base_dir=str(rdir)),
                trainer_config=dict(TRAINER, gpu=gpu, seed=seed, max_epoch=epochs),
                model_config=mc)}


def strip_paths(cfg: dict) -> dict:
    """The configuration without its file paths (data files and run directory), for comparisons."""
    out = json.loads(json.dumps(cfg))
    for v in out.get("data", {}).values():
        for k in ("train_dir", "valid_dir", "test_dir"):
            v.pop(k, None)
    for k, v in out.items():
        if isinstance(v, dict) and "base_config" in v:
            v["base_config"].pop("base_dir", None)
    return out


def write_config(cfg: dict, rdir: Path) -> Path:
    p = rdir / "config_in.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def truncate(pkl: Path, dst: Path, n: int) -> Path:
    """Smoke tests only: the first n sequences of each split."""
    d = E.read_file(pkl)
    out = {k: (v[:n] if isinstance(v, list) else v) for k, v in d.items()}
    dst.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(dst, "wb") as f:
        pickle.dump(out, f)
    return dst


def set_mark_loss(on: bool, env=None):
    env = os.environ if env is None else env
    if on:
        env["EASYTPP_MARK_LOSS"] = "1"
    else:
        env.pop("EASYTPP_MARK_LOSS", None)


def train_one(experiment: str, ds: str, model: str, seed: int, data_dir, out, epochs: int = EPOCHS, max_seqs: int = 0,
              microbatch: int | None = None, threads: int = 1, force: bool = False) -> dict:
    ex = experiment_of(experiment)
    if ds not in ex["datasets"]:
        raise ValueError(f"{experiment}: data set must be one of {ex['datasets']}")
    if model not in SPECS:
        raise ValueError(f"unknown model {model}")
    rdir = Path(out) / ds / model / f"s{seed}"
    if (rdir / "result.json").exists() and not force:
        return json.loads((rdir / "result.json").read_text())
    if force and rdir.exists():
        shutil.rmtree(rdir)
    rdir.mkdir(parents=True, exist_ok=True)
    pkl = data_file(data_dir, ds)
    if not pkl.exists():
        raise FileNotFoundError(f"{pkl}: run `python -m coarse_tpp.easytpp_train prepare` first")
    if max_seqs:
        pkl = truncate(pkl, Path(out) / "smoke_data" / pkl.name, max_seqs)
    mb = microbatch if microbatch is not None else MICROBATCH.get((ds, model), 1)
    set_mark_loss(ex["mark_loss"])
    os.environ["EASYTPP_WORKERS"] = "0"                 # loader in the main process, as in the paper's runs
    if mb > 1:
        os.environ["EASYTPP_MICROBATCH"] = str(mb)
    else:
        os.environ.pop("EASYTPP_MICROBATCH", None)
    E.activate("train")
    import torch
    torch.set_num_threads(int(threads))
    from easy_tpp.config_factory import Config
    from easy_tpp.model.torch_model.torch_basemodel import mark_loss_on
    from easy_tpp.runner import Runner
    assert mark_loss_on() == ex["mark_loss"]
    cuda = torch.cuda.is_available()
    cfg = make_config(ds, model, seed, epochs, rdir, pkl.resolve(), int(E.read_file(pkl)["dim_process"]), 0 if cuda else -1)
    cfg_path = write_config(cfg, rdir)
    rec = dict(experiment=experiment, dataset=ds, model=model, seed=seed, epochs=epochs, mark_loss=mark_loss_on(),
               data=str(pkl.resolve()), data_sha256=E.sha256(pkl), max_seqs=max_seqs or None, microbatch=mb,
               device=torch.cuda.get_device_name(0) if cuda else "cpu", started=time.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()
    try:
        config = Config.build_from_yaml_file(str(cfg_path), experiment_id=f"{model}_{E.EASYTPP_ID[ds]}_train")
        res = Runner.build_from_config(config).run()
        rec["status"] = "ok"
        rec["result"] = res
        n_ep = len(res.get("losses", {}).get("train_loss", [])) if isinstance(res, dict) else None
        if n_ep is not None and n_ep < epochs:
            rec["status"] = "nan_stop"
        rec["epochs_run"] = n_ep
    except Exception as e:  # recorded; the best checkpoint saved so far is kept
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
    finally:
        os.environ.pop("EASYTPP_MICROBATCH", None)
    rec["elapsed_min"] = round((time.time() - t0) / 60, 3)
    tdir = E.train_dir(rdir, ds, model)
    ck = tdir / "models" / "saved_model.pt" if tdir else None
    rec["checkpoint"] = str(ck.relative_to(rdir)) if ck else None
    rec["checkpoint_sha256"] = E.sha256(ck) if ck else None
    (rdir / "result.json").write_text(json.dumps(rec, indent=1, default=str))
    print(f"[{experiment} {ds} {model} s{seed}] {rec['status']} {rec['elapsed_min']} min ckpt={rec['checkpoint']}",
          flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="write the EasyTPP pickles from the data folders")
    p.add_argument("--out", required=True)
    p.add_argument("--datasets", nargs="*", default=list(E.EASYTPP_ID))
    p.add_argument("--data-root", default=None, help="data folders (default: COARSE_TPP_DATA or ./data)")
    for name in ("train", "config"):
        q = sub.add_parser(name)
        q.add_argument("--experiment", required=True, choices=tuple(EXPERIMENTS))
        q.add_argument("--data", required=True, help="directory of the prepared EasyTPP pickles")
        q.add_argument("--out", required=True, help="run root")
        q.add_argument("--epochs", type=int, default=EPOCHS)
        if name == "train":
            q.add_argument("--datasets", nargs="*", default=None)
            q.add_argument("--models", nargs="*", default=list(E.MODELS), choices=E.MODELS)
            q.add_argument("--seeds", nargs="*", type=int, default=list(E.SEEDS))
            q.add_argument("--max-seqs", type=int, default=0, help="smoke test: first n sequences per split")
            q.add_argument("--microbatch", type=int, default=None)
            q.add_argument("--threads", type=int, default=1)
            q.add_argument("--force", action="store_true")
        else:
            q.add_argument("--dataset", required=True)
            q.add_argument("--model", required=True, choices=E.MODELS)
            q.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    if getattr(a, "data_root", None):
        D.set_data_dir(a.data_root)
    if a.cmd == "prepare":
        for ds in a.datasets:
            print(f"{ds}: {E.prepare_file(ds, a.out)}", flush=True)
    elif a.cmd == "config":
        rdir = Path(a.out) / a.dataset / a.model / f"s{a.seed}"
        rdir.mkdir(parents=True, exist_ok=True)
        pkl = data_file(a.data, a.dataset)
        import torch
        cfg = make_config(a.dataset, a.model, a.seed, a.epochs, rdir, pkl.resolve(), int(E.read_file(pkl)["dim_process"]),
                          0 if torch.cuda.is_available() else -1)
        print(write_config(cfg, rdir))
    else:
        for ds in a.datasets or EXPERIMENTS[a.experiment]["datasets"]:
            for m in a.models:
                for s in a.seeds:
                    train_one(a.experiment, ds, m, s, a.data, a.out, a.epochs, a.max_seqs, a.microbatch, a.threads,
                              a.force)


if __name__ == "__main__":
    sys.exit(main())
