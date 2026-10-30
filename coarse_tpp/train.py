"""Train and evaluate one cell. Usage: python -m coarse_tpp.train CELL_JSON

Outputs in cfg["out_dir"]: result.json (written atomically after the test evaluation), best_<selection>.pt
(state dict of each selected checkpoint), perseq_<selection>.npz (per-sequence test sums) and train.log.
A cell that already has a valid result.json is not trained again.

Config keys (coarse_tpp.experiments writes them):
  dataset, seed, encoder {type, marks, ...}, head (mix | rmtpp | thp | weib | flow), K, mix_param (mixture parameterization,
  default standard), sigma_min / mu_min (lower bounds of the standard mixture head, coarse_tpp.models), treatment,
  eps (floor), iw_samples (iw_jitter, iw_jitter_triangular), mix_lambda (interval_jitter_mix),
  delta (recording resolution of the data; 0 = exact), delta_model (resolution assumed by the treatment; = delta
  unless misspecified), optim {lr, batch_size, max_epochs, patience, grad_clip}, mark_head (add a categorical mark
  head; the objective, the selection and the test metrics include the mark term; encoder.marks must be true),
  mark_only (the loss is the mark term alone and the selection is the dev mark log-likelihood; needs mark_head),
  primary_selection (optional override of the early-stopping criterion, e.g. "own"), raw_gaps (feed the stored
  gaps, with ties stored as 1e-6, instead of reading stored ties as exact zeros).

Treatments (training objective per target event):
  floor                log q(max(tau~, eps))
  jitter_gap           log q(tau~ + delta u), u ~ U(0, 1) drawn per step (targets only)
  iw_jitter            log (1/S) sum_s q(tau~ + delta u_s) (S = iw_samples)
  iw_jitter_triangular log m + log (1/S) sum_s q(w_s), w_s drawn from the triangular kernel of the recorded cell
                       and m its mass (coarse_tpp.data); S -> inf gives the triangular likelihood
  jitter_time          recorded times + U(0, delta), re-sorted (inputs and targets); log q of the new gaps
  jitter_time_gauss    as jitter_time with N(0, delta^2 / 12) noise
  point_mass           tie: log pi; positive gap: log(1 - pi) + log q(tau~)
  interval             log P(tau~ <= tau < tau~ + delta)
  triangular           triangular-kernel likelihood (coarse_tpp.lik)
  interval_jitter_mix  (1 - lambda) interval + lambda (jitter_gap + log delta)
  latent               trained on the latent times (synthetic data, delta = 0)
Standardization statistics: coarse_tpp.data.coarse_stats (shared by every treatment). Readouts and scores:
coarse_tpp.readout.
Selection: early stopping and the primary checkpoint use the dev score C under the treatment's readout. Also kept
and evaluated: own (best dev value of the treatment's own objective; jitter: one fixed draw) and B (best dev
latent-time score; synthetic only).
"""
from __future__ import annotations

import json
import math
import os
import random
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import lik
from . import readout as Ro
from . import results as R
from .data import (coarse_stats, jitter_targets_triangular_multi, jitter_targets_uniform, jitter_targets_uniform_multi,
                   jitter_times, load, per_target, score_mask)
from .models import build

EPS_JIT = 1e-10          # density floor for jittered / exact gaps (positive almost surely)
TIW_ZERO = 1e-300        # iw_jitter_triangular: an exactly-zero kernel draw (u = v) is evaluated here; positive draws are not floored
SELECTIONS = ("C", "own", "B")
FLOORS = {"floor_1e-10": 1e-10, "floor_1e-6": 1e-6, "floor_1e-5": 1e-5}


def log(fh, msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    fh.write(line + "\n"); fh.flush()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def selections(cfg):
    """Checkpoint selections kept: C, own, B, plus J (= C + mark) with a mark head, plus M (dev mark) for mark_only."""
    return SELECTIONS + (("J",) if cfg.get("mark_head") else ()) + (("M",) if cfg.get("mark_only") else ())


def primary(cfg):
    """Selection used for early stopping and the primary test metrics: dev C; dev J with a mark head; dev mark (M)
    for a mark_only cell; or cfg["primary_selection"] when given."""
    if cfg.get("primary_selection"):
        return cfg["primary_selection"]
    if cfg.get("mark_only"):
        if not cfg.get("mark_head"):
            raise ValueError("mark_only needs mark_head")
        return "M"
    return "J" if cfg.get("mark_head") else "C"


def objective(time_ll, lp, b, cfg):
    """Per-target training objective: time term + mark term; a mark_only cell drops the time term."""
    if lp is None:
        return time_ll
    mk = mark_ll(lp, b)
    return mk if cfg.get("mark_only") else time_ll + mk


def fwd(model, b, stats, cfg):
    """(law, tie_logit, mark_logp or None)."""
    if cfg.get("mark_head"):
        return model(b, stats, marks=True)
    law, tl = model(b, stats)
    return law, tl, None


def mark_ll(lp, b):
    """Per-target log-probability of the observed next mark, (B, L-1) float64 (0 on padding positions)."""
    return lp.gather(-1, b.marks[:, 1:].unsqueeze(-1)).squeeze(-1)


def provenance(ds: str, stats, device) -> dict:
    """Data and code fingerprints of a result."""
    import hashlib
    from dataclasses import asdict
    def sha(paths):
        h = hashlib.sha256()
        for p in sorted(paths):
            h.update(p.name.encode()); h.update(p.read_bytes())
        return h.hexdigest()[:16]
    if ds.startswith("synth_"):
        from .synthetic import parse, synth_dir
        proc, r, _ = parse(ds)
        data = [synth_dir() / f"{proc}_r{r}.npz"]
    else:
        data = sorted((D.data_dir() / ds).glob("*.pkl"))
    return dict(data_files=[p.name for p in data], data_sha=sha(data),
                code_sha=sha(Path(__file__).resolve().parent.glob("*.py")),
                stats=asdict(stats), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu")


def n_scored(views) -> int:
    """Scored test targets counted from the data alone."""
    if views.score is not None:
        return int(sum(float(np.sum(x[1:])) for x in views.score))
    return int(sum(len(x) - 1 for x in views.coarse.iet))


def rule(cfg):
    return Ro.RULE[cfg["treatment"]]


def train_ll(law, tl, tau, cfg, gen=None):
    """Per-event training log-likelihood of the treatment (tau = targets of the possibly jittered batch)."""
    t, d = cfg["treatment"], float(cfg["delta_model"])
    if t == "floor":
        return lik.event_loglik(law, tau, "floor", eps=float(cfg["eps"]))
    if t in ("latent", "jitter_time", "jitter_time_gauss") or d == 0:
        if t == "point_mass":
            return lik.event_loglik(law, tau, "point_mass", tie_logit=tl)
        return lik.event_loglik(law, tau, "floor", eps=EPS_JIT)
    if t == "jitter_gap":
        return lik.event_loglik(law, jitter_targets_uniform(tau, d, gen), "floor", eps=EPS_JIT)
    if t == "iw_jitter":          # log (1/S) sum_s f(tau~ + delta u_s); S = 1 is jitter_gap, S -> inf is interval - log delta
        S = int(cfg["iw_samples"])
        x = jitter_targets_uniform_multi(tau, d, S, gen)                       # (S, B, L); the law broadcasts over S
        return torch.logsumexp(lik.event_loglik(law, x, "floor", eps=EPS_JIT), dim=0) - math.log(S)
    if t == "iw_jitter_triangular":   # log mass + log (1/S) sum_s q(w_s), w_s from the triangular kernel; S -> inf is triangular
        S = int(cfg["iw_samples"])
        w, log_mass = jitter_targets_triangular_multi(tau, d, S, gen)
        # the density is evaluated at the drawn point itself (a floor max(w, eps) on positive draws would change the
        # limit); only an exact w = 0 (u = v in float64, probability ~0) is moved to TIW_ZERO
        return log_mass + torch.logsumexp(law.log_density(w.clamp(min=TIW_ZERO)), dim=0) - math.log(S)
    if t == "point_mass":
        return lik.event_loglik(law, tau, "point_mass", tie_logit=tl)
    if t == "interval_jitter_mix":   # interval likelihood minus lambda x (its Jensen gap to the jitter bound)
        lam = float(cfg["mix_lambda"])
        gap = lik.event_loglik(law, tau, "interval", delta=d)
        jit = lik.event_loglik(law, jitter_targets_uniform(tau, d, gen), "floor", eps=EPS_JIT) + math.log(d)
        return (1.0 - lam) * gap + lam * jit
    if t in ("interval", "triangular"):
        return lik.event_loglik(law, tau, t, delta=d)
    raise ValueError(t)


def prepare(b, cfg, gen):
    t = cfg["treatment"]
    if t in ("jitter_time", "jitter_time_gauss") and float(cfg["delta_model"]) > 0:
        return jitter_times(b, float(cfg["delta_model"]), t, gen)
    return b


@torch.no_grad()
def dev_criteria(model, views, stats, cfg, device, bs=64):
    """Mean per-event dev values of every selection criterion over scored targets (higher is better)."""
    model.eval()
    delta = float(cfg["delta"])
    gen = torch.Generator().manual_seed(12345)
    acc = {k: [0.0, 0.0] for k in selections(cfg)}
    sp = views.coarse
    for i0 in range(0, len(sp), bs):
        idx = np.arange(i0, min(i0 + bs, len(sp)))
        b = D.make_batch(sp, idx, device)
        m = (b.tmask * score_mask(views, idx, b.iet.shape[1], device)).double()
        bo = prepare(b, cfg, gen)
        law, tl, lp = fwd(model, bo, stats, cfg)
        own = objective(train_ll(law, tl, bo.iet[:, 1:], cfg, gen), lp, bo, cfg)
        if bo is not b:
            law, tl, lp = fwd(model, b, stats, cfg)
        c = Ro.C(law, tl, b.iet[:, 1:], delta, rule(cfg))
        acc["own"][0] += float((own * m).sum()); acc["C"][0] += float((c * m).sum())
        acc["own"][1] += float(m.sum()); acc["C"][1] += float(m.sum())
        if lp is not None:
            mk = mark_ll(lp, b)
            acc["J"][0] += float(((c + mk) * m).sum()); acc["J"][1] += float(m.sum())
            if "M" in acc:
                acc["M"][0] += float((mk * m).sum()); acc["M"][1] += float(m.sum())
        if views.lat_s is not None:
            sv = per_target(views, "lat_s", idx, b.iet.shape[1], device)
            bb = Ro.B(law, tl, torch.where(m > 0, sv, torch.ones_like(sv)), delta, rule(cfg))
            acc["B"][0] += float((bb * m).sum()); acc["B"][1] += float(m.sum())
    return {k: (v[0] / v[1] if v[1] else None) for k, v in acc.items()}


@torch.no_grad()
def evaluate(model, views, stats, cfg, device, bs=64):
    """Metrics of the loaded checkpoint over scored targets, plus per-sequence sums. Tie / non-tie means use the tie
    definition of the score C (readout.tie_mask)."""
    model.eval()
    delta = float(cfg["delta"])
    r = rule(cfg)
    sp = views.coarse
    keys = ["C"] + list(FLOORS)
    has_b = views.lat_s is not None
    if has_b:
        keys += ["oracle_C", "B", "oracle_B"]
    with_marks = bool(cfg.get("mark_head"))
    if with_marks:
        keys += ["mark", "acc", "J"]
    per_seq = {k: np.zeros(len(sp)) for k in keys}
    tie_sum = {k: 0.0 for k in keys}; non_sum = {k: 0.0 for k in keys}
    n_seq, n_tie = np.zeros(len(sp)), np.zeros(len(sp))
    for i0 in range(0, len(sp), bs):
        idx = np.arange(i0, min(i0 + bs, len(sp)))
        b = D.make_batch(sp, idx, device)
        L = b.iet.shape[1]
        m = (b.tmask * score_mask(views, idx, L, device)).double()
        tau = b.iet[:, 1:].double()
        tie = Ro.tie_mask(tau, delta).double() * m
        law, tl, lp = fwd(model, b, stats, cfg)
        v = {"C": Ro.C(law, tl, tau, delta, r)}
        for k, e in FLOORS.items():
            v[k] = lik.event_loglik(law, tau, "floor", eps=e)
        if with_marks:
            v["mark"] = mark_ll(lp, b)
            v["acc"] = (lp.argmax(-1) == b.marks[:, 1:]).double()
            v["J"] = v["C"] + v["mark"]
        if has_b:
            v["oracle_C"] = per_target(views, "oracle_C", idx, L, device)
            sv = per_target(views, "lat_s", idx, L, device)
            sv = torch.where(m > 0, sv, torch.ones_like(sv))
            v["B"] = Ro.B(law, tl, sv, delta, r)
            v["oracle_B"] = per_target(views, "oracle_lp", idx, L, device)
        for k, x in v.items():
            x = torch.where(m > 0, x, torch.zeros_like(x))
            per_seq[k][idx] = (x * m).sum(1).cpu().numpy()
            tie_sum[k] += float((x * tie).sum()); non_sum[k] += float((x * (m - tie)).sum())
        n_seq[idx] = m.sum(1).cpu().numpy(); n_tie[idx] = tie.sum(1).cpu().numpy()
    n, nt = float(n_seq.sum()), float(n_tie.sum())
    out = dict(n_events=int(round(n)), n_ties=int(round(nt)), tie_share=nt / max(n, 1))
    for k in keys:
        out[k] = float(per_seq[k].sum() / n)
        out[f"{k}_tie"] = tie_sum[k] / nt if nt else None
        out[f"{k}_nontie"] = non_sum[k] / (n - nt) if n > nt else None
    out["ll"] = out["C"]
    return out, dict(n=n_seq, **per_seq)


def device_of(cfg):
    return torch.device("cuda" if torch.cuda.is_available() and not cfg.get("cpu") else "cpu")


def run(cfg: dict) -> dict:
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    st, _, prev = R.load(cfg)
    if st in R.FINISHED:
        return prev
    fh = open(out / "train.log", "a")
    log(fh, f"===== attempt pid={os.getpid()} host={socket.gethostname()} "
            f"gpu={os.environ.get('CUDA_VISIBLE_DEVICES', '')} cell={cfg['cell_id']} =====")
    log(fh, "config " + json.dumps(cfg, sort_keys=True))
    torch.set_num_threads(int(cfg.get("threads", 2)))
    device = device_of(cfg)
    seed_all(cfg["seed"])
    t0 = time.time()
    ds = cfg["dataset"]
    raw = bool(cfg.get("raw_gaps"))
    tr, dv = load(ds, "train", raw=raw), load(ds, "dev", raw=raw)
    stats = coarse_stats(tr, float(cfg["delta"]))
    model = build(cfg, stats, tr.coarse.n_types).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(fh, f"data train={len(tr.coarse)} dev={len(dv.coarse)} latent={tr.latent is not None} params={n_params}")
    o = cfg["optim"]
    opt = torch.optim.Adam(model.parameters(), lr=o["lr"])
    rng = np.random.default_rng(cfg["seed"])
    gen = torch.Generator().manual_seed(int(cfg["seed"]) + 1000)
    sels, prim = selections(cfg), primary(cfg)
    best = {k: (-math.inf, 0, None) for k in sels}
    history, status = [], "ok"
    max_wall = float(cfg.get("max_wall_hours", 12)) * 3600
    for ep in range(1, o["max_epochs"] + 1):
        model.train()
        tot, nb = 0.0, 0
        for b in D.iterate(tr.coarse, o["batch_size"], device, shuffle=True, rng=rng):
            b = prepare(b, cfg, gen)
            law, tl, lp = fwd(model, b, stats, cfg)
            ll = objective(train_ll(law, tl, b.iet[:, 1:], cfg, gen), lp, b, cfg)
            loss = -(ll * b.tmask.double()).sum() / b.tmask.sum().clamp(min=1)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {ep}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), o.get("grad_clip", 1.0))
            opt.step()
            tot += loss.item(); nb += 1
        crit = dev_criteria(model, dv, stats, cfg, device)
        marks = []
        for k, v in crit.items():
            if v is not None and v > best[k][0]:
                best[k] = (v, ep, {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()})
                marks.append(k)
        history.append(dict(epoch=ep, train_loss=tot / max(nb, 1), **{f"dev_{k}": v for k, v in crit.items()}))
        log(fh, f"epoch {ep} train_loss={tot / max(nb, 1):.5f} " +
            " ".join(f"dev_{k}={v:.5f}" for k, v in crit.items() if v is not None) + (f" *{','.join(marks)}" if marks else ""))
        if ep - best[prim][1] >= o["patience"]:
            log(fh, f"early stop: dev {prim} has not improved for {o['patience']} epochs")
            break
        if time.time() - t0 > max_wall:
            status = "ok_walltime"; log(fh, "wall-clock limit reached"); break
    if best[prim][2] is None:
        raise RuntimeError("no checkpoint was selected")
    te = load(ds, "test", raw=raw)                           # first contact with the test split
    res_sel, dev_own = {}, None
    for k, (v, ep, state) in best.items():
        if state is None:
            continue
        model.load_state_dict(state)
        torch.save(state, out / f"best_{k}.pt")
        m_te, per = evaluate(model, te, stats, cfg, device)
        res_sel[k] = dict(selected_epoch=ep, dev_criterion=v, test=m_te)
        np.savez(out / f"perseq_{k}.npz", **per)
        if k == prim:
            dev_own, _ = evaluate(model, dv, stats, cfg, device)
    res = dict(cell_id=cfg["cell_id"], status=status, experiment=cfg["experiment"], arm=cfg["arm"], dataset=ds,
               seed=cfg["seed"], selected_epoch=best[prim][1], epochs_run=len(history), n_params=n_params,
               wall_sec=round(time.time() - t0, 1), expected_test_targets=n_scored(te), primary_selection=prim,
               test=res_sel[prim]["test"], dev=dev_own, by_selection=res_sel, history=history, config=cfg,
               torch=torch.__version__, device=str(device), provenance=provenance(ds, stats, device))
    bad = R.nonfinite_metrics(res)
    if te.lat_s is not None:            # synthetic: the latent-time score B must exist
        bad += [f"{sel}.test.B" for sel, x in res_sel.items()
                if x["test"].get("B") is None or not math.isfinite(x["test"]["B"])]
    if bad:
        res["status"], res["nonfinite"] = "invalid_metrics", bad
    tmp = out / f"result.json.tmp{os.getpid()}"
    tmp.write_text(json.dumps(R.sanitize(res), indent=1, allow_nan=False))
    os.replace(tmp, out / "result.json")
    log(fh, f"done status={res['status']} test_C={res['test']['ll']:.5f} sel_epoch={best[prim][1]} wall={res['wall_sec']}s")
    fh.close()
    return res


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    try:
        run(cfg)
    except Exception:
        out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
        with open(out / "train.log", "a") as fh:
            log(fh, "FAILED\n" + traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
