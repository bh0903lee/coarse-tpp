"""Re-scoring of trained intensity-based models (RMTPP, NHP, SAHP, AttNHP of EasyTPP) under floors and the interval
score with a single Gauss-Legendre rule per gap (Section 5.2). The trained-models archive provides the original
graded rules. The retrained models of Supplementary Tables S3-S5 (nine architectures, stratified S2P2 compensator,
joint scores) are scored by coarse_tpp.rescore_retrained. No training.

    python -m coarse_tpp.rescore_intensity score --ckpt CKPT_DIR --out OUT_DIR [--datasets ...] [--nodes 32]
    python -m coarse_tpp.rescore_intensity report --ckpt CKPT_DIR --out OUT_DIR

CKPT_DIR/<data set>/<MODEL>/<run>/ holds an EasyTPP run: saved_model.pt, config.yaml (its model_config block) and
optionally the training log `log`. The sequences are read from DATA_DIR/<data set> (gaps as stored, zero gaps stored
as 1e-6, marks 0-based), as the models saw them. The models were trained with a time-only objective (the mark term
removed from the EasyTPP log-likelihood): the per-event log-likelihood is log(lambda(tau) + e) - int_0^tau
(lambda(u) + e) du with tau the stored gap and e = float32 eps, the integral by Monte Carlo. Here the same quantities are evaluated deterministically with
Gauss-Legendre quadrature (`nodes` per integration range), and with the audited delta of the data set the interval
score log P(tau~ <= s < tau~ + delta) = -Lambda(0, tau~) + log(1 - exp(-Lambda(tau~, tau~ + delta))) is added.

Scores per split (mean over target events, every event but the first of a sequence): floor_1e-10 / floor_1e-6 /
floor_1e-5 (density at max(tau~, eps), a stored gap <= 1.000001e-6 counting as zero) and interval, each with tie /
non-tie means. Per (data set, model) the run with the best dev floor_1e-6 score is used, excluding runs whose
training log reports event counts different from the data files (more than one event apart).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from . import data as D
from .analysis import audit_rows
from .data import TIE_STORED

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("RMTPP", "NHP", "SAHP", "AttNHP")
FLOORS = {"floor_1e-10": 1e-10, "floor_1e-6": 1e-6, "floor_1e-5": 1e-5}
MAX_TOKENS = 4096                # batch_size * padded length
G_CHUNK = {"RMTPP": 256, "NHP": 128, "SAHP": 128, "AttNHP": 16}


def _easy_tpp():
    from .easytpp import activate
    activate("models")


# ------------------------------------------------------------------ data
def load_split(ds: str, split: str):
    """[(t, dt, k)] float64 / int64 arrays per sequence (length >= 2), gaps as stored, marks 0-based."""
    seqs, n_types = D._read_split(ds, split, 2)
    marks = D._check_marks([np.asarray([e["type_event"] for e in s], np.int64) for s in seqs], n_types, ds, split)
    out = [(np.asarray([e["time_since_start"] for e in s], np.float64),
            np.asarray([e["time_since_last_event"] for e in s], np.float64), k) for s, k in zip(seqs, marks)]
    return out, n_types


def make_batch(seqs, pad_id, device):
    """EasyTPP tokenizer conventions: right padding, times and gaps padded with pad_id, marks with pad_id;
    attention mask True = masked (future or padding key)."""
    B, N = len(seqs), max(len(s[0]) for s in seqs)
    ts = np.full((B, N), float(pad_id), np.float32); dts = ts.copy(); ks = np.full((B, N), pad_id, np.int64)
    for i, (t, dt, k) in enumerate(seqs):
        n = len(t); ts[i, :n] = t; dts[i, :n] = dt; ks[i, :n] = k
    non_pad = ks != pad_id
    attn = np.triu(np.ones((N, N), bool), k=1)[None] | (~non_pad)[:, None, :]
    f = lambda a: torch.as_tensor(a, device=device)
    return f(ts), f(dts), f(ks), f(non_pad), f(attn)


def batches(seqs):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    i = 0
    while i < len(order):
        n = len(seqs[order[i]][0]); b = max(1, MAX_TOKENS // n)
        j = min(len(order), i + b)
        while j > i + 1 and len(seqs[order[j - 1]][0]) * (j - i) > MAX_TOKENS:
            j -= 1
        yield [seqs[k] for k in order[i:j]]
        i = j


# ------------------------------------------------------------------ models
def build_model(model_id: str, cfg: dict, device):
    """Rebuild an EasyTPP model from the model_config block of its run config (defaults as EasyTPP ModelConfig)."""
    _easy_tpp()
    from types import SimpleNamespace
    from easy_tpp.model.torch_model import RMTPP, NHP, SAHP, AttNHP
    m = cfg["model_config"]
    ns = SimpleNamespace(hidden_size=m.get("hidden_size", 32), time_emb_size=m.get("time_emb_size", 16),
                         num_layers=m.get("num_layers", 2), num_heads=m.get("num_heads", 2),
                         use_ln=m.get("use_ln", False), dropout_rate=m.get("dropout_rate", 0.0),
                         loss_integral_num_sample_per_step=m.get("loss_integral_num_sample_per_step", 20),
                         use_mc_samples=m.get("use_mc_samples", True), thinning=None,
                         num_event_types=m["num_event_types"], num_event_types_pad=m["num_event_types_pad"],
                         pad_token_id=m["event_pad_index"], gpu=0 if device.type == "cuda" else -1,
                         sharing_param_layer=m.get("sharing_param_layer", False), rnn_type=m.get("rnn_type", "LSTM"),
                         model_id=model_id, model_specs=m.get("model_specs", {}))
    cls = dict(RMTPP=RMTPP, NHP=NHP, SAHP=SAHP, AttNHP=AttNHP)[model_id]
    return cls(ns).to(device).eval()


def total_intensity(model, model_id: str, batch, dtimes):
    """Total intensity (summed over marks, plus EasyTPP's float32 eps as in compute_loglikelihood) at `dtimes`
    [B, N-1, G] measured from the events 0..N-2, exactly as each model's loglike_loss evaluates its Monte Carlo
    samples. Returns [B, N-1, G] float64."""
    ts, dts, ks, non_pad, attn = batch
    if model_id == "RMTPP":
        _, rh = model.forward((ts, dts, ks, None, None))
        lam = model.evolve_and_get_intentsity(rh[:, :-1, :], dtimes)
    elif model_id == "NHP":
        _, rh = model.forward((ts, dts, ks, None, None))
        lam = model.layer_intensity(model.get_states(rh[:, :-1, :], dtimes))
    elif model_id == "SAHP":
        enc = model.forward(ts[:, :-1], dts[:, :-1], ks[:, :-1], attn[:, :-1, :-1])
        lam = model.softplus(model.state_decay(enc[:, :, None, :], dtimes[..., None]))
    elif model_id == "AttNHP":
        lam = model.compute_intensities_at_sample_times(ts[:, :-1], dts[:, :-1], ks[:, :-1],
                                                        dtimes + ts[:, :-1, None], attention_mask=attn[:, :-1, :-1])
    else:
        raise ValueError(model_id)
    return (lam + model.eps).sum(-1).double()


def intensity_chunked(model, model_id, batch, dtimes):
    g = G_CHUNK[model_id]
    return torch.cat([total_intensity(model, model_id, batch, dtimes[..., i:i + g].float())
                      for i in range(0, dtimes.shape[-1], g)], dim=-1)


# ------------------------------------------------------------------ scores
def gl(nodes: int):
    x, w = np.polynomial.legendre.leggauss(nodes)
    return torch.as_tensor((x + 1) / 2, dtype=torch.float64), torch.as_tensor(w / 2, dtype=torch.float64)


def event_scores(model, model_id, batch, delta, nodes, device):
    """Per target event (non-pad positions 1..N-1): float64 tensors of every score, tie mask, target mask."""
    ts, dts, ks, non_pad, attn = batch
    tgt = dts[:, 1:].double()
    tau = torch.where(tgt <= TIE_STORED, torch.zeros_like(tgt), tgt)
    xs = {k: torch.clamp(tau, min=e) for k, e in FLOORS.items()}               # density points
    ranges = [(torch.zeros_like(tau), xs[k]) for k in FLOORS] + [(torch.zeros_like(tau), tau)]
    if delta:
        ranges.append((tau, tau + delta))
    u, w = gl(nodes); u, w = u.to(device), w.to(device)
    pts = [a[..., None] + (b - a)[..., None] * u for a, b in ranges]            # [B, N-1, nodes] each
    pts.append(torch.stack([xs[k] for k in FLOORS], -1))                         # density points [B, N-1, 3]
    lam = intensity_chunked(model, model_id, batch, torch.cat(pts, -1))
    Lam = [((lam[..., r * nodes:(r + 1) * nodes] * w).sum(-1)) * (b - a) for r, (a, b) in enumerate(ranges)]
    lam_pt = lam[..., len(ranges) * nodes:]
    out = {k: torch.log(lam_pt[..., j]) - Lam[j] for j, k in enumerate(FLOORS)}
    if delta:
        y = Lam[4].clamp(min=1e-300)
        out["interval"] = -Lam[3] + torch.log(-torch.expm1(-y))
    out["tie"] = tau <= 0
    out["mask"] = non_pad[:, 1:]
    return out


def score_split(model, model_id, seqs, pad_id, delta, nodes, device):
    acc = {}
    n = n_tie = 0
    with torch.no_grad():
        for bs in batches(seqs):
            b = make_batch(bs, pad_id, device)
            o = event_scores(model, model_id, b, delta, nodes, device)
            m = o["mask"]; tie = o["tie"] & m
            n += int(m.sum()); n_tie += int(tie.sum())
            for k in [k for k in o if k not in ("tie", "mask")]:
                v = o[k]
                acc.setdefault(k, [0.0, 0.0, 0.0, 0])
                acc[k][0] += float(v[m].sum()); acc[k][1] += float(v[tie].sum()); acc[k][2] += float(v[m & ~o["tie"]].sum())
                acc[k][3] += int((~torch.isfinite(v[m])).sum())
    out = {"n": n, "n_tie": n_tie}
    for k, (s, st, sn, nbad) in acc.items():
        out[k] = s / n if n else None
        out[f"{k}_tie"] = st / n_tie if n_tie else None
        out[f"{k}_nontie"] = sn / (n - n_tie) if n > n_tie else None
        out[f"{k}_nonfinite"] = nbad
    return out


def candidates(ckpt: Path, ds, model_id):
    return sorted(p for p in (ckpt / ds / model_id).glob("*") if (p / "saved_model.pt").exists())


def checkpoint_identity(candidate):
    """Fingerprint the mutable EasyTPP run's configuration and saved weights."""
    paths = {"config_sha256": candidate / "config.yaml", "ckpt_sha256": candidate / "saved_model.pt"}
    if not all(p.is_file() for p in paths.values()):
        return None
    return {key: hashlib.sha256(p.read_bytes()).hexdigest() for key, p in paths.items()}


def score(ckpt, out, datasets=None, nodes=32, audit_path=None):
    ckpt, out = Path(ckpt), Path(out)
    audit = audit_rows(audit_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (out / "cells").mkdir(parents=True, exist_ok=True)
    for ds in datasets or sorted(p.name for p in ckpt.iterdir() if p.is_dir()):
        delta = (audit.get(ds) or {}).get("delta")
        splits = {sp: load_split(ds, sp) for sp in ("dev", "test")}
        for model_id in MODELS:
            for cand in candidates(ckpt, ds, model_id):
                dst = out / "cells" / f"{ds}__{model_id}__{cand.name}.json"
                identity = checkpoint_identity(cand)
                if dst.exists():
                    r = json.loads(dst.read_text())
                    if (identity is not None and r.get("identity") == identity
                            and r.get("nodes") == nodes and r.get("delta") == delta):
                        continue
                cfg = yaml.safe_load((cand / "config.yaml").read_text())
                model = build_model(model_id, cfg, device)
                model.load_state_dict(torch.load(cand / "saved_model.pt", map_location=device, weights_only=True))
                r = dict(dataset=ds, model=model_id, candidate=cand.name, delta=delta, nodes=nodes, identity=identity,
                         n_params=sum(p.numel() for p in model.parameters()))
                for sp, (seqs, _) in splits.items():
                    r[sp] = score_split(model, model_id, seqs, cfg["model_config"]["event_pad_index"], delta, nodes, device)
                tmp = dst.with_suffix(f".tmp{os.getpid()}")
                tmp.write_text(json.dumps(r, indent=1)); os.replace(tmp, dst)
                print(f"{ds} {model_id} {cand.name}: dev floor_1e-6 {r['dev']['floor_1e-6']:.4f} "
                      f"test floor_1e-6 {r['test']['floor_1e-6']:.4f} interval {r['test'].get('interval')}", flush=True)


# ------------------------------------------------------------------ report
def logged_counts(path: Path):
    """Modal train / test event counts in an EasyTPP training log (None when the log has no epoch line)."""
    p = path / "log"
    if not p.exists():
        return None
    log = p.read_text(errors="ignore")
    out = {}
    for sp in ("train", "test"):
        c = Counter(int(x) for x in re.findall(r"\(%s\) \]: %s loglike is -?[\d.e+]+, num_events is (\d+)" % (sp, sp), log))
        out[sp] = c.most_common(1)[0][0] if c else None
    return out


def current_counts(ds):
    return {sp: sum(len(s[0]) - 1 for s in load_split(ds, sp)[0]) for sp in ("train", "test")}


def select(cells, ckpt: Path):
    """{(ds, model): chosen cell} by the dev floor_1e-6 score among runs with finite scores whose training log (if
    any) matches the event counts of the data files, with verified current configuration and weights. Cached scores
    without an identity are re-scored."""
    out, cur = {}, {}
    for c in cells:
        cand = ckpt / c["dataset"] / c["model"] / c["candidate"]
        identity = checkpoint_identity(cand)
        c["stale"] = identity is None or c.get("identity") != identity
        if c["stale"]:
            continue
        if any(c[s].get("floor_1e-6") is None or not math.isfinite(c[s]["floor_1e-6"]) for s in ("dev", "test")):
            continue
        lc = logged_counts(cand)
        if lc:
            cur.setdefault(c["dataset"], current_counts(c["dataset"]))
            if not all(lc[sp] is None or abs(lc[sp] - cur[c["dataset"]][sp]) <= 1 for sp in ("train", "test")):
                c["excluded"] = True
                continue
        k = (c["dataset"], c["model"])
        if k not in out or c["dev"]["floor_1e-6"] > out[k]["dev"]["floor_1e-6"]:
            out[k] = c
    return out


def fmtv(x, d=4):
    return "—" if x is None or not math.isfinite(x) else f"{x:+.{d}f}" if abs(x) < 1e4 else f"{x:+.3e}"


def rank(vals):
    ok = [(m, v) for m, v in vals.items() if v is not None and math.isfinite(v)]
    return [m for m, _ in sorted(ok, key=lambda kv: -kv[1])]


def report(ckpt, out):
    out = Path(out)
    cells = [json.loads(p.read_text()) for p in sorted((out / "cells").glob("*.json"))]
    sel = select(cells, Path(ckpt))
    L = ["# Re-scoring of trained intensity-based models", "",
         f"runs scored: {len(cells)}; stale or unidentified: {sum(bool(c.get('stale')) for c in cells)}; "
         "excluded (training log does not match the data files): "
         f"{sum(bool(c.get('excluded')) for c in cells)}; chosen per (data set, model) by dev floor_1e-6. Mean per target event.", ""]
    if any(c.get("stale") for c in cells):
        L += ["Re-run score to refresh stale or unidentified package caches. "
              "Deposited archive tables use the archive's own reporting script.", ""]
    same = 0
    for ds in sorted({c["dataset"] for c in cells}):
        rows = {m: sel.get((ds, m)) for m in MODELS}
        if not any(rows.values()):
            continue
        any_row = next(c for c in rows.values() if c)
        tie = any_row["test"]["n_tie"] / any_row["test"]["n"]
        L += [f"## {ds} (test ties {tie:.4f}, delta {any_row['delta']})", "",
              "| model | floor 1e-10 | floor 1e-6 | floor 1e-5 | interval | ties: floor 1e-6 / interval |", "|---|---:|---:|---:|---:|---|"]
        for m in MODELS:
            c = rows[m]
            if c is None:
                L.append(f"| {m} | — | — | — | — | — |"); continue
            t = c["test"]
            L.append(f"| {m} | {fmtv(t['floor_1e-10'])} | {fmtv(t['floor_1e-6'])} | {fmtv(t['floor_1e-5'])} | {fmtv(t.get('interval'))} | "
                     f"{fmtv(t.get('floor_1e-6_tie'))} / {fmtv(t.get('interval_tie'))} |")
        r6 = rank({m: rows[m]["test"]["floor_1e-6"] if rows[m] else None for m in MODELS})
        rg = rank({m: rows[m]["test"].get("interval") if rows[m] else None for m in MODELS}) if any_row["delta"] else None
        if rg is not None:
            same += r6 == rg
        L += ["", f"order by floor 1e-6: {' > '.join(r6)}" + (f"; by interval: {' > '.join(rg)}" if rg else ""), ""]
    L += [f"Same order under both scores on {same} data sets with a resolution."]
    (out / "report.md").write_text("\n".join(L) + "\n")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("score", "report"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--nodes", type=int, default=32)
    ap.add_argument("--audit", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "score":
        score(a.ckpt, a.out, a.datasets, a.nodes, a.audit)
    else:
        print("\n".join(report(a.ckpt, a.out)))


if __name__ == "__main__":
    main()
