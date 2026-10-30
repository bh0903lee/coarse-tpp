"""Component edits: decomposition of the floor's self-preference margin into the location and the scale of the
mixture component at the tie (Supplementary S4, "Component location and scale", Table S9). No training.

    python -m coarse_tpp.component_edits score RUN_DIR [--shard i/n] [--limit N] [--cpu] [--force] [--retry-quarantined]
    python -m coarse_tpp.component_edits report RUN_DIR

RUN_DIR is a rank_change run. Units: combinations (data set, encoder) whose stored test tie share is at least TIE_MIN.
Treatments: the five scored on the same raw density, floor 1e-10 / 1e-6 / 1e-5, jitter_gap, interval; per (unit,
treatment) the learning rate with the best dev-C mean over seeds (coarse_tpp.analysis.select_lr, the rule of Table 4);
checkpoint best_C.pt. Work item = (unit, seed). For every model the per-event mixture (log_w, mu, log_sigma) of the
scored test targets is extracted with the masks of coarse_tpp.train.evaluate (tie = recorded zero gap).

Floor score of model j at floor eps_m: S_j(m) = mean over scored events of log q_j(max(tau~, eps_m)).
Self-preference margin: M_m = S_self(m) - max_{j != self} S_j(m), self = the model trained with eps_m. Split per
scored event (so the two parts add up exactly): tie part (T_self - T_r) / n, non-tie part (U_self - U_r) / n, where
T / U are sums over tie / non-tie events and r the runner-up. The report forms the margin on seed means (the paper's
ranking uses means over five seeds) and counts the seeds with a positive margin as well.

Edits, applied ONLY to the eps_m-trained model and ONLY at tie events. Per tie event
  k* = argmax_k [log w_k + log N(log eps_m; mu_k, sigma_k) - log eps_m]   (the component carrying the density at the floor)
  scale           sigma_k* <- sigma_nt, ONE value per (unit, seed, eps_m model): the weighted median of sigma_k over all
                  K components of all scored NON-TIE test events of that model, each entry weighted by its mixture
                  weight w_k (so every non-tie event carries total weight 1); lower weighted median
  location        mu_k* <- log(delta), delta = the recording resolution of the data set (cfg["delta"])
  both            scale and location together
  scale_event     (secondary contrast) sigma_k* <- weighted median of the sigmas of the same tie event's other
                  components (weights renormalised over them); fallback sigma_nt when their total weight < 1e-12.
                  The other components of a tie event may be narrow too, so this is not a widening to the non-tie level
  scale_event_location   scale_event and location together
Weights are unchanged; the edited model is scored at all three floors. Non-tie events and the other four models are
unchanged, so the runner-up is unchanged by an edit.

Validity: the model is rebuilt as coarse_tpp.train.run does (coarse_stats of the train split, build(cfg, stats,
n_types)); the stored test floor_1e-6 (result.json, by_selection.C.test) and the event / tie counts must be
reproduced (|diff| <= REPRO_TOL), else the work item is quarantined and left out of the report. A quarantined file is
not a valid cache: `score` recomputes it with --retry-quarantined (for instance on another device) and otherwise
skips it with a notice; --force recomputes every item.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import analysis as A
from . import data as D
from . import lik
from . import readout as Ro
from .data import coarse_stats, load, score_mask
from .models import build

ARMS = ("floor_1e-10", "floor_1e-6", "floor_1e-5", "jitter_gap", "interval")
FLOORS = {"floor_1e-10": 1e-10, "floor_1e-6": 1e-6, "floor_1e-5": 1e-5}
SELF = tuple(FLOORS)
EDITS = ("scale", "location", "both", "scale_event", "scale_event_location")
PRIMARY = ("scale", "location", "both")
TIE_MIN = 0.01
REPRO_TOL = 1e-5
FALLBACK_W = 1e-12
VERSION = 1
QS = (0.1, 0.25, 0.5, 0.75, 0.9)
LOG_2PI = math.log(2 * math.pi)


# ------------------------------------------------------------------ pure functions (tests/test_component_edits.py)
def comp_terms(log_w, mu, log_sigma, x):
    """(N, K): log w_k + log of the k-th log-normal density at x (x (N,) > 0), i.e. the components of log q(x)."""
    lx = torch.log(x.double()).unsqueeze(-1)
    z = (lx - mu) / log_sigma.exp()
    return log_w - 0.5 * LOG_2PI - log_sigma - 0.5 * z * z - lx


def floor_score(log_w, mu, log_sigma, tau, eps):
    """Per-event floor score log q(max(tau, eps)) (coarse_tpp.lik, as coarse_tpp.train.evaluate)."""
    return lik.event_loglik(lik.LogNormMix(log_w, mu, log_sigma), tau, "floor", eps=eps)


def kstar(log_w, mu, log_sigma, eps):
    """Index of the component carrying the density at the floor, its log share of q(eps), and z = (log eps - mu) / sigma."""
    x = torch.full(log_w.shape[:-1], float(eps), dtype=torch.float64)
    c = comp_terms(log_w, mu, log_sigma, x)
    k = c.argmax(-1)
    share = c.gather(-1, k.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(c, -1)
    g = lambda t: t.gather(-1, k.unsqueeze(-1)).squeeze(-1)
    z = (math.log(eps) - g(mu)) / g(log_sigma).exp()
    return k, share, z


def weighted_median_rows(values, weights):
    """Row-wise weighted median (lower median: the smallest value whose cumulative weight reaches half the row total).
    Rows with zero total weight give nan."""
    v, o = values.sort(-1)
    w = weights.gather(-1, o)
    tot = w.sum(-1, keepdim=True)
    cw = w.cumsum(-1)
    idx = (cw < 0.5 * tot).sum(-1, keepdim=True).clamp(max=values.shape[-1] - 1)
    out = v.gather(-1, idx).squeeze(-1)
    return torch.where(tot.squeeze(-1) > 0, out, torch.full_like(out, float("nan")))


def pooled_weighted_median(values, weights):
    """Weighted median of all entries (each row of `weights` is one event's mixture weights)."""
    return float(weighted_median_rows(values.reshape(1, -1), weights.reshape(1, -1))[0])


def other_sigma_median(log_w, log_sigma, k):
    """(sigma_med, total weight of the other components): weight-weighted median sigma of the components other than k."""
    w = log_w.exp().clone()
    w.scatter_(-1, k.unsqueeze(-1), 0.0)
    return weighted_median_rows(log_sigma.exp(), w), w.sum(-1)


def edit(mu, log_sigma, k, sigma_new=None, mu_new=None):
    """Copies of (mu, log_sigma) with component k (per row) set to sigma_new / mu_new (scalar or (N,)); others unchanged."""
    mu, log_sigma = mu.clone(), log_sigma.clone()
    idx = k.unsqueeze(-1)
    if sigma_new is not None:
        s = torch.as_tensor(sigma_new, dtype=torch.float64).expand(k.shape)
        log_sigma.scatter_(-1, idx, torch.log(s).unsqueeze(-1))
    if mu_new is not None:
        m = torch.as_tensor(mu_new, dtype=torch.float64).expand(k.shape)
        mu.scatter_(-1, idx, m.unsqueeze(-1))
    return mu, log_sigma


def margin(sums, self_arm, n):
    """sums {arm: (tie_sum, nontie_sum)} of one floor; n scored events. Margin of self over the best other arm, split
    into tie and non-tie parts per scored event (tie + nontie = total exactly)."""
    S = {a: (t + u) / n for a, (t, u) in sums.items()}
    r = max((a for a in sums if a != self_arm), key=lambda a: S[a])
    tie = (sums[self_arm][0] - sums[r][0]) / n
    non = (sums[self_arm][1] - sums[r][1]) / n
    return dict(total=tie + non, tie=tie, nontie=non, runner_up=r, scores=S)


def quantiles(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return dict(mean=float(x.mean()), **{f"q{int(q * 100)}": float(np.quantile(x, q)) for q in QS})


def width_values(p, k):
    """(sigma_nt, sigma_evt, fallback) for the tie rows of model record p (k = k* per tie row): sigma_nt is the scalar
    of the `scale` edit, sigma_evt the per-event value of `scale_event` (fallback sigma_nt where the other components
    carry no weight)."""
    rows, nt = p["tie"], ~p["tie"]
    sig_nt = pooled_weighted_median(p["ls"][nt].exp(), p["lw"][nt].exp())
    sig_w, w_other = other_sigma_median(p["lw"][rows], p["ls"][rows], k)
    fb = (w_other < FALLBACK_W) | ~torch.isfinite(sig_w)
    return sig_nt, torch.where(fb, torch.full_like(sig_w, sig_nt), sig_w), fb


def edit_variants(mu, ls, k, sig_nt, sig_evt, logd):
    """{edit name: (mu, log_sigma)} of the tie rows."""
    return {"scale": edit(mu, ls, k, sigma_new=sig_nt), "location": edit(mu, ls, k, mu_new=logd),
            "both": edit(mu, ls, k, sigma_new=sig_nt, mu_new=logd), "scale_event": edit(mu, ls, k, sigma_new=sig_evt),
            "scale_event_location": edit(mu, ls, k, sigma_new=sig_evt, mu_new=logd)}


def floor_sums(P, eps, mu=None, ls=None, rows=None):
    """(tie_sum, nontie_sum) of the floor score; optional replacement (mu, ls) for the rows `rows` (tie rows)."""
    v = floor_score(P["lw"], P["mu"], P["ls"], P["tau"], eps)
    if rows is not None:
        v = v.clone()
        v[rows] = floor_score(P["lw"][rows], mu, ls, P["tau"][rows], eps)
    t = P["tie"]
    return float(v[t].sum()), float(v[~t].sum())


# ------------------------------------------------------------------ cache state
def ckpt_sha(path) -> str:
    """SHA-256 of a checkpoint file: the identity a cached result is bound to."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_models(sel, ds, enc, seed):
    """{arm: (cell_id, checkpoint sha)} of the models the current selection assigns to a work item."""
    out = {}
    for arm in ARMS:
        cfg = sel[(ds, enc)][arm][seed]["config"]
        out[arm] = (cfg["cell_id"], ckpt_sha(Path(cfg["out_dir"]) / "best_C.pt"))
    return out


def cache_state(path, expected=None):
    """'ok' (current version, complete, not quarantined), 'quarantined' (current version, quarantined), 'stale' (the
    cached models differ from the current selection `expected` = {arm: (cell_id, checkpoint sha)}: another learning
    rate was selected or a checkpoint changed) or 'invalid' (absent, unreadable, other version, incomplete)."""
    try:
        r = json.loads(Path(path).read_text())
        if r.get("version") != VERSION or not all(f in r["floors"] for f in FLOORS) or \
                not all(e in r["edits"].get(f, {}) for f in FLOORS for e in EDITS):
            return "invalid"
        if expected is not None:
            for arm, (cid, sha) in expected.items():
                m = r["models"].get(arm) or {}
                if m.get("cell_id") != cid or m.get("ckpt_sha256") != sha:
                    return "stale"
        return "quarantined" if r.get("quarantined", True) else "ok"
    except Exception:
        return "invalid"


def valid(path, expected=None):
    return cache_state(path, expected) == "ok"


def should_score(path, force=False, retry_quarantined=False, expected=None):
    st = cache_state(path, expected)
    if force or st in ("invalid", "stale"):
        return True
    return st == "quarantined" and retry_quarantined


# ------------------------------------------------------------------ units
def selected(run):
    """{(dataset, encoder): {arm: {seed: result}}} at the dev-C lr of each (unit, arm), mixture head, K = 8."""
    out = {}
    for (ds, enc, head, K, arm), seeds in A.select_lr(A.load_cells(Path(run))).items():
        if head == "mix" and K == 8 and arm in ARMS:
            out.setdefault((ds, enc), {})[arm] = seeds
    return out


def tied_units(sel):
    """Units whose stored test tie share is >= TIE_MIN (identical across the unit's cells; checked)."""
    out = []
    for u, arms in sorted(sel.items()):
        ts = {round(r["by_selection"]["C"]["test"]["tie_share"], 12) for s in arms.values() for r in s.values()}
        if len(ts) != 1:
            raise SystemExit(f"{u}: test tie share differs across cells {ts}")
        if ts.pop() >= TIE_MIN:
            out.append(u)
    return out


def work_items(sel, units):
    items = []
    for u in units:
        if set(sel[u]) != set(ARMS):
            raise SystemExit(f"{u}: missing treatments {set(ARMS) - set(sel[u])}")
        seeds = sorted(set.intersection(*(set(sel[u][a]) for a in ARMS)))
        items += [(u[0], u[1], s) for s in seeds]
    return items


def item_name(ds, enc, seed):
    return f"{ds}__{enc}__s{seed}"


# ------------------------------------------------------------------ extraction and scoring
@torch.no_grad()
def extract(model, views, stats, cfg, device, bs=64):
    """Per scored test target: (log_w, mu, log_sigma) (N, K) float64 on the CPU, tau (N,), tie (N,) bool; masks and
    tie rule of coarse_tpp.train.evaluate."""
    model.eval()
    delta = float(cfg["delta"])
    sp = views.coarse
    acc = {k: [] for k in ("lw", "mu", "ls", "tau")}
    for i0 in range(0, len(sp), bs):
        idx = np.arange(i0, min(i0 + bs, len(sp)))
        b = D.make_batch(sp, idx, device)
        m = (b.tmask * score_mask(views, idx, b.iet.shape[1], device)).double() > 0
        law, _ = model(b, stats)
        tau = b.iet[:, 1:].double()
        for k, x in (("lw", law.log_w), ("mu", law.mu), ("ls", law.log_sigma), ("tau", tau)):
            acc[k].append(x[m].detach().cpu())
    out = {k: torch.cat(v) for k, v in acc.items()}
    out["tie"] = Ro.tie_mask(out["tau"], delta)
    return out


def score_item(ds, enc, seed, sel, device, cache):
    if ds not in cache:
        cache.clear()
        cache[ds] = (load(ds, "train"), load(ds, "test"))
    tr, te = cache[ds]
    deltas = {float(sel[(ds, enc)][a][seed]["config"]["delta"]) for a in ARMS}
    if len(deltas) != 1:
        raise SystemExit(f"{ds}/{enc}: the treatments were trained with different resolutions {deltas}")
    delta = deltas.pop()
    rec = dict(version=VERSION, dataset=ds, encoder=enc, seed=seed, delta=delta, device=str(device), models={},
               floors={}, edits={}, kstar={}, quarantined=False)
    P = {}
    for arm in ARMS:
        res = sel[(ds, enc)][arm][seed]
        cfg = res["config"]
        stats = coarse_stats(tr, delta)
        model = build(cfg, stats, tr.coarse.n_types).to(device)
        ckpt = Path(cfg["out_dir"]) / "best_C.pt"
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        P[arm] = p = extract(model, te, stats, cfg, device)
        n, nt = int(len(p["tau"])), int(p["tie"].sum())
        st = res["by_selection"]["C"]["test"]
        e6 = float(floor_score(p["lw"], p["mu"], p["ls"], p["tau"], 1e-6).mean())
        chk = dict(cell_id=cfg["cell_id"], lr_tag=cfg["lr_tag"], ckpt_sha256=ckpt_sha(ckpt), floor_1e6=e6, stored=st["floor_1e-6"],
                   abs_diff=abs(e6 - st["floor_1e-6"]), n=n, n_tie=nt, stored_n=st["n_events"], stored_n_tie=st["n_ties"],
                   log_std=float(stats.log_std), sigma_lo=float(stats.log_std) * math.exp(-5.0))
        chk["ok"] = chk["abs_diff"] <= REPRO_TOL and n == st["n_events"] and nt == st["n_ties"]
        rec["models"][arm] = chk
        rec["quarantined"] |= not chk["ok"]
    rec["n"] = rec["models"]["floor_1e-6"]["n"]
    rec["n_tie"] = rec["models"]["floor_1e-6"]["n_tie"]
    for fm, eps in FLOORS.items():
        rec["floors"][fm] = {arm: floor_sums(P[arm], eps) for arm in ARMS}
    for sm, eps in FLOORS.items():                       # edits of the eps_m-trained model at its tie events
        p = P[sm]
        rows = p["tie"]
        lw, mu, ls = p["lw"][rows], p["mu"][rows], p["ls"][rows]
        k, share, z = kstar(lw, mu, ls, eps)
        g = lambda t: t.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        sig_nt, sig_w, fb = width_values(p, k)
        logd = math.log(delta)
        variants = edit_variants(mu, ls, k, sig_nt, sig_w, logd)
        rec["edits"][sm] = {"none": {fm: floor_sums(p, e)[0] for fm, e in FLOORS.items()}}
        for name, (mu_e, ls_e) in variants.items():
            rec["edits"][sm][name] = {fm: floor_sums(p, e, mu_e, ls_e, rows)[0] for fm, e in FLOORS.items()}
        sig = g(ls).exp()
        lo = rec["models"][sm]["sigma_lo"]
        rec["kstar"][sm] = dict(
            n=int(len(k)), n_fallback=int(fb.sum()), sigma_nontie_median=sig_nt, sigma_lo=lo, log_eps=math.log(eps),
            log_delta=logd, abs_z=quantiles(z.abs().numpy()), mu_minus_logeps=quantiles((g(mu) - math.log(eps)).numpy()),
            sigma=quantiles(sig.numpy()), weight=quantiles(g(lw).exp().numpy()), share=quantiles(share.exp().numpy()),
            log_ratio_scale=quantiles(torch.log(sig_nt / sig).numpy()), frac_scale_widens=float((sig_nt > sig).double().mean()),
            sigma_event=quantiles(sig_w.numpy()), log_ratio_scale_event=quantiles(torch.log(sig_w / sig).numpy()),
            frac_scale_event_widens=float((sig_w > sig).double().mean()),
            frac_sigma_at_lo=float((sig <= lo * (1 + 1e-6)).double().mean()), frac_abs_z_lt1=float((z.abs() < 1).double().mean()))
    return rec


def cmd_score(run, out, shard="0/1", limit=0, cpu=False, threads=4, force=False, retry_quarantined=False):
    torch.set_num_threads(int(threads))
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    sel = selected(run)
    units = tied_units(sel)
    print(f"tied units (test tie share >= {TIE_MIN:.0%}): " + ", ".join(f"{d}/{e}" for d, e in units), flush=True)
    items = work_items(sel, units)
    i, n = (int(x) for x in shard.split("/"))
    todo = [it for k, it in enumerate(items) if k % n == i]
    if limit:
        todo = todo[:limit]
    (out / "cells").mkdir(parents=True, exist_ok=True)
    cache = {}
    for ds, enc, seed in todo:
        dst = out / "cells" / f"{item_name(ds, enc, seed)}.json"
        exp = expected_models(sel, ds, enc, seed)
        if not should_score(dst, force, retry_quarantined, exp):
            if cache_state(dst, exp) == "quarantined":
                print(f"{dst.name}: quarantined cache, not re-scored (use --retry-quarantined or --force)", flush=True)
            continue
        t0 = time.time()
        rec = score_item(ds, enc, seed, sel, device, cache)
        rec["wall_sec"] = round(time.time() - t0, 1)
        tmp = dst.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(rec, indent=1)); os.replace(tmp, dst)
        mg = {fm: margin(rec["floors"][fm], fm, rec["n"])["total"] for fm in FLOORS}
        print(f"{item_name(ds, enc, seed)} n={rec['n']} ties={rec['n_tie']} max|floor diff|="
              f"{max(m['abs_diff'] for m in rec['models'].values()):.2e} quarantined={rec['quarantined']} "
              f"margins " + " ".join(f"{k}={v:+.4f}" for k, v in mg.items()) + f" ({rec['wall_sec']}s)", flush=True)


# ------------------------------------------------------------------ report
def seed_mean_sums(recs, fm, self_arm=None, edit_name=None):
    """{arm: (mean tie_sum, mean nontie_sum)} over seeds at floor fm; with an edit, self_arm's tie sum is replaced."""
    out = {}
    for arm in ARMS:
        t = [r["floors"][fm][arm][0] for r in recs]
        if edit_name and arm == self_arm:
            t = [r["edits"][self_arm][edit_name][fm] for r in recs]
        out[arm] = (float(np.mean(t)), float(np.mean([r["floors"][fm][arm][1] for r in recs])))
    return out


def f4(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:+.4f}"


def pct(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.0%}"


def unit_rows(recs):
    """Per floor: margin (seed means) before and after each edit, removed share, rank-1 retained, seed counts."""
    n = recs[0]["n"]
    out = {}
    for fm in FLOORS:
        base = margin(seed_mean_sums(recs, fm), fm, n)
        row = dict(base=base, edits={})
        for e in EDITS:
            m = margin(seed_mean_sums(recs, fm, fm, e), fm, n)
            seeds_pos = sum(margin({**r["floors"][fm], fm: (r["edits"][fm][e][fm], r["floors"][fm][fm][1])}, fm, n)["total"] > 0
                            for r in recs)
            row["edits"][e] = dict(m=m, removed=(base["total"] - m["total"]) / base["total"] if base["total"] > 0 else float("nan"),
                                   rank1=m["total"] > 0, seeds_rank1=int(seeds_pos))
        row["seeds_rank1"] = int(sum(margin(r["floors"][fm], fm, n)["total"] > 0 for r in recs))
        out[fm] = row
    return out


def peak_rows(recs):
    """Mean tie score per tie event of the eps_m model at the three floors, before and after each edit."""
    out = {}
    for sm in FLOORS:
        out[sm] = {}
        for e in ("none",) + EDITS:
            out[sm][e] = {fm: float(np.mean([r["edits"][sm][e][fm] / r["n_tie"] for r in recs])) for fm in FLOORS}
    return out


def cmd_report(run, out):
    sel = selected(run)
    units = tied_units(sel)
    expected = work_items(sel, units)
    recs, missing, quar, stale = {}, [], [], []
    for ds, enc, seed in expected:
        p = out / "cells" / f"{item_name(ds, enc, seed)}.json"
        st = cache_state(p, expected_models(sel, ds, enc, seed))
        if st == "quarantined":
            quar.append(item_name(ds, enc, seed)); continue
        if st == "stale":
            stale.append(item_name(ds, enc, seed)); continue
        if st != "ok":
            missing.append(item_name(ds, enc, seed)); continue
        recs.setdefault((ds, enc), []).append(json.loads(p.read_text()))
    allrecs = [r for v in recs.values() for r in v]
    maxdiff = max((m["abs_diff"] for r in allrecs for m in r["models"].values()), default=float("nan"))
    L = ["# Component edits: location and scale of the component at the tie (no training)", "",
         "Exploratory diagnostic of saved checkpoints on test events (Supplementary S4), not a correction.", "",
         f"Units (test tie share >= {TIE_MIN:.0%}): " + ", ".join(f"{d}/{e}" for d, e in units) + ".",
         f"Work items scored: {len(allrecs)}/{len(expected)}; missing {len(missing)}; stale {len(stale)} (cached for other models "
         f"or checkpoints than the current selection; re-run score); quarantined {len(quar)} "
         f"(stored test floor 1e-6 not reproduced within {REPRO_TOL:g}); max |floor 1e-6 - stored| = {maxdiff:.2e}.", "",
         "Margin M_m = mean floor score (eps_m) of the model trained with eps_m minus the best of the other four "
         f"({', '.join(ARMS)}), on seed means, nats per scored event; tie + non-tie = M. Edits apply only to the eps_m model's "
         "component k* (largest weighted density at eps_m) at tie events; weights and other components unchanged.", "",
         "- scale: sigma_k* <- weighted median of sigma over all K components of all scored non-tie test events of the same "
         "model (weights = mixture weights, so each non-tie event carries weight 1; lower weighted median). location: mu_k* <- "
         "log(delta). both: scale and location.",
         "- scale_event (secondary contrast): sigma_k* <- weighted median of the same tie event's other components (weights "
         "renormalised; fallback the scale value); scale_event_location: with location.",
         "- Primary quantity: the edited margin itself (rank1 kept iff the edited margin > 0 on seed means; (k) = seeds with a "
         "positive edited margin). The removal share (M - M_edit) / M is secondary: it exceeds 100% whenever an edit drives the "
         "tie score below the runner-up's, as location typically does.", ""]
    if missing:
        L += ["Missing: " + ", ".join(missing[:30]) + (" ..." if len(missing) > 30 else ""), ""]
    if stale:
        L += ["Stale: " + ", ".join(stale), ""]
    if quar:
        L += ["Quarantined: " + ", ".join(quar), ""]
    U = {u: unit_rows(v) for u, v in sorted(recs.items())}
    for i, fm in enumerate(FLOORS):
        L += [f"## 1.{i + 1} Floor {fm} (eps = {FLOORS[fm]:g})", "",
              "| data set | encoder | runner-up | M | tie | non-tie | seeds M>0 | " + " | ".join(f"M_{e} (rank1)" for e in EDITS) + " |",
              "|---|---|---|---:|---:|---:|---:|" + "---:|" * len(EDITS)]
        for (ds, enc), rows in U.items():
            b, E = rows[fm]["base"], rows[fm]["edits"]
            cells = " | ".join(f"{f4(E[e]['m']['total'])} {'Y' if E[e]['rank1'] else 'N'}({E[e]['seeds_rank1']})" for e in EDITS)
            L.append(f"| {ds} | {enc} | {b['runner_up']} | {f4(b['total'])} | {f4(b['tie'])} | {f4(b['nontie'])} | "
                     f"{rows[fm]['seeds_rank1']}/{len(recs[(ds, enc)])} | {cells} |")
        L.append("")
    L += ["## 1.4 Secondary: removal share (M - M_edit) / M (units with M > 0 only)", "",
          "| floor | data set | encoder | " + " | ".join(f"rm {e}" for e in EDITS) + " |", "|---|---|---|" + "---:|" * len(EDITS)]
    for fm in FLOORS:
        for (ds, enc), rows in U.items():
            if rows[fm]["base"]["total"] > 0:
                L.append(f"| {fm} | {ds} | {enc} | " + " | ".join(pct(rows[fm]["edits"][e]["removed"]) for e in EDITS) + " |")
    L += ["", "## 2. Summary across units (median edited margin over all units; rank1 kept among units with M > 0)", "",
          "| floor | units | M > 0 | median M | median tie part | " +
          " | ".join(f"median M_{e} | rank1 kept {e} | (rm {e})" for e in EDITS) + " |",
          "|---|---:|---:|---:|---:|" + "---:|---:|---:|" * len(EDITS)]
    med = lambda xs: float(np.median(xs)) if len(xs) else float("nan")
    for fm in FLOORS:
        rs = [U[u][fm] for u in U]
        if not rs:
            continue
        pos = [r for r in rs if r["base"]["total"] > 0]
        L.append(f"| {fm} | {len(rs)} | {sum(r['base']['total'] > 0 for r in rs)} | {f4(med([r['base']['total'] for r in rs]))} | "
                 f"{f4(med([r['base']['tie'] for r in rs]))} | " +
                 " | ".join(f"{f4(med([r['edits'][e]['m']['total'] for r in rs]))} | {sum(r['edits'][e]['rank1'] for r in pos)}/{len(pos)} | "
                            f"({pct(med([r['edits'][e]['removed'] for r in pos]))})" for e in EDITS) + " |")
    L += ["", "## 3. Where the edited eps_m model's tie score peaks (mean tie-event floor score, seed mean)", "",
          "| data set | encoder | model | edit | " + " | ".join(FLOORS) + " | peak |", "|---|---|---|---|---:|---:|---:|---|"]
    for (ds, enc), v in sorted(recs.items()):
        pk = peak_rows(v)
        for sm in FLOORS:
            for e in ("none",) + EDITS:
                x = pk[sm][e]
                L.append(f"| {ds} | {enc} | {sm} | {e} | " + " | ".join(f"{x[f]:+.3f}" for f in FLOORS) + f" | {max(x, key=x.get)} |")
    L += ["", "## 4. Component k* at tie events (median over seeds of the per-seed statistic)", "",
          "sigma in raw log-time units; sigma_lo = s e^-5, the head's lower bound; |z| = |mu* - log eps| / sigma*.", "",
          "| data set | encoder | model | median abs z | frac abs z < 1 | median mu* - log eps | median sigma* | sigma_lo | frac at lo | "
          "median w* | median share | sigma (scale) | median log(sigma_scale / sigma*) | frac scale widens | "
          "median sigma (scale_event) | median log(sigma_event / sigma*) | frac scale_event widens | fallback |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (ds, enc), v in sorted(recs.items()):
        for sm in FLOORS:
            K = [r["kstar"][sm] for r in v]
            md = lambda key, q="q50": float(np.median([k[key][q] for k in K if k[key]]))
            L.append(f"| {ds} | {enc} | {sm} | {md('abs_z'):.3g} | {np.median([k['frac_abs_z_lt1'] for k in K]):.2f} | "
                     f"{md('mu_minus_logeps'):+.3g} | {md('sigma'):.3g} | {K[0]['sigma_lo']:.3g} | "
                     f"{np.median([k['frac_sigma_at_lo'] for k in K]):.2f} | {md('weight'):.3g} | {md('share'):.3f} | "
                     f"{np.median([k['sigma_nontie_median'] for k in K]):.3g} | {md('log_ratio_scale'):+.2f} | "
                     f"{np.median([k['frac_scale_widens'] for k in K]):.2f} | {md('sigma_event'):.3g} | {md('log_ratio_scale_event'):+.2f} | "
                     f"{np.median([k['frac_scale_event_widens'] for k in K]):.2f} | {sum(k['n_fallback'] for k in K)} |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "summary.json").write_text(json.dumps(dict(
        units=[list(u) for u in units], n_items=len(expected), n_scored=len(allrecs), missing=missing, stale=stale, quarantined=quar,
        max_abs_diff=maxdiff,
        margins={f"{d}/{e}": {fm: dict(base={k: v for k, v in rows[fm]["base"].items()},
                                        edits={x: dict(total=rows[fm]["edits"][x]["m"]["total"], removed=rows[fm]["edits"][x]["removed"],
                                                       rank1=rows[fm]["edits"][x]["rank1"]) for x in EDITS})
                              for fm in FLOORS} for (d, e), rows in U.items()}), indent=1, default=float))
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("score", "report"))
    ap.add_argument("run", help="rank_change run directory")
    ap.add_argument("--out", default=None, help="output directory (default RUN_DIR/component_edits)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="re-score every item, ignoring cached files")
    ap.add_argument("--retry-quarantined", action="store_true", help="re-score items whose cached file is quarantined")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else Path(a.run) / "component_edits"
    if a.cmd == "score":
        cmd_score(Path(a.run), out, a.shard, a.limit, a.cpu, a.threads, a.force, a.retry_quarantined)
    else:
        print("\n".join(cmd_report(Path(a.run), out)))


if __name__ == "__main__":
    main()
