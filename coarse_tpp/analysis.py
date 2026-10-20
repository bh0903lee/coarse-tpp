"""Summary statistics of the experiments. Usage:

    python -m coarse_tpp.analysis EXPERIMENT RUN_DIR [--audit results/audit/audit.json] [--base RUN_DIR ...]

writes RUN_DIR/report.md and RUN_DIR/summary.json.

Common rules (Section 6 and S1): cells with status ok are used (ok_walltime and failed cells are excluded), and
within a group (data set, encoder, head, K) only the seeds whose cells exist for every (treatment, learning rate)
of the grid. Within a unit (group, treatment) the learning rate is chosen by the dev mean over seeds of the
selection criterion (dev C unless stated), and the test values at that learning rate are used. Differences are
paired by seed (synthetic data: averaged within each realization first, n = realizations), and intervals are
one-sided 95% t intervals.
  preserved     lower bound of [interval - floor 1e-10] >= -0.01 nats; violated: upper bound < -0.01
  advantage     interval - strongest non-interval treatment (chosen by dev C per unit): Holm-adjusted p < 0.05
                and mean >= 0.01 nats
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from . import experiments as X
from . import results as R
from . import synthetic as Syn

MIN_REAL = 4          # realizations needed to judge a synthetic comparison (4 of 5)
MIN_SEEDS = 4         # common seeds needed to judge a real-data comparison (4 of 5)
NAN = float("nan")


# ------------------------------------------------------------------ loading and selection
def cell_key(cfg):
    """(group, arm, lr_tag, seed); a group = (dataset, encoder, head, K) holds every compared treatment."""
    return ((cfg["dataset"], cfg["encoder"]["type"], cfg["head"], cfg["K"]), cfg["arm"], cfg["lr_tag"], cfg["seed"])


def common_seed_cells(cells, expected):
    """Within a group keep only the seeds whose cells exist for every (treatment, lr) of the grid."""
    need = {}
    for g, arm, lr, sd in expected:
        need.setdefault(g, set()).add((arm, lr))
    have = {}
    for r in cells:
        g, arm, lr, sd = cell_key(r["config"])
        have.setdefault((g, sd), set()).add((arm, lr))
    keep = {(g, sd) for (g, sd), combos in have.items() if combos >= need.get(g, set())}
    return [r for r in cells if (cell_key(r["config"])[0], r["config"]["seed"]) in keep]


def load_cells(run: Path, with_status=False):
    ok, n_exp, n_wall, failed, expected = [], 0, 0, [], []
    for line in (Path(run) / "grid.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        n_exp += 1
        m = json.loads(line)
        cfg = json.loads(Path(m["config"]).read_text())
        expected.append(cell_key(cfg))
        st, _, res = R.load(cfg)
        if st == "ok" and res is not None:
            ok.append(res)
        elif st == "ok_walltime":
            n_wall += 1
        elif st in R.TERMINAL_BAD:
            failed.append(m["cell_id"])
    out = common_seed_cells(ok, expected)
    st = dict(n_expected=n_exp, n_ok=len(ok), n_used=len(out), n_walltime=n_wall, failed=failed,
              finished=len(ok) + n_wall + len(failed) == n_exp)
    return (out, st) if with_status else out


def status_lines(st):
    return [f"cells: {st['n_ok']}/{st['n_expected']} ok, {st['n_used']} used after the complete-seed restriction, "
            f"{st['n_walltime']} stopped by the wall-clock limit (excluded), {len(st['failed'])} failed (excluded); "
            f"grid finished: {st['finished']}", ""]


def unit_key(res):
    c = res["config"]
    return (c["dataset"], c["encoder"]["type"], c["head"], c["K"], res["arm"])


def select_lr(cells, sel="C"):
    """{unit: {seed: result}} at the lr with the best dev mean (over the unit's seeds) of selection `sel`."""
    by = {}
    for r in cells:
        by.setdefault(unit_key(r), {}).setdefault(r["config"]["lr_tag"], {})[r["seed"]] = r
    crit = lambda x: x["by_selection"].get(sel, x["by_selection"]["C"])["dev_criterion"]
    return {u: max(lrs.items(), key=lambda kv: np.mean([crit(x) for x in kv[1].values()]))[1] for u, lrs in by.items()}


def val(res, key, sel="C"):
    v = res["by_selection"].get(sel, {}).get("test", {}).get(key)
    return NAN if v is None else float(v)


def dev_c(unit_seeds):
    return float(np.mean([x["by_selection"]["C"]["dev_criterion"] for x in unit_seeds.values()]))


def mean_key(seeds, key, sel="C"):
    v = [val(x, key, sel) for x in seeds.values()]
    return float(np.mean(v)) if v and all(x == x for x in v) else NAN


def t_one_sided(x):
    """(mean, one-sided 95% lower bound, one-sided p for H0: mean <= 0)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else NAN), NAN, NAN
    m, sd = x.mean(), x.std(ddof=1)
    se = sd / math.sqrt(n)
    if se == 0:
        return m, m, 0.0 if m > 0 else 1.0
    tq = stats.t.ppf(0.95, n - 1)
    return m, m - tq * se, float(stats.t.sf(m / se, n - 1))


def holm(ps):
    """Holm-adjusted p-values (same order)."""
    ps = np.asarray(ps, float)
    idx = np.argsort(ps)
    adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def paired(units, a, b, key="C", sel="C"):
    """Seed-paired test differences a - b of two units."""
    sa, sb = units.get(a), units.get(b)
    if not sa or not sb:
        return None
    return [val(sa[s], key, sel) - val(sb[s], key, sel) for s in sorted(set(sa) & set(sb))]


def fmt(x, d=4):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.{d}f}"


def lo_str(dd, need=2):
    """(text, mean, lower, upper) of paired differences; 'not judged' below `need` units."""
    n = len(dd or [])
    if n < max(2, need):
        return (f"not judged (n = {n})" if n else "—"), NAN, NAN, NAN
    mu, lo, _ = t_one_sided(dd)
    return f"{mu:+.4f} [{lo:+.4f}, {mu + (mu - lo):+.4f}] (n = {n})", mu, lo, mu + (mu - lo)


def state(lo, hi, margin=-0.01):
    if not math.isfinite(lo):
        return "not judged"
    return "preserved" if lo >= margin else ("violated" if hi < margin else "inconclusive")


def audit_rows(path=None):
    p = Path(path or X.DEFAULT_AUDIT)
    return {r["dataset"]: r for r in json.loads(p.read_text())} if p.exists() else {}


def cost_lines(cells, arms, base):
    w = {}
    for x in cells:
        w.setdefault(x["arm"], []).append(x["wall_sec"])
    b = float(np.median(w.get(base, [NAN])))
    return [f"- {m}: x{float(np.median(w[m])) / b:.2f} (median {float(np.median(w[m])):.0f} s, n = {len(w[m])})"
            for m in arms if m in w]


# ------------------------------------------------------------------ recorded data
def report_real(run, title, arms, audit, controls=(), tied_min=0.01, same_density=None):
    """Test C table, advantage over the strongest non-interval treatment, preservation, and the rankings of the
    treatments by the floor score (eps = 1e-6) and by C."""
    cells, st = load_cells(run, with_status=True)
    units = select_lr(cells)
    groups = sorted({u[:4] for u in units})
    non = [m for m in arms if m != "interval"]
    tie = lambda ds: audit.get(ds, {}).get("tie_share", NAN)
    L = [f"# {title}", "", "Score: test C (recorded-cell log-likelihood per event) at the dev-C learning rate; seed-paired.", ""]
    L += status_lines(st)
    L += ["## Test C by treatment (mean over seeds)", "", "| data set | encoder | head | ties | " + " | ".join(arms) + " |",
          "|---|---|---|---:|" + "---:|" * len(arms)]
    for g in groups:
        cols = [fmt(mean_key(units[g + (m,)], "C")) if g + (m,) in units else "—" for m in arms]
        L.append(f"| {g[0]} | {g[1]} | {g[2]} | {tie(g[0]):.4f} | " + " | ".join(cols) + " |")
    # advantage
    fam = []
    for g in groups:
        if not tie(g[0]) >= tied_min or g[0] in controls:
            continue
        cand = [(dev_c(units[g + (m,)]), m) for m in non if g + (m,) in units]
        if len(cand) < len(non) or g + ("interval",) not in units:
            fam.append((g, None, None)); continue
        best = max(cand)[1]
        fam.append((g, best, paired(units, g + ("interval",), g + (best,))))
    # fewer than MIN_SEEDS common seeds: not judged, and kept in the Holm family with p = 1
    ps = [t_one_sided(d)[2] if d and len(d) >= MIN_SEEDS else 1.0 for _, _, d in fam]
    adj = holm(ps) if ps else []
    L += ["", f"## Advantage: interval - strongest non-interval treatment (dev-C choice), data sets with >= {tied_min:.0%} ties, "
          f"Holm over {len(fam)} comparisons", "",
          "| data set | encoder | head | strongest | mean | lower 95% | Holm p | better |", "|---|---|---|---|---:|---:|---:|---|"]
    n_better = 0
    for (g, best, d), a in zip(fam, adj):
        if not d:
            L.append(f"| {g[0]} | {g[1]} | {g[2]} | missing | | | | |"); continue
        if len(d) < MIN_SEEDS:
            L.append(f"| {g[0]} | {g[1]} | {g[2]} | {best} | not judged (n = {len(d)}) | | | no |"); continue
        m, lo, _ = t_one_sided(d)
        ok = bool(a < 0.05 and m >= 0.01)
        n_better += ok
        L.append(f"| {g[0]} | {g[1]} | {g[2]} | {best} | {fmt(m)} | {fmt(lo)} | {a:.3g} | {'yes' if ok else 'no'} |")
    # preservation
    L += ["", "## Preservation: interval - floor 1e-10 (mean [one-sided 95% bounds])", ""]
    counts = {"preserved": 0, "inconclusive": 0, "violated": 0, "not judged": 0}
    pres = {}
    for g in groups:
        s, mu, lo, hi = lo_str(paired(units, g + ("interval",), g + ("floor_1e-10",)), MIN_SEEDS)
        if s == "—":
            continue
        stt = state(lo, hi); counts[stt] += 1; pres["|".join(map(str, g))] = dict(mean=mu, lo=lo, hi=hi, state=stt)
        L.append(f"- {g[0]} {g[1]} {g[2]}: {s} {stt}")
    L += ["", "Differences to the other treatments (interval - treatment, test C):", ""]
    for g in groups:
        parts = []
        for m in non:
            d = paired(units, g + ("interval",), g + (m,))
            if d:
                parts.append(f"{m} {fmt(float(np.mean(d)))}")
        L.append(f"- {g[0]} {g[1]} {g[2]}: " + ", ".join(parts))
    # rankings
    def ranking(ms):
        out = []
        for g in groups:
            ms_ = [m for m in ms if g + (m,) in units]
            if len(ms_) < 3:
                continue
            e6 = [np.mean([val(x, "floor_1e-6") for x in units[g + (m,)].values()]) for m in ms_]
            cc = [np.mean([val(x, "C") for x in units[g + (m,)].values()]) for m in ms_]
            out.append((g, stats.kendalltau(e6, cc)[0], ms_[int(np.argmax(e6))], ms_[int(np.argmax(cc))],
                        float(max(cc) - cc[int(np.argmax(e6))])))   # mean C lost by taking the floor's top treatment
        return out
    rk = ranking(arms)
    L += ["", "## Ranking of the treatments by the floor score (eps = 1e-6) and by C (Kendall tau; top treatment)", "",
          "| data set | encoder | head | ties | tau | top by floor 1e-6 | top by C | C deficit of floor top |",
          "|---|---|---|---:|---:|---|---|---:|"]
    for g, tau, a, b, dc in rk:
        L.append(f"| {g[0]} | {g[1]} | {g[2]} | {tie(g[0]):.4f} | {fmt(tau, 2)} | {a} | {b}{' (same)' if a == b else ''} | {fmt(dc)} |")
    changed = sum(a != b for _, _, a, b, _ in rk)
    tied = [r for r in rk if tie(r[0][0]) >= tied_min]
    taus = [t for _, t, _, _, _ in rk]
    summ = dict(n_rankings=len(rk), top_changed=changed, n_tied=len(tied), top_changed_tied=sum(a != b for _, _, a, b, _ in tied),
                own_floor_first=sum(a == "floor_1e-6" for _, _, a, _, _ in rk),
                own_floor_first_tied=sum(a == "floor_1e-6" for _, _, a, _, _ in tied),
                floor6_first_by_C=sum(b == "floor_1e-6" for _, _, _, b, _ in rk),
                tau_median=float(np.median(taus)) if taus else None, tau_min=min(taus) if taus else None,
                tau_max=max(taus) if taus else None, n_better=int(n_better), n_family=len(fam),
                preservation=counts, preservation_detail=pres, finished=st["finished"])
    L += ["", f"Top treatment differs between the two scores in {changed}/{len(rk)} combinations "
          f"({summ['top_changed_tied']}/{len(tied)} with >= {tied_min:.0%} ties); the model trained with floor 1e-6 is first "
          f"by the floor score in {summ['own_floor_first']} ({summ['own_floor_first_tied']} tied) and by C in "
          f"{summ['floor6_first_by_C']}; median Kendall tau {fmt(summ['tau_median'], 2)}."]
    if same_density:
        rs = ranking(same_density)
        summ["same_density_top_changed"] = sum(a != b for _, _, a, b, _ in rs)
        L += [f"Restricted to the treatments scored on the same raw density ({', '.join(same_density)}): top treatment "
              f"differs in {summ['same_density_top_changed']}/{len(rs)} combinations."]
    if "point_mass" in arms:
        L += ["", "## interval - point mass (seed-paired test C)", ""]
        for g in groups:
            d = paired(units, g + ("interval",), g + ("point_mass",))
            if d:
                m, lo, _ = t_one_sided(d)
                L.append(f"- {g[0]} {g[1]} {g[2]}: {fmt(m)} (lower {fmt(lo)}, n = {len(d)})")
    L += ["", f"**Preservation: {counts['preserved']} preserved, {counts['inconclusive']} inconclusive, "
          f"{counts['violated']} violated; advantage in {n_better}/{len(fam)} tied comparisons.**"]
    return L, summ


def report_intensity(run, audit):
    L, summ = report_real(run, "Intensity heads (RMTPP, THP)", X.INTENSITY_ARMS, audit)
    cells = load_cells(run)
    units = select_lr(cells)
    base = [a for a in X.INTENSITY_ARMS if a != "interval"]
    L += ["", "## Per head: interval - strongest non-interval treatment on all six data sets (Holm within the head)", ""]
    for h in X.INTENSITY_HEADS:
        fam = []
        for ds in X.INTENSITY_DATASETS:
            g = (ds, "gru", h, 8)
            cand = [(dev_c(units[g + (m,)]), m) for m in base if g + (m,) in units]
            if len(cand) == len(base) and g + ("interval",) in units:
                best = max(cand)[1]
                d = paired(units, g + ("interval",), g + (best,))
                if d and len(d) >= MIN_SEEDS:
                    fam.append((ds, best, d))
        adj = holm([t_one_sided(d)[2] for _, _, d in fam]) if fam else []
        nb = 0
        for (ds, best, d), a in zip(fam, adj):
            m, lo, _ = t_one_sided(d); ok = bool(a < 0.05 and m >= 0.01); nb += ok
            L.append(f"- {h} {ds}: vs {best} {fmt(m)} (lower {fmt(lo)}, Holm p {a:.3g}) {'better' if ok else ''}")
        summ[f"{h}_better_all"] = f"{nb}/{len(fam)}"
    return L, summ


# ------------------------------------------------------------------ synthetic data
def syn_values(cells, key, sel="C"):
    """{(proc, delta, r, arm): mean over seeds of test `key`} at the lr chosen by `sel`, and the dev-C means."""
    units = select_lr(cells, sel)
    out, devc = {}, {}
    for (ds, enc, head, K, arm), seeds in units.items():
        p = Syn.parse(ds)
        if p is None:
            continue
        proc, r, d = p
        out[(proc, d, r, arm)] = float(np.mean([val(x, key, sel) for x in seeds.values()]))
        devc[(proc, d, r, arm)] = dev_c(seeds)
    return out, devc


def best_non_interval(devc, proc, d, r, arms=X.SYN_NON_INTERVAL):
    cand = [(devc[(proc, d, r, m)], m) for m in arms if (proc, d, r, m) in devc]
    return max(cand)[1] if len(cand) == len(arms) else None


def report_synthetic(run, audit=None):
    cells, st = load_cells(run, with_status=True)
    B, devc = syn_values(cells, "B")
    C, _ = syn_values(cells, "C")
    oB, _ = syn_values(cells, "oracle_B")
    main = Syn.main_settings()
    L = ["# Synthetic data with known latent times", "", "Score: latent-time score B (and C) on the test split; per "
         "(process, delta, realization) the mean over seeds at the dev-C learning rate.", ""] + status_lines(st)
    summ = {}
    L += ["## Consistency: B(treatment, delta = 0.01) - B(latent, delta = 0) (mean over realizations)", ""]
    for proc in ("SP", "SC", "NSR"):
        for m in ("interval", "jitter_gap"):
            v = [B.get((proc, 0.01, r, m), NAN) - B.get((proc, 0.0, r, "latent"), NAN) for r in Syn.REALIZATIONS]
            v = [x for x in v if x == x]
            L.append(f"- {proc} {m}: {fmt(float(np.mean(v)) if v else NAN)}")
    for label, M in (("B", B), ("C", C)):
        L += ["", f"## Pre-specified comparison on {label}: interval - strongest non-interval treatment (dev-C choice per "
              "process, delta and realization), realization means over the main settings, Holm over the five processes", "",
              "| process | settings | mean | lower 95% | p | Holm p | per realization | pass |", "|---|---|---:|---:|---:|---:|---|---|"]
        rows, ps = [], []
        for proc in X.SYN_JUDGED:
            ds_ = [d for p_, d in main if p_ == proc]
            per_r = []
            for r in Syn.REALIZATIONS:
                vals = []
                for d in ds_:
                    b = best_non_interval(devc, proc, d, r)
                    if b is None or (proc, d, r, "interval") not in M:
                        vals = None; break
                    vals.append(M[(proc, d, r, "interval")] - M[(proc, d, r, b)])
                if vals:
                    per_r.append(float(np.mean(vals)))
            m, lo, p = t_one_sided(per_r) if len(per_r) >= MIN_REAL else (NAN,) * 3
            rows.append((proc, ds_, m, lo, p, per_r)); ps.append(p if math.isfinite(p) else 1.0)
        adj = holm(ps)
        npass = 0
        for (proc, ds_, m, lo, p, per_r), a in zip(rows, adj):
            ok = math.isfinite(m) and a < 0.05 and m >= 0.01
            npass += ok
            L.append(f"| {proc} | {ds_} | {fmt(m)} | {fmt(lo)} | {p:.3g} | {a:.3g} | {', '.join(fmt(x, 3) for x in per_r)} | {'yes' if ok else 'no'} |")
        verdict = "pass" if npass >= 4 else "partial" if npass == 3 else "fail"
        if all(not math.isfinite(r_[2]) for r_ in rows):
            verdict = "not judged (missing cells)"
        summ[f"pass_{label}"] = dict(n_pass=int(npass), verdict=verdict)
        L.append(f"\n**{label}: {npass}/5 processes ({verdict}; pass = 4 or more, partial = 3).**")
    L += ["", "## interval - other treatments on B (mean over realizations), and the share of ties in the training data", "",
          "| process | delta | ties | - floor 1e-10 | - jitter_gap | - jitter_time | - point_mass | - triangular | strongest (dev C) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    ties = {}
    for x in json.loads(Syn.TIE_TABLE.read_text()):
        ties.setdefault((x["process"], x["delta"]), []).append(x["tie_share"])
    for proc in X.SYN_PROCS:
        for d in Syn.DELTAS[1:]:
            def dm(o):
                v = [B[(proc, d, r, "interval")] - B[(proc, d, r, o)] for r in Syn.REALIZATIONS
                     if (proc, d, r, "interval") in B and (proc, d, r, o) in B]
                return float(np.mean(v)) if v else NAN
            strongest = sorted({best_non_interval(devc, proc, d, r) or "—" for r in Syn.REALIZATIONS})
            L.append(f"| {proc} | {d} | {np.mean(ties[(proc, d)]):.3f} | {fmt(dm('floor_1e-10'))} | {fmt(dm('jitter_gap'))} | "
                     f"{fmt(dm('jitter_time'))} | {fmt(dm('point_mass'))} | {fmt(dm('triangular'))} | {', '.join(strongest)} |")
    arms_all = ("latent",) + X.SYN_ARMS
    L += ["", "## B - true-law B (mean over realizations; more negative = farther from the true law)", "",
          "| process | delta | " + " | ".join(arms_all) + " |", "|---|---:|" + "---:|" * len(arms_all)]
    for proc in X.SYN_PROCS:
        for d in Syn.DELTAS:
            cols = []
            for m in arms_all:
                v = [B[(proc, d, r, m)] - oB[(proc, d, r, m)] for r in Syn.REALIZATIONS if (proc, d, r, m) in B]
                cols.append(fmt(float(np.mean(v)) if v else NAN, 3))
            L.append(f"| {proc} | {d} | " + " | ".join(cols) + " |")
    L += ["", "## Selection by each treatment's own objective: mean of interval - strongest non-interval on B over the "
          "main settings", ""]
    Bs, _ = syn_values(cells, "B", "own")
    agg = [Bs[(proc, d, r, "interval")] - Bs[(proc, d, r, b)] for proc, d in main for r in Syn.REALIZATIONS
           for b in [best_non_interval(devc, proc, d, r)] if b and (proc, d, r, "interval") in Bs and (proc, d, r, b) in Bs]
    L.append(f"- own selection: {fmt(float(np.mean(agg)) if agg else NAN)} (n = {len(agg)})")
    summ["finished"] = st["finished"]
    return L, summ


def syn_paired(units, proc, d, a, b, key, sel="C", realizations=Syn.REALIZATIONS):
    """Realization-level paired differences a - b: seed-paired differences averaged within each realization."""
    dd = []
    for r in realizations:
        ds = Syn.name(proc, r, d)
        x = paired(units, (ds, "gru", "mix", 8, a), (ds, "gru", "mix", 8, b), key, sel)
        if x:
            dd.append(float(np.mean(x)))
    return dd


def syn_mean(units, proc, d, m, key, sel="C"):
    v = [mean_key(units[(Syn.name(proc, r, d), "gru", "mix", 8, m)], key, sel) for r in Syn.REALIZATIONS
         if (Syn.name(proc, r, d), "gru", "mix", 8, m) in units]
    return float(np.mean(v)) if v else NAN


def iw_gap(res, delta):
    """dev C - dev own - log delta of the primary checkpoint (at the selected epoch)."""
    h = next((x for x in res["history"] if x["epoch"] == res["selected_epoch"]), None)
    if h is None or h.get("dev_own") is None:
        return NAN
    return h["dev_C"] - h["dev_own"] - math.log(delta)


def report_iw(run, audit, large=False):
    cells, st = load_cells(run, with_status=True)
    units = select_lr(cells)
    S_list = X.IW_LARGE_S if large else X.IW_S
    iw = [f"iw_jitter_S{S}" for S in S_list]
    settings = X.IW_LARGE_SETTINGS if large else Syn.main_settings()
    L = ["# Multi-sample jitter" + (" with large S" if large else ""), ""] + status_lines(st)
    L += ["## Gap between the dev interval score and each model's own objective, C - own - log delta, at the selected "
          "checkpoint (mean over realizations and seeds)", "",
          "| process | delta | jitter_gap | " + " | ".join(iw) + " | units decreasing in S |", "|---|---:|---:|" + "---:|" * len(iw) + "---|"]
    mono = [0, 0]
    for proc, d in settings:
        rows = {m: [] for m in ("jitter_gap",) + tuple(iw)}
        mu = [0, 0]
        for r in Syn.REALIZATIONS:
            ds = Syn.name(proc, r, d)
            per = {}
            for m in rows:
                s = units.get((ds, "gru", "mix", 8, m))
                if s:
                    for seed, x in s.items():
                        per.setdefault(seed, {})[m] = iw_gap(x, d)
                    rows[m].append(float(np.mean([iw_gap(x, d) for x in s.values()])))
            for seed, v in per.items():
                if all(m in v for m in iw):
                    mu[1] += 1; mu[0] += all(v[iw[k]] > v[iw[k + 1]] for k in range(len(iw) - 1))
        mono[0] += mu[0]; mono[1] += mu[1]
        L.append(f"| {proc} | {d} | " + " | ".join(fmt(float(np.mean(v)) if v else NAN) for v in rows.values()) + f" | {mu[0]}/{mu[1]} |")
    L += ["", "## Test B: treatment - interval (realization-paired, mean [one-sided 95% bounds])", "",
          "| process | delta | B interval | jitter_gap - interval | " + " | ".join(f"{m} - interval" for m in iw) + " |",
          "|---|---:|---:|---|" + "---|" * len(iw)]
    for proc, d in settings:
        cols = [lo_str(syn_paired(units, proc, d, m, "interval", "B"), MIN_REAL)[0] for m in ("jitter_gap",) + tuple(iw)]
        L.append(f"| {proc} | {d} | {fmt(syn_mean(units, proc, d, 'interval', 'B'))} | " + " | ".join(cols) + " |")
    if not large:
        L += ["", "## Real data: test C, treatment - interval (seed-paired), and the dev gap per S", "",
              "| data set | " + " | ".join(f"{m} - interval" for m in ("jitter_gap",) + tuple(iw)) + " | units decreasing in S |",
              "|---|" + "---|" * (len(iw) + 2)]
        for ds in X.TIED_REAL:
            g = (ds, "gru", "mix", 8)
            cols = [lo_str(paired(units, g + (m,), g + ("interval",)), MIN_SEEDS)[0] for m in ("jitter_gap",) + tuple(iw)]
            d = audit.get(ds, {}).get("delta")
            per = {}
            for m in iw:
                for seed, x in (units.get(g + (m,)) or {}).items():
                    per.setdefault(seed, {})[m] = iw_gap(x, d) if d else NAN
            ok = [all(v[iw[k]] > v[iw[k + 1]] for k in range(len(iw) - 1)) for v in per.values() if all(m in v for m in iw)]
            mono[0] += sum(ok); mono[1] += len(ok)
            L.append(f"| {ds} | " + " | ".join(cols) + f" | {sum(ok)}/{len(ok)} |")
    L += ["", f"Gap decreasing in S for {mono[0]}/{mono[1]} realizations and seeds ({mono[0] / max(mono[1], 1):.0%}).", "",
          "## Cost: median training time relative to jitter_gap", ""] + cost_lines(cells, ("jitter_gap",) + tuple(iw) + ("interval",), "jitter_gap")
    return L, dict(monotone=mono, finished=st["finished"])


def report_smoothing(run, audit, new_seeds=False):
    cells, st = load_cells(run, with_status=True)
    units = select_lr(cells)
    L = ["# Smoothing objectives" + (" (new seeds)" if new_seeds else ""), ""] + status_lines(st)
    summ = {}
    margin = -0.005
    if not new_seeds:
        main = Syn.main_settings()
        L += ["## Test B by treatment (synthetic, mean over realizations and seeds)", "",
              "| process | delta | " + " | ".join(X.SMOOTH_ARMS) + " | true law |", "|---|---:|" + "---:|" * (len(X.SMOOTH_ARMS) + 1)]
        for proc, d in main:
            L.append(f"| {proc} | {d} | " + " | ".join(fmt(syn_mean(units, proc, d, m, "B")) for m in X.SMOOTH_ARMS)
                     + f" | {fmt(syn_mean(units, proc, d, 'interval', 'oracle_B'))} |")
        for P in X.SMOOTH_ARMS[:3]:
            s1 = lo_str(syn_paired(units, "SC", 1.0, P, "interval", "B"), MIN_REAL)
            s2 = lo_str(syn_paired(units, "SC", 1.0, P, "jitter_gap", "B"), MIN_REAL)
            others = [(proc, d, lo_str(syn_paired(units, proc, d, P, "interval", "B"), MIN_REAL)) for proc, d in main if (proc, d) != ("SC", 1.0)]
            nb = sum(x[2][2] >= -0.01 for x in others if math.isfinite(x[2][2]))
            cs = [lo_str(syn_paired(units, proc, d, P, "interval", "C"), MIN_REAL) for proc, d in main]
            cr = [(ds, lo_str(paired(units, (ds, "gru", "mix", 8, P), (ds, "gru", "mix", 8, "interval")), MIN_SEEDS)) for ds in X.TIED_REAL]
            nc = sum(x[2] >= margin for x in cs if math.isfinite(x[2])) + sum(x[1][2] >= margin for x in cr if math.isfinite(x[1][2]))
            summ[P] = dict(SC_vs_interval=s1[1:3], SC_vs_jitter=s2[1:3], B_no_loss=f"{nb}/{len(others)}", C_no_loss=f"{nc}/{len(cs) + len(cr)}")
            L += ["", f"### {P}", "", f"- SC delta = 1, B: {P} - interval {s1[0]}; {P} - jitter_gap {s2[0]}",
                  f"- other settings, B: {P} - interval lower bound >= -0.01 on {nb}/{len(others)}: " +
                  "; ".join(f"{p} {d}: {x[0]}" for p, d, x in others),
                  f"- C non-inferiority (lower bound of {P} - interval >= {margin}): {nc}/{len(cs) + len(cr)}; real data: " +
                  "; ".join(f"{ds}: {x[0]}" for ds, x in cr)]
        L += ["", "## Cost: median training time relative to jitter_gap", ""] + cost_lines(cells, X.SMOOTH_ARMS, "jitter_gap")
    else:
        L += ["## C(triangular) - C(interval), seed-paired over the new seeds (non-inferiority margin -0.005)", "",
              "| data set | mean [one-sided 95% bounds] | non-inferior |", "|---|---|---|"]
        n_ok = 0
        for ds in X.TIED_REAL:
            s, mu, lo, hi = lo_str(paired(units, (ds, "gru", "mix", 8, "triangular"), (ds, "gru", "mix", 8, "interval")), 32)
            ok = math.isfinite(lo) and lo >= margin
            n_ok += ok
            summ[ds] = dict(mean=mu, lo=lo, hi=hi)
            L.append(f"| {ds} | {s} | {'yes' if ok else 'no'} |")
        L += ["", f"**Non-inferior on {n_ok}/{len(X.TIED_REAL)} data sets.**", "",
              "## Cost: median training time relative to interval", ""] + cost_lines(cells, X.SMOOTH_SEEDS_ARMS, "interval")
    summ["finished"] = st["finished"]
    return L, summ


def report_misspecification(run, audit, base=None):
    cells, st = load_cells(run, with_status=True)
    units = select_lr(cells)
    L = ["# Misspecified resolution", ""] + status_lines(st)
    L += ["## Real data: test C(delta_model = m x delta) - C(m = 1), same treatment, seed-paired", "",
          "| data set | treatment | " + " | ".join(f"m = {m:g}" for m in X.MISSPEC_MULTS if m != 1.0) + " |",
          "|---|---|" + "---|" * (len(X.MISSPEC_MULTS) - 1)]
    for ds in X.TIED_REAL:
        g = (ds, "gru", "mix", 8)
        for a in X.MISSPEC_ARMS:
            cols = []
            for m in X.MISSPEC_MULTS:
                if m == 1.0:
                    continue
                d = paired(units, g + (X.misspec_tag(a, m),), g + (X.misspec_tag(a, 1.0),))
                cols.append(f"{np.mean(d):+.4f}" if d else "—")
            L.append(f"| {ds} | {a} | " + " | ".join(cols) + " |")
    if base:
        bcells = load_cells(base)
        bunits = select_lr(bcells)
        allu = dict(bunits); allu.update(units)
        L += ["", "## Synthetic data: test B(interval, delta_model = delta / 2) - B(interval) (seed 0)", ""]
        for proc, d in Syn.main_settings():
            v = []
            for r in X.MISSPEC_SYN_REALIZATIONS:
                ds = Syn.name(proc, r, d)
                dd = paired(allu, (ds, "gru", "mix", 8, X.misspec_tag("interval", 0.5)), (ds, "gru", "mix", 8, "interval"), "B")
                if dd:
                    v += dd
            L.append(f"- {proc} {d}: {', '.join(f'{x:+.3f}' for x in v) or '—'}")
    return L, dict(finished=st["finished"])


def report_marks(run, audit):
    cells, st = load_cells(run, with_status=True)
    tc = [c for c in cells if c["arm"] != "mark_only"]
    mo = [c for c in cells if c["arm"] == "mark_only"]
    units = select_lr(tc, sel="J")
    munits = select_lr(mo, sel="M")
    L = ["# Marks", "", "Treatments with a mark head: learning rate and checkpoint by dev J = C + mark. Control "
         "trained on marks alone: by dev mark log-likelihood. Test mark log-likelihood per event.", ""] + status_lines(st)
    L += ["| data set | ties | " + " | ".join(X.MARK_ARMS) + " | mark_only | " + " | ".join(f"{m} - floor 1e-10" for m in X.MARK_ARMS[1:]) + " | mark_only - floor 1e-10 |",
          "|---|---:|" + "---:|" * (len(X.MARK_ARMS) + 1) + "---|" * len(X.MARK_ARMS)]
    for ds in X.MARK_DATASETS:
        g = (ds, "gru", "mix", 8)
        cols = [fmt(mean_key(units[g + (m,)], "mark", "J")) if g + (m,) in units else "—" for m in X.MARK_ARMS]
        s = munits.get(g + ("mark_only",))
        cols.append(fmt(mean_key(s, "mark", "M")) if s else "—")
        diffs = [lo_str(paired(units, g + (m,), g + ("floor_1e-10",), "mark", "J"), MIN_SEEDS)[0] for m in X.MARK_ARMS[1:]]
        b = units.get(g + ("floor_1e-10",))
        dd = [val(s[k], "mark", "M") - val(b[k], "mark", "J") for k in sorted(set(s) & set(b))] if s and b else []
        diffs.append(lo_str(dd, MIN_SEEDS)[0])
        L.append(f"| {ds} | {audit.get(ds, {}).get('tie_share', NAN):.4f} | " + " | ".join(cols + diffs) + " |")
    return L, dict(finished=st["finished"])


def report_coarsened(run, audit, latent=None):
    L, summ = report_real(run, "Coarsened real data", X.COARSEN_ARMS, audit)
    p = Path(latent) if latent else Path(run) / "latent_scores.jsonl"
    if not p.exists():
        L += ["", f"Latent-time scores: run `python -m coarse_tpp.latent_score {run}` to add the B tables."]
        return L, summ
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    good = [dict(config=r["config"], seed=r["seed"], arm=r["arm"],
                 by_selection={"C": {"dev_criterion": r["dev_crit"], "test": dict(B=r["test"]["B"], C=r["test"]["C"])}})
            for r in rows if not r["mismatch"]]
    units = select_lr(good)
    L += ["", f"## Latent-time score B ({len(good)} of {len(rows)} cells reproduce their stored C)", ""]
    for enc in ("gru", "transformer"):
        L += [f"### {enc}", "", "| data set | B interval | - floor 1e-10 | - jitter_gap | - jitter_time | - point_mass |",
              "|---|---:|---|---|---|---|"]
        for ds in X.COARSENED:
            g = (ds, enc, "mix", 8)
            if g + ("interval",) not in units:
                continue
            cols = [lo_str(paired(units, g + ("interval",), g + (b,), "B"), MIN_SEEDS)[0]
                    for b in ("floor_1e-10", "jitter_gap", "jitter_time", "point_mass")]
            L.append(f"| {ds} | {fmt(mean_key(units[g + ('interval',)], 'B'))} | " + " | ".join(cols) + " |")
        L.append("")
    return L, summ


def run_report(experiment, run, audit_path=None, base=None):
    audit = audit_rows(audit_path)
    run = Path(run)
    if experiment == "synthetic":
        L, s = report_synthetic(run)
    elif experiment == "rank_change":
        L, s = report_real(run, "Rankings of training treatments (mixture head)", X.RANK_ARMS, audit,
                           controls=X.RANK_CONTROLS, same_density=X.SAME_DENSITY_ARMS)
    elif experiment == "resplit":
        L, s = report_real(run, "New random splits", X.RESPLIT_ARMS, audit, controls=X.RESPLIT_CONTROLS)
    elif experiment == "new_datasets":
        L, s = report_real(run, "Further data sets", X.RESPLIT_ARMS, audit, controls=X.NEW_CONTROLS)
    elif experiment == "process_logs":
        L, s = report_real(run, "Business process logs", X.PROCESS_ARMS, audit)
    elif experiment == "intensity_heads":
        L, s = report_intensity(run, audit)
    elif experiment in ("iw_jitter", "iw_jitter_large"):
        L, s = report_iw(run, audit, large=experiment == "iw_jitter_large")
    elif experiment in ("inbin_smoothing", "inbin_smoothing_seeds"):
        L, s = report_smoothing(run, audit, new_seeds=experiment == "inbin_smoothing_seeds")
    elif experiment == "delta_misspecification":
        L, s = report_misspecification(run, audit, base)
    elif experiment == "marks":
        L, s = report_marks(run, audit)
    elif experiment == "coarsened_real":
        L, s = report_coarsened(run, audit)
    elif experiment == "floor_convention":
        raise SystemExit("floor_convention is reported by scripts/rescore_trained.py score/report (see README)")
    else:
        raise SystemExit(f"unknown experiment {experiment}")
    (run / "report.md").write_text("\n".join(L) + "\n")
    (run / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    return L, s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=sorted(X.BUILDERS))
    ap.add_argument("run")
    ap.add_argument("--audit", default=None)
    ap.add_argument("--base", default=None, help="delta_misspecification: run directory of the synthetic experiment")
    a = ap.parse_args(argv)
    L, _ = run_report(a.experiment, a.run, a.audit, a.base)
    print("\n".join(L))


if __name__ == "__main__":
    main()
