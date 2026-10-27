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


def planned_groups(run):
    """Groups in the grid, including those without any usable result."""
    groups = set()
    for line in (Path(run) / "grid.jsonl").read_text().splitlines():
        if line.strip():
            cfg = json.loads(Path(json.loads(line)["config"]).read_text())
            groups.add(cell_key(cfg)[0])
    return sorted(groups)


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
    groups = planned_groups(run)
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
            L.append(f"| {g[0]} | {g[1]} | {g[2]} | pending (missing cells) | | | 1 | no |"); continue
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
            s = "pending (missing cells)"
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
        for g in planned_groups(run):
            if g[2] != h:
                continue
            ds = g[0]
            cand = [(dev_c(units[g + (m,)]), m) for m in base if g + (m,) in units]
            if len(cand) == len(base) and g + ("interval",) in units:
                best = max(cand)[1]
                d = paired(units, g + ("interval",), g + (best,))
                fam.append((ds, best, d))
            else:
                fam.append((ds, None, None))
        adj = holm([t_one_sided(d)[2] if d and len(d) >= MIN_SEEDS else 1.0 for _, _, d in fam]) if fam else []
        nb = 0
        for (ds, best, d), a in zip(fam, adj):
            if not d or len(d) < MIN_SEEDS:
                L.append(f"- {h} {ds}: pending (n = {len(d or [])}); Holm family p = 1")
                continue
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


LATENT_SCHEMA = 2       # rows of coarse_tpp.latent_score that carry B_alt


def latent_identity_current(row):
    """Validate checkpoint-bound rows. Rows without an identity (deposited rows, or rows written before version 1.1.2)
    remain readable but are not checked against the current checkpoint; re-score them to validate."""
    if "identity" not in row:
        return True
    source = Path(row.get("source_config", ""))
    if not source.is_file():
        return False
    cfg = json.loads(source.read_text())
    state, _, res = R.load(cfg)
    return bool(state == "ok" and row["identity"] is not None
                and row["identity"] == R.checkpoint_identity(cfg, res, "C"))


def latent_rows(paths, with_stale=False):
    """Rows of the latent_scores.jsonl files of several runs as result-like dicts keyed by (dataset, encoder, arm,
    lr_tag, seed); by_selection.C holds dev_criterion, test {B, B_alt, C} and dev {B, B_alt}. Rows flagged mismatch
    and rows of an older schema (without B_alt; they need re-scoring with coarse_tpp.latent_score) are left out;
    with_stale=True also returns the number of such older rows."""
    out, stale = {}, 0
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("mismatch") or not latent_identity_current(r):
                continue
            if "B_alt" not in (r.get("test") or {}) or "B_alt" not in (r.get("dev") or {}):
                stale += 1
                continue
            c = r["config"]
            key = (c["dataset"], c["encoder"]["type"], r["arm"], c["lr_tag"], r["seed"])
            out[key] = dict(config=dict(dataset=c["dataset"], encoder=c["encoder"], head=c["head"], K=c["K"], lr_tag=c["lr_tag"]),
                            seed=r["seed"], arm=r["arm"], wall_sec=r.get("wall_sec"),
                            by_selection={"C": {"dev_criterion": r["dev_crit"], "test": r["test"], "dev": r["dev"]}})
    return (out, stale) if with_stale else out


def grid_units(runs, arm="triangular", keep=lambda ds: True):
    """(dataset, encoder) combinations planned for `arm` in the grids (grid.jsonl) of `runs`, independent of which
    results exist: the expected population of a report never shrinks with missing results."""
    out = set()
    for run in runs:
        p = Path(run) / "grid.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            cfg = json.loads(Path(m["config"]).read_text())
            if cfg.get("arm", m.get("arm")) == arm and keep(cfg["dataset"]):
                out.add((cfg["dataset"], cfg["encoder"]["type"]))
    return sorted(out)


def is_recorded(ds: str) -> bool:
    """A recorded data set of the paper (neither synthetic nor coarsened on purpose)."""
    return not ds.startswith("synth_") and not ds.startswith("coarsened_")


def report_coarsened(run, audit, latent=None, datasets=X.COARSENED, arms=X.COARSEN_ARMS):
    L, summ = report_real(run, "Coarsened real data", arms, audit)
    p = Path(latent) if latent else Path(run) / "latent_scores.jsonl"
    if not p.exists():
        L += ["", f"Latent-time scores: run `python -m coarse_tpp.latent_score {run}` to add the B tables."]
        return L, summ
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    good = [dict(config=r["config"], seed=r["seed"], arm=r["arm"],
                 by_selection={"C": {"dev_criterion": r["dev_crit"], "test": dict(B=r["test"]["B"], C=r["test"]["C"])}})
            for r in rows if not r["mismatch"] and latent_identity_current(r)]
    units = select_lr(good)
    others = [a for a in arms if a not in ("interval", "floor_1e-6", "floor_1e-5")]
    L += ["", f"## Latent-time score B ({len(good)} of {len(rows)} cells reproduce their stored C)", ""]
    for enc in ("gru", "transformer"):
        L += [f"### {enc}", "", "| data set | B interval | " + " | ".join(f"- {b}" for b in others) + " |",
              "|---|---:|" + "---|" * len(others)]
        for ds in datasets:
            g = (ds, enc, "mix", 8)
            if g + ("interval",) not in units:
                continue
            cols = [lo_str(paired(units, g + ("interval",), g + (b,), "B"), MIN_SEEDS)[0] for b in others]
            L.append(f"| {ds} | {fmt(mean_key(units[g + ('interval',)], 'B'))} | " + " | ".join(cols) + " |")
        L.append("")
    return L, summ


# ------------------------------------------------------------------ triangular kernel (Supplementary S4)
TRI_BASE = ("floor_1e-10", "floor_1e-6", "jitter_gap", "jitter_time", "point_mass", "interval")
BETTER_MEAN, LOSS_C = 0.01, -0.005
PENDING = "pending"


def better(d):
    """One-sided 95% lower bound > 0 and mean >= BETTER_MEAN (n >= 2)."""
    if not d or len(d) < 2:
        return False
    m, lo, _ = t_one_sided(d)
    return bool(lo > 0 and m >= BETTER_MEAN)


def tstat(d):
    if not d:
        return None
    m, lo, _ = t_one_sided(d)
    return dict(mean=float(m), lower=float(lo) if math.isfinite(lo) else None, n=len(d))


def fmt_d(d):
    s = tstat(d)
    if not s:
        return "—", "—", ""
    return fmt(s["mean"]), fmt(s["lower"]) if s["lower"] is not None else "—", "**yes**" if better(d) else ""


def dev_val(seeds: dict, key: str) -> float:
    return float(np.mean([x["by_selection"]["C"]["dev"][key] for x in seeds.values()]))


def unit_select(cells, ds, enc, arms, seeds, lrs=tuple(X.LRS)):
    """{arm: {seed: result}} at the dev-C lr chosen on `seeds`, or (None, missing (arm, lr, seed) list)."""
    missing = [(a, lr, s) for a in arms for lr in lrs for s in seeds if (ds, enc, a, lr, s) not in cells]
    if missing:
        return None, missing
    out = {}
    for a in arms:
        best = max(lrs, key=lambda lr: np.mean([cells[(ds, enc, a, lr, s)]["by_selection"]["C"]["dev_criterion"] for s in seeds]))
        out[a] = {s: cells[(ds, enc, a, best, s)] for s in seeds}
    return out, []


def best_readout(sel, arms=TRI_BASE):
    """(treatment, key): per treatment the readout (B or B_alt) with the higher dev mean, then the treatment with the
    highest dev value under its readout."""
    cand = []
    for a in arms:
        k = max(("B", "B_alt"), key=lambda key: dev_val(sel[a], key))
        cand.append((dev_val(sel[a], k), a, k))
    _, a, k = max(cand)
    return a, k


def diff(sel, a, b, ka="B", kb="B"):
    seeds = sorted(set(sel[a]) & set(sel[b]))
    return [val(sel[a][s], ka) - val(sel[b][s], kb) for s in seeds]


def syn_units(cells, settings=None):
    """{(proc, delta): {realization data set: {arm: {seed: result}}}} over the synthetic main settings (default
    Syn.main_settings()), lr per (data set, arm) by dev C over the arm's seeds."""
    main = set(Syn.main_settings() if settings is None else settings)
    by = {}
    for (ds, enc, arm, lr, s), r in cells.items():
        p = Syn.parse(ds)
        if p and (p[0], p[2]) in main:
            by.setdefault(ds, {}).setdefault(arm, {}).setdefault(lr, {})[s] = r
    out = {}
    for ds, arms in by.items():
        proc, _, d = Syn.parse(ds)
        sel = {}
        for a, lrs in arms.items():
            best = max(lrs, key=lambda lr: np.mean([x["by_selection"]["C"]["dev_criterion"] for x in lrs[lr].values()]))
            sel[a] = lrs[best]
        out.setdefault((proc, d), {})[ds] = sel
    return out


def diff_syn(real, a, b=None, ka="B", kb="B", use_best=False):
    """Per realization the mean over common seeds of a - b (b = the best-readout baseline when use_best); returns
    (list over realizations, realizations skipped)."""
    out, skipped = [], 0
    for ds, sel in sorted(real.items()):
        if a not in sel:
            skipped += 1; continue
        if use_best:
            if not all(x in sel for x in TRI_BASE):
                skipped += 1; continue
            common_all = set.intersection(*(set(sel[x]) for x in TRI_BASE + (a,)))
            if not common_all:
                skipped += 1; continue
            bb, kbb = best_readout({x: {s: sel[x][s] for s in common_all} for x in TRI_BASE})
        else:
            bb, kbb = b, kb
        if bb not in sel:
            skipped += 1; continue
        common = sorted(set(sel[a]) & set(sel[bb]))
        if not common:
            skipped += 1; continue
        out.append(float(np.mean([val(sel[a][s], ka) for s in common]) - np.mean([val(sel[bb][s], kbb) for s in common])))
    return out, skipped


def cost_table(cells, ref_arm="jitter_gap"):
    """Median and quartiles of wall(arm) / wall(ref_arm) over matched (data set, encoder, lr, seed) cells."""
    ref = {(ds, enc, lr, s): r["wall_sec"] for (ds, enc, a, lr, s), r in cells.items() if a == ref_arm and r.get("wall_sec")}
    ratios = {}
    for (ds, enc, a, lr, s), r in cells.items():
        k = (ds, enc, lr, s)
        if a != ref_arm and r.get("wall_sec") and k in ref:
            ratios.setdefault(a, []).append(r["wall_sec"] / ref[k])
    if not ratios:
        return ["(no matched cells)"]
    L = ["| arm | matched cells | median ratio | 25% | 75% |", "|---|---:|---:|---:|---:|"]
    for a, v in sorted(ratios.items()):
        q = np.quantile(v, [0.25, 0.5, 0.75])
        L.append(f"| {a} | {len(v)} | {q[1]:.2f} | {q[0]:.2f} | {q[2]:.2f} |")
    return L


def triangular_c_table(runs, seeds=X.REAL_SEEDS):
    """Test C of the triangular objective against interval and jitter_gap on the recorded data sets: the expected
    combinations (data set, encoder) are those planned for the triangular arm on recorded data in the grids of `runs`
    (the triangular_recorded experiment); synthetic and coarsened data never enter this table. Values are the
    result.json of the runs; no loss = one-sided lower bound >= LOSS_C against both. Returns (status, lines, n_ok,
    n_units); status is PENDING without a planned combination or with incomplete ones."""
    units = grid_units(runs, keep=is_recorded)
    cells = {}
    for run in runs:
        if not (Path(run) / "grid.jsonl").exists():
            continue
        for x in load_cells(Path(run)):
            c = x["config"]
            if x["arm"] in ("triangular", "interval", "jitter_gap") and is_recorded(c["dataset"]):
                cells[(c["dataset"], c["encoder"]["type"], x["arm"], c["lr_tag"], x["seed"])] = x
    L = ["| data set | encoder | test ties | n | triangular - interval | lower | triangular - jitter_gap | lower | no loss |",
         "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    if not units:
        return PENDING, L + ["", "No grid with the triangular arm on recorded data among the runs (pass the triangular_recorded "
                             "run with --base): pending."], 0, 0
    status, n_ok = True, 0
    for ds, enc in units:
        sel, missing = unit_select(cells, ds, enc, ("triangular", "interval", "jitter_gap"), seeds)
        if sel is None:
            L.append(f"| {ds} | {enc} | — | — | pending: {len(missing)} missing, e.g. {missing[:3]} |")
            status = PENDING if status is not False else False
            continue
        d1, d2 = diff(sel, "triangular", "interval", "C", "C"), diff(sel, "triangular", "jitter_gap", "C", "C")
        s1, s2 = tstat(d1), tstat(d2)
        tie = float(np.mean([x["test"]["tie_share"] for x in sel["triangular"].values()]))
        nl = bool(s1["lower"] is not None and s2["lower"] is not None and s1["lower"] >= LOSS_C and s2["lower"] >= LOSS_C)
        n_ok += nl
        if not nl:
            status = False
        L.append(f"| {ds} | {enc} | {tie:.3f} | {s1['n']} | {fmt(s1['mean'])} | {fmt(s1['lower'])} | {fmt(s2['mean'])} | "
                 f"{fmt(s2['lower'])} | {'yes' if nl else '**no**'} |")
    L.append(f"\nNo loss on {n_ok}/{len(units)} planned combinations.")
    return status, L, n_ok, len(units)


def report_triangular(run, bases=(), seeds=X.REAL_SEEDS):
    """Triangular-kernel objective against the other treatments on the latent-time score B (coarsened data, both
    readouts, from the latent_scores.jsonl files of `run` and `bases`), on B of the synthetic settings, and on C of the
    recorded data (result.json of the runs). The decision rule was fixed before the runs (Supplementary S4)."""
    runs = [Path(run)] + [Path(b) for b in bases]
    cells, stale = latent_rows([r / "latent_scores.jsonl" for r in runs], with_stale=True)
    tri_units = grid_units(runs, keep=lambda ds: ds.startswith("coarsened_"))      # planned, not merely scored
    L = ["# Triangular-kernel likelihood against the other treatments", "",
         "Exploratory comparison (Supplementary S4). B = log q_B(s) of the latent time since the last record, on the test "
         "split; learning rate per unit by dev C on the common seed set; real units seed-paired, synthetic settings paired "
         "over realizations. better = one-sided 95% lower bound > 0 and mean >= 0.01 nats. jitter_time shares the smoothed "
         "readout but also jitters the inputs; iw_jitter_triangular keeps the recorded inputs.",
         f"Latent scores read from: {', '.join(str(r) for r in runs if (r / 'latent_scores.jsonl').exists())}; "
         f"{len(cells)} cells without mismatch" + (f"; {stale} rows of an older schema without B_alt left out (re-score them "
                                                   "with coarse_tpp.latent_score)" if stale else "") + ".",
         f"Planned coarsened combinations with the triangular arm (from the grids): {len(tri_units)}"
         + ("" if tri_units else " — none found: pass the triangular_coarsened / coarsened_more runs; verdict pending") + ".", ""]
    summary = dict(real=[], syn=[])
    L += ["## 1. Coarsened real data", "",
          "| data set | r | encoder | n | B triangular | B jitter_time | - jitter_time | lower | better | best readout | - best | lower | better | "
          "- jitter_gap | - interval | " + " | ".join(f"- {s}" for s in X.TRI_IW_ARMS) + " |",
          "|---|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---:|---:|" + "---:|" * len(X.TRI_IW_ARMS)]
    for ds, enc in tri_units:
        r = X.coarsened_r(ds)
        sel, missing = unit_select(cells, ds, enc, ("triangular",) + TRI_BASE, seeds)
        if sel is None:
            L.append(f"| {ds} | {r} | {enc} | pending: {len(missing)} missing cells, e.g. {missing[:3]} |")
            summary["real"].append(dict(dataset=ds, r=r, encoder=enc, status=PENDING, n_missing=len(missing)))
            continue
        da = diff(sel, "triangular", "jitter_time")
        br = best_readout(sel)
        db = diff(sel, "triangular", br[0], "B", br[1])
        dj, dg = diff(sel, "triangular", "jitter_gap"), diff(sel, "triangular", "interval")
        tiw = []
        for s in X.TRI_IW_ARMS:
            ts, _ = unit_select(cells, ds, enc, (s,), seeds)
            tiw.append(fmt_d(diff({**sel, **ts}, "triangular", s))[0] if ts else "—")
        ma, la, ba = fmt_d(da); mb, lb, bb = fmt_d(db)
        mean_b = lambda a: fmt(float(np.mean([val(x, "B") for x in sel[a].values()])))
        L.append(f"| {ds} | {r} | {enc} | {len(da)} | {mean_b('triangular')} | {mean_b('jitter_time')} | {ma} | {la} | {ba} | "
                 f"{br[0]} ({br[1]}) | {mb} | {lb} | {bb} | {fmt_d(dj)[0]} | {fmt_d(dg)[0]} | " + " | ".join(tiw) + " |")
        summary["real"].append(dict(dataset=ds, r=r, encoder=enc, status="complete", a=better(da), b=better(db),
                                    a_stat=tstat(da), b_stat=tstat(db), baseline=list(br)))
    syn = syn_units(cells)
    L += ["", "## 2. Synthetic main settings (Syn.main_settings(); paired over realizations; seeds common to both arms)", "",
          "| process | delta | - jitter_time (n) | lower | better | - best readout (n) | lower | better | - jitter_gap | - interval | " +
          " | ".join(f"- {s}" for s in X.TRI_IW_ARMS) + " | realizations skipped (a / b) |",
          "|---|---:|---:|---:|---|---:|---:|---|---:|---:|" + "---:|" * len(X.TRI_IW_ARMS) + "---|"]
    for st in sorted(syn):
        real = syn[st]
        da, sa = diff_syn(real, "triangular", "jitter_time")
        db, sb = diff_syn(real, "triangular", use_best=True)
        ma, la, ba = fmt_d(da); mb, lb, bb = fmt_d(db)
        L.append(f"| {st[0]} | {st[1]} | {ma} ({len(da)}) | {la} | {ba} | {mb} ({len(db)}) | {lb} | {bb} | "
                 f"{fmt_d(diff_syn(real, 'triangular', 'jitter_gap')[0])[0]} | {fmt_d(diff_syn(real, 'triangular', 'interval')[0])[0]} | " +
                 " | ".join(fmt_d(diff_syn(real, "triangular", s)[0])[0] for s in X.TRI_IW_ARMS) + f" | {sa} / {sb} |")
        summary["syn"].append(dict(process=st[0], delta=st[1], a=better(da), b=better(db), a_stat=tstat(da), b_stat=tstat(db)))
    L += ["", f"## 3. Recorded data: test C of triangular against interval and jitter_gap (no loss = lower bound >= {LOSS_C})", ""]
    noloss, rows, n_ok, n_units = triangular_c_table(runs, seeds)
    L += rows
    L += ["", "## 4. Training time, matched cells (same data set, encoder, lr, seed): ratio to jitter_gap", ""] + cost_table(cells)
    elig = [x for x in summary["real"] if (x["r"] or 0) >= 0.3]
    pend = [x for x in elig if x["status"] == PENDING]
    comp = [x for x in elig if x["status"] == "complete"]
    n_a = sum(bool(x["a"]) for x in comp)
    n_both = sum(bool(x["a"] and x["b"]) for x in comp)
    need = math.ceil(2 * len(elig) / 3) if elig else None
    if noloss is False or (comp and not pend and noloss is True and n_both < need):
        verdict = False
    elif not elig or pend or noloss == PENDING:
        verdict = PENDING
    else:
        verdict = bool(n_both >= need and noloss is True)
    L += ["", "## 5. Decision rule (fixed before the runs)", "",
          f"planned coarsened units with r >= 0.3: {len(elig)} (complete {len(comp)}, pending {len(pend)}); better than jitter_time: {n_a}; "
          f"better than jitter_time AND than the best-readout baseline: {n_both} (needed: {need}); recorded-data C without loss on "
          f"every combination: {noloss} ({n_ok}/{n_units})", f"**condition met: {verdict}**"]
    summary.update(verdict=verdict, n_eligible=len(elig), n_pending=len(pend), n_better_a=n_a, n_both_better=n_both, needed=need,
                   c_noloss=noloss, c_noloss_count=[n_ok, n_units])
    return L, summary


# ------------------------------------------------------------------ component bounds (Supplementary S4)
BOUND_ARMS = ("none",) + tuple(X.BOUNDS)
FLOOR_KEYS = tuple(X.FLOOR_ARMS)


def bound_arm(treatment, bound):
    return treatment if bound == "none" else X.bounds_tag(treatment, bound)


def report_component_bounds(run, bases=(), seeds=X.REAL_SEEDS):
    """Per data set and bound {none (the rank_change run), sigma_k >= 2, mu_k >= log 1e-4, both}: the self-preference
    margin of each floor-trained model (its test floor score minus the best of the other four same-density treatments,
    with tie / non-tie contributions), the floor spread of each floor-trained model, the top treatment under each
    floor and under C, Kendall's tau between the floor 1e-6 and C rankings, and the seed-paired change in test C of
    every treatment against the unbounded arm. Learning rates by dev C on the seeds common to every arm and treatment
    of the data set (GRU only)."""
    cells = {}
    for r in [Path(run)] + [Path(b) for b in bases]:
        if not (r / "grid.jsonl").exists():
            continue
        for x in load_cells(r):
            c = x["config"]
            if c["encoder"]["type"] == "gru" and c["head"] == "mix":
                cells[(c["dataset"], "gru", x["arm"], c["lr_tag"], x["seed"])] = x
    datasets = sorted({ds for (ds, enc, a, lr, s) in cells if a in {X.bounds_tag(t, b) for t in X.BOUNDS_ARMS for b in X.BOUNDS}})
    L = ["# Component bounds: width / location intervention on the floor's self-preference", "",
         "Exploratory (Supplementary S4). Learning rate per (bound, treatment) by dev C on the seed set common to all arms and "
         "treatments of the data set; seed means. Margin at a floor = test floor score of the model trained with that floor "
         "minus the best of the other four same-density treatments (nats per event; tie + non-tie contributions per scored "
         "event). The unbounded arm is the rank_change run (pass it with --base).", ""]
    summary = {}
    if not datasets:
        L.append("no component_bounds cells: pending")
    for ds in datasets:
        arms_all = [bound_arm(m, b) for b in BOUND_ARMS for m in X.BOUNDS_ARMS]
        common = [s for s in seeds if all((ds, "gru", x, lr, s) in cells for x in arms_all for lr in X.LRS)]
        L += [f"## {ds}", "", f"common seeds: {common} (of {list(seeds)})", ""]
        if len(common) < len(seeds):
            miss = [(x, lr, s) for x in arms_all for lr in X.LRS for s in seeds if (ds, "gru", x, lr, s) not in cells]
            L += [f"pending: {len(miss)} missing cells, e.g. {miss[:4]}", ""]
            summary[ds] = PENDING
            continue
        sel, _ = unit_select(cells, ds, "gru", arms_all, common)
        L += ["| bound | " + " | ".join(f"margin {f} (tie + non-tie)" for f in FLOOR_KEYS) +
              " | floor spread of the three floor-trained models | top by " + " / ".join(FLOOR_KEYS) + " / C | Kendall tau (floor 1e-6, C) | "
              "C change vs none (seed-paired mean, per treatment) |", "|---|---|---|---|---|---|---:|---|"]
        out = {}
        for bound in BOUND_ARMS:
            us = {m: sel[bound_arm(m, bound)] for m in X.BOUNDS_ARMS}
            mean = lambda m, key: float(np.mean([val(x, key) for x in us[m].values()]))
            tie = float(np.mean([x["test"]["tie_share"] for x in us["floor_1e-6"].values()]))
            cols, margins = [], {}
            for f in FLOOR_KEYS:
                own = mean(f, f)
                bo = max((m for m in X.BOUNDS_ARMS if m != f), key=lambda m: mean(m, f))
                mg = own - mean(bo, f)
                tie_c = tie * (mean(f, f + "_tie") - mean(bo, f + "_tie"))
                margins[f] = dict(total=mg, tie=tie_c, nontie=mg - tie_c, best_other=bo)
                cols.append(f"{mg:+.3f} ({tie_c:+.3f} + {mg - tie_c:+.3f})")
            spreads = [max(mean(f, g) for g in FLOOR_KEYS) - min(mean(f, g) for g in FLOOR_KEYS) for f in FLOOR_KEYS]
            tops = [X.BOUNDS_ARMS[int(np.argmax([mean(m, key) for m in X.BOUNDS_ARMS]))] for key in FLOOR_KEYS + ("C",)]
            tau = float(stats.kendalltau([mean(m, "floor_1e-6") for m in X.BOUNDS_ARMS], [mean(m, "C") for m in X.BOUNDS_ARMS])[0])
            cost = {} if bound == "none" else {m: float(np.mean(diff(sel, bound_arm(m, bound), m, "C", "C"))) for m in X.BOUNDS_ARMS}
            L.append(f"| {bound} | " + " | ".join(cols) + f" | {' / '.join(f'{x:.3f}' for x in spreads)} | {' / '.join(tops)} | "
                     f"{tau:+.2f} | {', '.join(f'{m} {v:+.4f}' for m, v in cost.items()) or '—'} |")
            out[bound] = dict(margins=margins, spreads=dict(zip(FLOOR_KEYS, spreads)), tops=dict(zip(FLOOR_KEYS + ("C",), tops)),
                              tau=tau, cost=cost, tie_share=tie)
        summary[ds] = out
        L.append("")
    return L, summary


def run_report(experiment, run, audit_path=None, base=None):
    audit = audit_rows(audit_path)
    run = Path(run)
    bases = [base] if isinstance(base, (str, Path)) else list(base or [])
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
        L, s = report_misspecification(run, audit, bases[0] if bases else None)
    elif experiment == "marks":
        L, s = report_marks(run, audit)
    elif experiment == "coarsened_real":
        L, s = report_coarsened(run, audit)
    elif experiment == "coarsened_more":
        L, s = report_coarsened(run, audit, datasets=X.COARSENED_MORE, arms=X.COARSEN_MORE_ARMS)
    elif experiment in ("triangular_coarsened", "triangular_jitter", "triangular_recorded"):
        L, s = report_triangular(run, bases)
    elif experiment == "component_bounds":
        L, s = report_component_bounds(run, bases)
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
    ap.add_argument("--base", nargs="*", default=None,
                    help="further run directories: delta_misspecification (the synthetic run); triangular_* (the runs whose "
                         "latent_scores.jsonl and result.json hold the other treatments); component_bounds (the rank_change run)")
    a = ap.parse_args(argv)
    L, _ = run_report(a.experiment, a.run, a.audit, a.base)
    print("\n".join(L))


if __name__ == "__main__":
    main()
