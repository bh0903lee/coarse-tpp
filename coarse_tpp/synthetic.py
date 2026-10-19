"""Synthetic processes with known latent times, exact true-law quantities, and their coarsened recordings.

Processes and parameters (Omi, Ueda and Aihara, NeurIPS 2019, Section 4.1; mean inter-event time 1):
  SP   stationary Poisson, lambda = 1
  NSP  non-stationary Poisson, lambda(t) = 0.99 sin(2 pi t / 20000) + 1
  SR   stationary renewal, log-normal gaps, sigma = sqrt(ln 37), mu = -sigma^2 / 2 (mean 1, sd 6)
  NSR  non-stationary renewal: gamma(shape 4, scale 0.25) renewal in the rescaled time
       R(t) = int_0^t r, r(t) = 0.99 sin(2 pi t / 20000) + 1
  SC   self-correcting, lambda(t) = exp(t - N(t))
  H1   Hawkes, mu = 0.2, lambda = mu + sum_j alpha beta exp(-beta (t - t_j)), (alpha, beta) = (0.8, 1)
  H2   Hawkes, mu = 0.2, kernels (0.4, 1) and (0.4, 20)

For every event i (history = all earlier events of the full simulated sequence, started empty at t = 0):
  oracle_lp[i]   true log density of t_i given the latent history
  Lam(i, x)      compensator from t_{i-1} to x >= t_{i-1}
Recording: t~ = delta floor((t + phi) / delta) - phi with one phase phi ~ U(0, delta) per (realization, delta);
stored as integer cell indices so that recorded gaps are exact multiples of delta.

Data sets: realizations r = 1..5 of N_TOTAL = 122,000 events (seed BASE_SEED + 1000 r + process index); the first
BURN = 2,000 are discarded (true-law quantities still use the full past). Blocks of 1,280 events are assigned
[train x 8, dev, test]; train windows are non-overlapping 128-event windows; dev / test windows have length 128
and stride 96, their first 32 positions are context only (score = 0), so every dev / test event is scored once with
at least 32 events of history. Name: synth_<PROC>_r<r>_d<delta code>, delta code '0' or e.g. '0p1' for 0.1.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

from . import data as D

PERIOD = 20000.0
LN_SIG = math.sqrt(math.log(37.0))
LN_MU = -LN_SIG ** 2 / 2
G_SHAPE, G_SCALE = 4.0, 0.25
HAWKES = {"H1": (0.2, ((0.8, 1.0),)), "H2": (0.2, ((0.4, 1.0), (0.4, 20.0)))}
PROCESSES = ("SP", "NSP", "SR", "NSR", "SC", "H1", "H2")
PROC_INDEX = {p: i for i, p in enumerate(PROCESSES)}
BASE_SEED = 20261007
N_TOTAL, BURN, BLOCK, WIN, STRIDE, CTX = 122_000, 2_000, 1_280, 128, 96, 32
DELTAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
REALIZATIONS = (1, 2, 3, 4, 5)
TIE_TABLE = Path(__file__).resolve().parent / "resources" / "synthetic_tie_table.json"


def _R(t):
    return t - 0.99 * PERIOD / (2 * math.pi) * (np.cos(2 * math.pi * t / PERIOD) - 1.0)


def _r(t):
    return 0.99 * np.sin(2 * math.pi * t / PERIOD) + 1.0


def _R_inv(y, t0):
    t = np.maximum(t0, y)                                 # R(t) ~ t
    for _ in range(100):
        f = _R(t) - y
        if abs(f) < 1e-12 * max(1.0, abs(y)):
            break
        t = max(t0, t - f / max(_r(t), 1e-6))
    return t


# ------------------------------------------------------------------ simulation
def simulate(proc: str, n: int, rng: np.random.Generator):
    """Latent event times (float64, increasing) and duplicate flags (all False)."""
    dup = np.zeros(n, dtype=bool)
    if proc == "SP":
        return np.cumsum(rng.exponential(1.0, n)), dup
    if proc == "SR":
        return np.cumsum(rng.lognormal(LN_MU, LN_SIG, n)), dup
    if proc == "NSR":
        y = np.cumsum(rng.gamma(G_SHAPE, G_SCALE, n))
        t = np.empty(n); prev = 0.0
        for i in range(n):
            prev = t[i] = _R_inv(y[i], prev)
        return t, dup
    if proc == "SC":
        t = np.empty(n); prev = 0.0
        e = rng.exponential(1.0, n)
        for i in range(n):                                # Lambda from t_{i-1}: exp(x - i) - exp(t_{i-1} - i) = E
            prev = t[i] = i + math.log(math.exp(prev - i) + e[i])
        return t, dup
    if proc == "NSP":
        out, t, lmax = [], 0.0, 1.99
        while len(out) < n:
            t += rng.exponential(1.0 / lmax)
            if rng.random() * lmax <= 0.99 * math.sin(2 * math.pi * t / PERIOD) + 1.0:
                out.append(t)
        return np.asarray(out), dup
    if proc in HAWKES:
        mu, ker = HAWKES[proc]
        A = [0.0] * len(ker)
        out, t = [], 0.0
        while len(out) < n:
            lam_bar = mu + sum(a * b * Ak for (a, b), Ak in zip(ker, A))
            w = rng.exponential(1.0 / lam_bar)
            t += w
            A = [Ak * math.exp(-b * w) for (a, b), Ak in zip(ker, A)]
            if rng.random() * lam_bar <= mu + sum(a * b * Ak for (a, b), Ak in zip(ker, A)):
                out.append(t)
                A = [Ak + 1.0 for Ak in A]
        return np.asarray(out), np.zeros(len(out), dtype=bool)
    raise ValueError(proc)


# ------------------------------------------------------------------ true law
class Oracle:
    """Exact conditional quantities of the true process along one simulated sequence."""

    def __init__(self, proc: str, t: np.ndarray, dup: np.ndarray | None = None):
        self.proc, self.t = proc, t
        self.dup = np.zeros(len(t), dtype=bool) if dup is None else dup
        self.prev = np.concatenate([[0.0], t[:-1]])          # t_{i-1} (0 for the first event)
        if proc in HAWKES:
            mu, ker = HAWKES[proc]
            beta = np.array([b for _, b in ker])
            # Ap[k, i] = sum over events j < i of exp(-beta_k (t_{i-1} - t_j)): the kernel state at t_{i-1}
            self.Ap = np.zeros((len(ker), len(t)))
            A = np.zeros(len(ker))
            for i in range(len(t)):
                self.Ap[:, i] = A
                A = A * np.exp(-beta * (t[i] - self.prev[i])) + 1.0

    def Lam(self, x):
        """Compensator from t_{i-1} to x_i, elementwise, x >= t_{i-1}."""
        p, t = self.prev, self.t
        d = x - p
        if self.proc == "SP":
            return d
        if self.proc == "NSP":
            return d - 0.99 * PERIOD / (2 * math.pi) * (np.cos(2 * math.pi * x / PERIOD) - np.cos(2 * math.pi * p / PERIOD))
        if self.proc == "SR":
            return -stats.lognorm.logsf(np.maximum(d, 0), LN_SIG, scale=math.exp(LN_MU))
        if self.proc == "NSR":
            return -stats.gamma.logsf(_R(x) - _R(p), G_SHAPE, scale=G_SCALE)
        if self.proc == "SC":
            i = np.arange(len(t))
            return np.exp(x - i) - np.exp(p - i)
        mu, ker = HAWKES[self.proc]
        return mu * d + sum(a * self.Ap[k] * (-np.expm1(-b * d)) for k, (a, b) in enumerate(ker))

    def log_lambda_at_t(self):
        t, p = self.t, self.prev
        if self.proc == "SP":
            return np.zeros(len(t))
        if self.proc == "NSP":
            return np.log(_r(t))
        if self.proc == "SC":
            return t - np.arange(len(t))
        mu, ker = HAWKES[self.proc]
        return np.log(mu + sum(a * b * self.Ap[k] * np.exp(-b * (t - p)) for k, (a, b) in enumerate(ker)))

    def log_density(self):
        """True log density of every t_i given its latent past."""
        t, p = self.t, self.prev
        if self.proc == "SR":
            return stats.lognorm.logpdf(t - p, LN_SIG, scale=math.exp(LN_MU))
        if self.proc == "NSR":
            return stats.gamma.logpdf(_R(t) - _R(p), G_SHAPE, scale=G_SCALE) + np.log(_r(t))
        return self.log_lambda_at_t() - self.Lam(t)

    def increments(self):
        """Time-rescaling increments Lambda(t_{i-1}, t_i) (Exp(1) under the truth)."""
        return self.Lam(self.t)

    def log_bin_prob(self, a, b):
        """log P(t_i in [a_i, b_i) | latent past), a_i <= t_i < b_i; the part of the cell before t_{i-1} has
        probability 0."""
        lo = np.maximum(a, self.prev)
        La, Lb = self.Lam(lo), self.Lam(b)
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = -La + np.log(-np.expm1(-(Lb - La)))
        return np.where(np.isinf(La), -np.inf, lp)                # cell entirely in the far tail: probability 0


def coarse_bins(t: np.ndarray, delta: float, phi: float) -> np.ndarray:
    """Integer cell index k with recorded time delta * k - phi (floor recording with phase phi)."""
    return np.floor((t + phi) / delta).astype(np.int64)


# ------------------------------------------------------------------ data sets
def synth_dir() -> Path:
    return D.data_dir() / "synthetic"


def dcode(d: float) -> str:
    return "0" if d == 0 else repr(float(d)).replace(".", "p")


def name(proc, r, d):
    return f"synth_{proc}_r{r}_d{dcode(d)}"


def parse(ds: str):
    m = re.fullmatch(r"synth_([A-Z0-9]+)_r(\d+)_d([0-9p]+)", ds)
    if not m:
        return None
    return m.group(1), int(m.group(2)), float(m.group(3).replace("p", "."))


def gen_seed(proc, r):
    return BASE_SEED + 1000 * r + PROC_INDEX[proc]


def generate(proc: str, r: int, out_dir: Path | None = None, n_total: int = N_TOTAL):
    """Simulate one realization and write <out_dir>/<proc>_r<r>.npz (latent times, true-law quantities and the
    phase and true cell log-probabilities of every delta)."""
    out_dir = Path(out_dir or synth_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(gen_seed(proc, r))
    t, dup = simulate(proc, n_total, rng)
    o = Oracle(proc, t, dup)
    arrays = dict(t=t, dup=dup, oracle_lp=o.log_density(), incr=o.increments())
    for j, d in enumerate(DELTAS):
        if d == 0:
            continue
        phi = np.random.default_rng(BASE_SEED + 7 + 1000 * r + 100 * j).uniform(0, d)
        k = coarse_bins(t, d, phi)
        a, b = d * k - phi, d * (k + 1) - phi
        arrays[f"phi_{dcode(d)}"] = np.array(phi)
        arrays[f"oracle_C_{dcode(d)}"] = o.log_bin_prob(a, b)
    np.savez_compressed(out_dir / f"{proc}_r{r}.npz", **arrays)
    return arrays


_CACHE = {}


def arrays(proc, r):
    key = (str(synth_dir()), proc, r)
    if key not in _CACHE:
        with np.load(synth_dir() / f"{proc}_r{r}.npz") as z:
            _CACHE[key] = {k: z[k] for k in z.files}
    return _CACHE[key]


def event_arrays(proc, r, d):
    """Full-sequence per-event arrays (index i over all N_TOTAL events)."""
    z = arrays(proc, r)
    t = z["t"]
    prev = np.concatenate([[0.0], t[:-1]])
    out = dict(lat_gap=t - prev, oracle_lp=z["oracle_lp"], dup=z["dup"])
    if d == 0:
        out.update(gap=t - prev, s=t - prev, oracle_C=z["oracle_lp"])
    else:
        phi = float(z[f"phi_{dcode(d)}"])
        k = coarse_bins(t, d, phi)
        kp = np.concatenate([[coarse_bins(np.array([0.0]), d, phi)[0]], k[:-1]])
        out.update(gap=d * (k - kp).astype(np.float64), s=t - (d * kp - phi), oracle_C=z[f"oracle_C_{dcode(d)}"])
    return out


def windows(split: str, n_total: int = N_TOTAL):
    """(start, end, first_scored) event-index triples of the windows of `split`."""
    n = n_total - BURN
    out = []
    for bi, b0 in enumerate(range(0, n - BLOCK + 1, BLOCK)):
        kind = "train" if bi % 10 < 8 else ("dev" if bi % 10 == 8 else "test")
        if kind != split:
            continue
        lo, hi = BURN + b0, BURN + b0 + BLOCK
        if split == "train":
            out += [(s, s + WIN, s + 1) for s in range(lo, hi, WIN)]
        else:
            s = lo - CTX
            while s + CTX < hi:
                out.append((s, min(s + WIN, hi), s + CTX))
                s += STRIDE
    return out


def load_views(ds: str, split: str):
    proc, r, d = parse(ds)
    ev = event_arrays(proc, r, d)
    seqs = windows(split, len(ev["gap"]))
    sl = lambda k: [ev[k][a:b].astype(np.float64) for a, b, _ in seqs]
    gaps = sl("gap")
    times = [np.cumsum(g) for g in gaps]
    marks = [np.zeros(b - a, dtype=np.int64) for a, b, _ in seqs]
    score = [(np.arange(a, b) >= f).astype(np.float64) for a, b, f in seqs]
    coarse = D.Split(gaps, marks, times, 1)
    lat = D.Split(sl("lat_gap"), marks, [np.cumsum(g) for g in sl("lat_gap")], 1)
    return D.Views(coarse, lat, sl("oracle_lp"), lat_s=sl("s"), oracle_C=sl("oracle_C"), score=score, delta=d)


def tie_table():
    rows = []
    for proc in PROCESSES:
        for r in REALIZATIONS:
            for d in DELTAS:
                v = load_views(name(proc, r, d), "train")
                g = np.concatenate([x[1:] for x in v.coarse.iet])
                rows.append(dict(process=proc, r=r, delta=d, tie_share=float(np.mean(g == 0)), n=int(len(g))))
    return rows


def main_settings():
    """(process, delta) settings of the pre-specified comparison: SP, SR, SC, H1, H2 with a train tie share,
    averaged over the five realizations, in [0.10, 0.50]."""
    rows = json.loads(TIE_TABLE.read_text())
    acc = {}
    for x in rows:
        acc.setdefault((x["process"], x["delta"]), []).append(x["tie_share"])
    judged = ("SP", "SR", "SC", "H1", "H2")
    return sorted(k for k, v in acc.items() if k[0] in judged and k[1] > 0 and 0.10 <= np.mean(v) <= 0.50)


def ensure(processes=PROCESSES, realizations=REALIZATIONS, check: bool = True):
    """Generate any missing realization and compare the train tie shares with the reference table."""
    for proc in processes:
        for r in realizations:
            if not (synth_dir() / f"{proc}_r{r}.npz").exists():
                print(f"generating {proc} r{r}", flush=True)
                generate(proc, r)
    if not check:
        return
    ref = {(x["process"], x["r"], x["delta"]): x for x in json.loads(TIE_TABLE.read_text())}
    bad = []
    for proc in processes:
        for r in realizations:
            for d in DELTAS:
                v = load_views(name(proc, r, d), "train")
                g = np.concatenate([x[1:] for x in v.coarse.iet])
                x = ref[(proc, r, d)]
                if float(np.mean(g == 0)) != x["tie_share"] or int(len(g)) != x["n"]:
                    bad.append((proc, r, d))
    if bad:
        raise SystemExit(f"generated data differ from the reference tie table: {bad[:5]} (a realization generated "
                         "with --n-total? keep it in a separate data directory)")
    print(f"synthetic data ok: {len(processes) * len(realizations) * len(DELTAS)} (process, realization, delta) rows "
          "match the reference tie table")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Generate the synthetic realizations into DATA_DIR/synthetic.")
    ap.add_argument("--processes", nargs="*", default=list(PROCESSES))
    ap.add_argument("--realizations", nargs="*", type=int, default=list(REALIZATIONS))
    ap.add_argument("--n-total", type=int, default=N_TOTAL, help="events per realization (smaller for a smoke test)")
    ap.add_argument("--data", default=None)
    a = ap.parse_args(argv)
    if a.data:
        D.set_data_dir(a.data)
    if a.n_total != N_TOTAL:
        for proc in a.processes:
            for r in a.realizations:
                f = synth_dir() / f"{proc}_r{r}.npz"
                if f.exists():
                    with np.load(f) as z:
                        n_old = len(z["t"])
                    if n_old != a.n_total:     # never replace a realization of another size
                        raise SystemExit(f"{f} holds {n_old} events; write a smaller realization to a separate data "
                                         "directory (--data or COARSE_TPP_DATA)")
                generate(proc, r, n_total=a.n_total)
                print(f"generated {proc} r{r} ({a.n_total} events)")
        return
    ensure(tuple(a.processes), tuple(a.realizations), check=True)


if __name__ == "__main__":
    main()
