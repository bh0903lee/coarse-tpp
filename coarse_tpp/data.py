"""Event-sequence data: loading, batching, jitter and standardization.

A data set directory DATA_DIR/<name>/{train,dev,test}.pkl holds {split: sequences, "dim_process": n_types}, where
a sequence is a list (or dict) of events {"time_since_start", "time_since_last_event", "type_event"}. iet[j] is the
recorded gap of event j; the state at position j uses iet[0..j] (and marks[0..j]) and predicts iet[j+1]. The first
gap iet[0] is an input only and never a target. Synthetic data sets (names synth_*) are built from
DATA_DIR/synthetic/*.npz by coarse_tpp.synthetic and carry latent gaps, s_i = t_i - t~_{i-1}, true-law quantities
and a score mask, aligned one to one with the recorded events.

Stored ties: the NHP / EasyTPP-style files store a zero gap as 1e-6. For them a recorded gap <= TIE_STORED =
1.000001e-6 is read back as an exact zero. The threshold is a reading convention: a genuinely positive gap below it
is also read as a tie (in the paper's data, one gap in ContTime and two in Earthquake). Business process logs (names bpi_*, unit hours) store exact zeros and
genuine millisecond gaps (1 ms = 2.8e-7 h), so the mapping is not applied to them (`stores_floor`).

Jitter: jitter_gap adds U(0, delta) to the target gaps only; jitter_time / jitter_time_gauss perturb the recorded
times of the window (inputs and targets), re-sort them and recompute the gaps.

DATA_DIR is the environment variable COARSE_TPP_DATA, or ./data of the repository.
"""
from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("COARSE_TPP_DATA", ROOT / "data"))
TIE_STORED = 1.000001e-6


def set_data_dir(path) -> None:
    global DATA_DIR
    DATA_DIR = Path(path)


def data_dir() -> Path:
    return DATA_DIR


# ------------------------------------------------------------------ containers and batches
@dataclass
class Split:
    iet: list            # float64 arrays, one per sequence (recorded gaps)
    marks: list          # int64 arrays (0-based types)
    times: list          # float64 arrays (time_since_start)
    n_types: int

    def __len__(self):
        return len(self.iet)

    @property
    def n_targets(self) -> int:
        return int(sum(len(x) - 1 for x in self.iet))


@dataclass
class Stats:
    log_mean: float
    log_std: float
    scale: float                      # exp(log_mean)
    max_train_iet: float
    median: float = float("nan")      # quantiles of the train targets
    q99: float = float("nan")
    q001: float = float("nan")
    q999: float = float("nan")


@dataclass
class Batch:
    iet: torch.Tensor       # (B, L) gaps, 0 on padding
    times: torch.Tensor     # (B, L) time_since_start, 0 on padding
    marks: torch.Tensor     # (B, L) 0-based types, 0 on padding
    mask: torch.Tensor      # (B, L) 1 on real events
    tmask: torch.Tensor     # (B, L-1) 1 where position j has a real target j+1
    lengths: torch.Tensor   # (B,)
    idx: np.ndarray         # sequence indices in the split


def make_batch(split: Split, idx, device) -> Batch:
    idx = np.asarray(idx)
    lens = [len(split.iet[i]) for i in idx]
    L = max(lens)
    iet = np.zeros((len(idx), L)); mk = np.zeros((len(idx), L), dtype=np.int64)
    mask = np.zeros((len(idx), L)); tm = np.zeros((len(idx), L))
    for b, i in enumerate(idx):
        n = lens[b]
        iet[b, :n] = split.iet[i]; mk[b, :n] = split.marks[i]; mask[b, :n] = 1
        tm[b, :n] = split.times[i]
    iet_t = torch.tensor(iet, dtype=torch.float32, device=device)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
    tmask = mask_t[:, 1:] * mask_t[:, :-1]
    return Batch(iet_t, torch.tensor(tm, dtype=torch.float32, device=device),
                 torch.tensor(mk, device=device), mask_t, tmask,
                 torch.tensor(lens, device=device), idx)


def iterate(split: Split, batch_size: int, device, shuffle: bool, rng: np.random.Generator | None = None):
    order = np.arange(len(split))
    if shuffle:
        (rng or np.random.default_rng()).shuffle(order)
    for i in range(0, len(order), batch_size):
        yield make_batch(split, order[i:i + batch_size], device)


# ------------------------------------------------------------------ loading
def stores_floor(ds: str) -> bool:
    """True when the data set stores ties as 1e-6 (every non-process-log data set); process logs store exact zeros."""
    return not ds.startswith("bpi_")


def _gaps(s, floor: bool = True):
    g = np.asarray([e["time_since_last_event"] for e in s], dtype=np.float64)
    return np.where(g <= TIE_STORED, 0.0, g) if floor else g


@dataclass
class Views:
    coarse: Split
    latent: Split | None = None     # same sequences, latent gaps (synthetic only)
    oracle_lp: list | None = None   # per sequence: true log density of t_i given the latent past
    lat_s: list | None = None       # per sequence: s_i = t_i - t~_{i-1}
    oracle_C: list | None = None    # per sequence: true log probability of the recorded cell of t_i
    score: list | None = None       # per sequence: 1 where the event is scored (dev / test context = 0)
    delta: float | None = None


def score_mask(views: Views, idx, L: int, device):
    """(B, L-1) target mask factor from `score` (ones when the data set has no context positions)."""
    out = np.ones((len(idx), L - 1))
    if views.score is not None:
        for b, i in enumerate(idx):
            sc = views.score[i][1:]
            out[b, :len(sc)] = sc
    return torch.tensor(out, dtype=torch.float32, device=device)


def per_target(views: Views, key: str, idx, L: int, device):
    """(B, L-1) float64 tensor of a per-event array (targets 1..n-1), 0 on padding."""
    arrs = getattr(views, key)
    out = np.zeros((len(idx), L - 1))
    for b, i in enumerate(idx):
        a = arrs[i][1:]
        out[b, :len(a)] = a
    return torch.tensor(out, dtype=torch.float64, device=device)


def _read_split(ds: str, split: str, min_len: int):
    p = DATA_DIR / ds / f"{split}.pkl"
    with open(p, "rb") as fh:
        d = pickle.load(fh, encoding="latin-1")
    inner = d[split]
    seqs = [s for s in (list(inner.values()) if isinstance(inner, dict) else list(inner)) if len(s) >= min_len]
    return seqs, int(d.get("dim_process", 1))


def load(ds: str, split: str, min_len: int = 2, raw: bool = False) -> Views:
    """Views of a split; raw=True keeps the stored gaps (stored ties stay 1e-6, also as encoder inputs)."""
    if ds.startswith("synth_"):
        from .synthetic import load_views
        return load_views(ds, split)
    seqs, n_types = _read_split(ds, split, min_len)
    arr = lambda s, k, dt=np.float64: np.asarray([e[k] for e in s], dtype=dt)
    fl = stores_floor(ds) and not raw
    marks = _check_marks([arr(s, "type_event", np.int64) for s in seqs], n_types, ds, split)
    coarse = Split([_gaps(s, fl) for s in seqs], marks, [arr(s, "time_since_start") for s in seqs], n_types)
    return Views(coarse)


# Two distributed files store their types 1-based (test splits of the Retweet and StackOverflow subsamples,
# while their train / dev splits are 0-based). Only these splits are shifted, and only when their range is
# exactly 1..n_types; any other out-of-range type is an error. Single-type data (dim_process = 1) get type 0.
ONE_BASED_SPLITS = {("retweet_sub", "test"), ("stackoverflow_sub", "test")}


def _check_marks(marks: list, n_types: int, ds: str, split: str) -> list:
    if not marks:
        return marks
    if n_types == 1:
        return [np.zeros_like(m) for m in marks]
    lo, hi = min(int(m.min()) for m in marks if len(m)), max(int(m.max()) for m in marks if len(m))
    if (ds, split) in ONE_BASED_SPLITS and lo >= 1 and hi == n_types:
        print(f"[coarse_tpp.data] {ds}/{split}: types stored 1-based ({lo}..{hi}); shifted to 0-based", file=sys.stderr)
        marks = [m - 1 for m in marks]
        lo, hi = lo - 1, hi - 1
    if lo < 0 or hi >= n_types:
        raise ValueError(f"{ds}/{split}: type_event range {lo}..{hi} outside [0, {n_types})")
    return marks


def load_fine_s(ds: str, split: str, min_len: int = 2) -> list:
    """Coarsened data sets (coarse_tpp.prepare.coarsen keeps the original time per event as `time_since_start_fine`):
    per sequence, the latent time since the last recorded time, s_i = t_fine_i - t~_{i-1} (s_0 = 0, no target),
    aligned with load(ds, split).coarse.iet."""
    seqs, _ = _read_split(ds, split, min_len)
    out = []
    for s in seqs:
        q = np.asarray([e["time_since_start"] for e in s], dtype=np.float64)          # recorded (floored) times
        tf = np.asarray([e["time_since_start_fine"] for e in s], dtype=np.float64)
        v = np.zeros(len(s))
        v[1:] = tf[1:] - q[:-1]
        out.append(v)
    return out


# ------------------------------------------------------------------ jitter and standardization
def jitter_times(batch: Batch, delta: float, kind: str, gen: torch.Generator) -> Batch:
    """A copy of `batch` with jittered, re-sorted event times (inputs and targets change).
      jitter_time         u ~ U(0, delta)                time-level uniform dequantization of the recorded times
      jitter_time_gauss   u ~ N(0, (delta / sqrt 12)^2)  Gaussian tie-breaking noise (same variance as U(0, delta))
    Times are relative to the window start (float64); the first gap iet[0] (input only) is kept. Marks follow their
    events through the re-sort."""
    iet = batch.iet.double()
    m = batch.mask.double()
    t = torch.cumsum(torch.cat([torch.zeros_like(iet[:, :1]), iet[:, 1:]], dim=1), dim=1)
    if kind == "jitter_time":
        u = torch.rand(iet.shape, generator=gen, dtype=torch.float64) * delta
    elif kind == "jitter_time_gauss":
        u = torch.randn(iet.shape, generator=gen, dtype=torch.float64) * (delta / 12 ** 0.5)
    else:
        raise ValueError(kind)
    tj = torch.where(m > 0, t + u.to(iet.device), torch.full_like(t, float("inf")))
    tj, perm = torch.sort(tj, dim=1, stable=True)
    marks = batch.marks if batch.marks is None else torch.gather(batch.marks, 1, perm.to(batch.marks.device))
    g = torch.nan_to_num(torch.diff(tj, dim=1), nan=0.0, posinf=0.0)
    g = torch.where(batch.tmask > 0, g, torch.zeros_like(g)).clamp(min=0.0)
    new = torch.cat([iet[:, :1], g], dim=1).to(batch.iet.dtype) * batch.mask
    return Batch(new, batch.times, marks, batch.mask, batch.tmask, batch.lengths, batch.idx)


def jitter_targets_uniform(tau: torch.Tensor, delta: float, gen: torch.Generator) -> torch.Tensor:
    """jitter_gap: gap-level uniform dequantization of the targets only, tau~ + U(0, delta) (inputs unchanged)."""
    u = torch.rand(tau.shape, generator=gen, dtype=torch.float64).to(tau.device)
    return tau.double() + delta * u


def jitter_targets_uniform_multi(tau: torch.Tensor, delta: float, S: int, gen: torch.Generator) -> torch.Tensor:
    """iw_jitter: S independent uniform dequantizations of the targets, shape (S, *tau.shape).
    S = 1 consumes the generator exactly as jitter_targets_uniform does (same draw)."""
    u = torch.rand((int(S),) + tuple(tau.shape), generator=gen, dtype=torch.float64).to(tau.device)
    return tau.double().unsqueeze(0) + delta * u


def coarse_stats(views: Views, delta: float) -> Stats:
    """Standardization statistics shared by every treatment: log(tau~ + delta / 2) of the train targets when
    delta > 0, log(max(tau, 1e-6)) when delta = 0."""
    tg = np.concatenate([x[1:] for x in views.coarse.iet])
    v = tg + 0.5 * delta if delta > 0 else np.maximum(tg, 1e-6)
    lt = np.log(v)
    m, s = float(lt.mean()), float(lt.std())
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    return Stats(log_mean=m, log_std=s, scale=float(np.exp(m)), max_train_iet=float(tg.max()),
                 median=float(np.median(v)), q99=float(np.quantile(v, 0.99)), q001=float(np.quantile(v, 0.001)),
                 q999=float(np.quantile(v, 0.999)))
