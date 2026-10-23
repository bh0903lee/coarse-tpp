"""Tiny data sets and training helpers shared by the tests (CPU, seconds)."""
import pickle

import numpy as np

from coarse_tpp import experiments as X, train as TR

OPT = dict(batch_size=32, max_epochs=2, patience=20, grad_clip=1.0)


def tied_toy(root, name="tied_toy", n_types=1, bpi=False, seed=3):
    """A small data set with about a third of its gaps tied (zero), integer gaps otherwise (delta = 1). Process-log
    style files (bpi=True) store exact zeros, the others the 1e-6 convention."""
    rng = np.random.default_rng(seed)
    for sp, n in (("train", 30), ("dev", 8), ("test", 8)):
        seqs = []
        for _ in range(n):
            g = np.where(rng.random(25) < 0.35, 0.0, np.round(rng.exponential(3.0, 25)) + 1); g[0] = 0.0
            stored = g if bpi else np.maximum(g, 1e-6)
            seqs.append([dict(time_since_start=float(t), time_since_last_event=float(x), type_event=int(k))
                         for t, x, k in zip(np.cumsum(g), stored, rng.integers(0, n_types, 25))])
        (root / name).mkdir(exist_ok=True)
        pickle.dump({sp: seqs, "dim_process": n_types}, open(root / name / f"{sp}.pkl", "wb"))
    return name


def train_cells(experiment, run, ds, arms, delta, lrs=("lr1e3",), seeds=(0,), max_epochs=2, extra=None, **kw):
    """Train tiny cells of `experiment` on the CPU and write the grid of `run`; returns the configs."""
    cs = []
    for a in arms:
        for lr in lrs:
            for s in seeds:
                c = X.cell(experiment, run, ds, a, s, lr, delta, **kw)
                c.update(optim=dict(OPT, max_epochs=max_epochs, lr=X.LRS[lr]), cpu=True)
                if extra:
                    c.update(extra)
                TR.run(c)
                cs.append(c)
    X.write(experiment, run, cells=cs)
    return cs
