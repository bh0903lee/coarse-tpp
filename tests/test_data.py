"""Data loading, stored ties, jitter and standardization."""
import pickle

import numpy as np
import pytest
import torch

from coarse_tpp import data as D


def split(iets, marks=None):
    marks = marks or [np.zeros(len(x), int) for x in iets]
    return D.Split([np.asarray(x, float) for x in iets], [np.asarray(m) for m in marks], [np.cumsum(x) for x in iets], 3)


def write(tmp, name, seqs_by_split, dim=1):
    (tmp / name).mkdir(parents=True, exist_ok=True)
    for sp, seqs in seqs_by_split.items():
        pickle.dump({sp: seqs, "dim_process": dim}, open(tmp / name / f"{sp}.pkl", "wb"))


def ev(gaps, types=None):
    t = np.cumsum(gaps)
    types = types if types is not None else [0] * len(gaps)
    return [dict(time_since_start=float(a), time_since_last_event=float(b), type_event=int(c)) for a, b, c in zip(t, gaps, types)]


@pytest.fixture
def datadir(tmp_path):
    old = D.data_dir()
    D.set_data_dir(tmp_path)
    yield tmp_path
    D.set_data_dir(old)


def test_stored_ties_are_zero_but_process_logs_keep_millisecond_gaps(datadir):
    seq = ev([0.0, 1e-6, 2.7e-7, 0.5])
    write(datadir, "toy", {sp: [seq] for sp in ("train", "dev", "test")})
    write(datadir, "bpi_toy", {sp: [seq] for sp in ("train", "dev", "test")})
    assert np.array_equal(D.load("toy", "train").coarse.iet[0], [0.0, 0.0, 0.0, 0.5])
    assert np.array_equal(D.load("bpi_toy", "train").coarse.iet[0], [0.0, 1e-6, 2.7e-7, 0.5])
    assert np.array_equal(D.load("toy", "train", raw=True).coarse.iet[0], [0.0, 1e-6, 2.7e-7, 0.5])


def test_marks_one_based_split_and_range_check(datadir):
    seqs = [ev([0.1, 0.2, 0.3], [1, 2, 3])]
    write(datadir, "retweet_sub", {"test": seqs}, dim=3)
    assert D.load("retweet_sub", "test").coarse.marks[0].tolist() == [0, 1, 2]
    write(datadir, "other", {"test": seqs}, dim=3)
    with pytest.raises(ValueError):
        D.load("other", "test")
    write(datadir, "single", {"test": [ev([0.1, 0.2], [1, 1])]}, dim=1)
    assert D.load("single", "test").coarse.marks[0].tolist() == [0, 0]


def test_jitter_shapes_and_support():
    sp = split([[0.5, 0.0, 0.3, 0.0, 1.2], [0.1, 0.6, 0.0]])
    b = D.make_batch(sp, np.arange(2), "cpu")
    g = torch.Generator().manual_seed(0)
    for kind in ("jitter_time", "jitter_time_gauss"):
        j = D.jitter_times(b, 0.3, kind, g)
        assert torch.all(j.iet >= 0) and torch.equal(j.iet[:, 0], b.iet[:, 0])
        assert torch.all(j.iet[:, 1:] * (1 - b.tmask) == 0)
    t = D.jitter_targets_uniform(b.iet[:, 1:], 0.3, g)
    assert torch.all(t >= b.iet[:, 1:].double()) and torch.all(t < b.iet[:, 1:].double() + 0.3)


def test_jitter_moves_marks_with_their_events():
    sp = split([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[0, 1, 2, 0, 1, 2]])
    b = D.make_batch(sp, np.arange(1), "cpu")
    j = D.jitter_times(b, 1.0, "jitter_time", torch.Generator().manual_seed(5))
    assert sorted(j.marks[0].tolist()) == sorted(b.marks[0].tolist())


def test_multi_sample_jitter_with_one_sample_is_jitter_gap():
    tau = torch.rand(4, 7, dtype=torch.float64)
    a = D.jitter_targets_uniform(tau, 0.2, torch.Generator().manual_seed(3))
    b = D.jitter_targets_uniform_multi(tau, 0.2, 1, torch.Generator().manual_seed(3))[0]
    assert torch.equal(a, b)


def test_coarse_stats_rule():
    st = D.coarse_stats(D.Views(split([[1.0, 0.0, 0.2, 0.4]])), 0.2)
    assert abs(st.log_mean - np.mean(np.log(np.array([0.0, 0.2, 0.4]) + 0.1))) < 1e-12
    st0 = D.coarse_stats(D.Views(split([[1.0, 0.0, 0.2, 0.4]])), 0.0)
    assert abs(st0.log_mean - np.mean(np.log(np.array([1e-6, 0.2, 0.4])))) < 1e-12
