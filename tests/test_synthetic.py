"""Synthetic processes: windows, recorded views, true cell probabilities."""
import numpy as np
import pytest

from coarse_tpp import data as D, synthetic as S


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    old = D.data_dir()
    D.set_data_dir(tmp)
    S.generate("H1", 1, n_total=15_000)
    yield tmp
    D.set_data_dir(old)


def test_windows_score_every_dev_test_event_once():
    n_total = 30_000
    nb = (n_total - S.BURN) // S.BLOCK
    kinds = ["train" if b % 10 < 8 else "dev" if b % 10 == 8 else "test" for b in range(nb)]
    for split in ("dev", "test"):
        seen = []
        for a, b, f in S.windows(split, n_total):
            assert f - a in (0, S.CTX) and b - a <= S.WIN
            seen += list(range(f, b))
        assert len(seen) == len(set(seen)) == kinds.count(split) * S.BLOCK
    tr = S.windows("train", n_total)
    assert all(b - a == S.WIN for a, b, _ in tr) and len(tr) * S.WIN == kinds.count("train") * S.BLOCK


def test_views_consistency(synth):
    v = S.load_views("synth_H1_r1_d0p1", "test")
    for i in range(3):
        g, s, lat = v.coarse.iet[i], v.lat_s[i], v.latent.iet[i]
        assert np.allclose(np.round(g / 0.1), g / 0.1, atol=1e-9)
        assert np.all(s[1:] >= 0) and np.all(s[1:] < g[1:] + 0.1 + 1e-9) and np.all(s[1:] >= g[1:] - 1e-9)
        assert np.all(lat[1:] <= s[1:] + 1e-12)
    assert all(np.all(c <= 1e-12) for c in v.oracle_C)
    v0 = S.load_views("synth_H1_r1_d0", "test")
    assert np.allclose(v0.coarse.iet[0], v0.latent.iet[0])


def test_true_cell_probabilities_sum_to_one():
    rng = np.random.default_rng(3)
    for proc in ("SP", "SR", "SC", "H2", "NSR"):
        t, dup = S.simulate(proc, 400, rng)
        o = S.Oracle(proc, t, dup)
        i, d = 200, 0.05
        edges = o.prev[i] + d * np.arange(0, 4000)
        with np.errstate(all="ignore"):
            p = [np.exp(o.log_bin_prob(np.full(len(t), lo), np.full(len(t), hi))[i]) for lo, hi in zip(edges[:-1], edges[1:])]
        assert abs(sum(p) - 1) < 2e-3, (proc, sum(p))


def test_main_settings_of_the_comparison():
    m = S.main_settings()
    assert len(m) == 9 and ("SC", 1.0) in m and all(p in ("SP", "SR", "SC", "H1", "H2") for p, _ in m)


def test_small_realization_does_not_replace_another_size(tmp_path):
    old = D.data_dir()
    try:
        D.set_data_dir(tmp_path)
        S.generate("SP", 1, n_total=4_000)
        with pytest.raises(SystemExit, match="4000 events"):
            S.main(["--processes", "SP", "--realizations", "1", "--n-total", "5000"])
        S.main(["--processes", "SP", "--realizations", "1", "--n-total", "4000"])   # same size: regenerated
    finally:
        D.set_data_dir(old)
