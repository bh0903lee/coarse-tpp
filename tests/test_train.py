"""End-to-end training of tiny cells on CPU, the runner, and the objectives of the treatments."""
import json
import math
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from coarse_tpp import data as D, experiments as X, lik, readout as Ro, results as R, synthetic as S, train as TR

ROOT = Path(__file__).resolve().parents[1]
OPT = dict(batch_size=32, max_epochs=2, patience=20, grad_clip=1.0)


@pytest.fixture(scope="module")
def datadir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    old = D.data_dir()
    D.set_data_dir(tmp)
    S.generate("SC", 1, n_total=15_000)
    rng = np.random.default_rng(0)
    for sp, n in (("train", 40), ("dev", 10), ("test", 10)):
        seqs = []
        for _ in range(n):
            g = np.round(rng.exponential(2.0, 30)); g[0] = 0.0
            seqs.append([dict(time_since_start=float(t), time_since_last_event=float(max(x, 1e-6)), type_event=int(k))
                         for t, x, k in zip(np.cumsum(g), g, rng.integers(0, 3, 30))])
        (tmp / "toy").mkdir(exist_ok=True)
        pickle.dump({sp: seqs, "dim_process": 3}, open(tmp / "toy" / f"{sp}.pkl", "wb"))
    yield tmp
    D.set_data_dir(old)


def cfg_of(tmp, ds, arm, delta, **kw):
    c = X.cell("test", tmp / "run", ds, arm, 0, "lr1e3", delta, **kw)
    c.update(optim=dict(OPT, lr=1e-3), cpu=True)
    return c


@pytest.mark.parametrize("arm", ["interval", "point_mass", "jitter_time", "triangular", "iw_jitter_S2",
                                 "interval_jitter_mix_0.5", "floor_1e-10"])
def test_synthetic_cell_trains_and_scores(datadir, tmp_path, arm):
    c = cfg_of(tmp_path, "synth_SC_r1_d1p0", arm, 1.0)
    res = TR.run(c)
    assert res["status"] == "ok" and set(res["by_selection"]) == {"C", "own", "B"}
    t = res["test"]
    for k in ("C", "B", "oracle_B", "floor_1e-6"):
        assert math.isfinite(t[k])
    assert t["n_events"] == res["expected_test_targets"] and R.load(c)[0] == "ok"
    assert TR.run(c)["wall_sec"] == res["wall_sec"]                   # a finished cell is not trained again


def test_intensity_heads_and_latent(datadir, tmp_path):
    for head in ("rmtpp", "thp"):
        res = TR.run(cfg_of(tmp_path, "toy", "interval", 1.0, head=head))
        assert res["status"] == "ok" and math.isfinite(res["test"]["C"])
    res = TR.run(cfg_of(tmp_path, "synth_SC_r1_d0", "latent", 0.0))
    assert res["status"] == "ok"


def test_mark_head_and_mark_only(datadir, tmp_path):
    enc = dict(X.GRU, marks=True)
    c = cfg_of(tmp_path, "toy", "interval", 1.0, encoder=enc); c["mark_head"] = True
    res = TR.run(c)
    assert res["primary_selection"] == "J" and abs(res["test"]["J"] - res["test"]["C"] - res["test"]["mark"]) < 1e-9
    c = cfg_of(tmp_path, "toy", "mark_only", 1.0, encoder=enc)
    res = TR.run(c)
    assert res["primary_selection"] == "M" and "M" in res["by_selection"]


def test_mark_only_gives_the_time_head_no_gradient(datadir):
    v = D.load("toy", "train")
    st = D.coarse_stats(v, 1.0)
    cfg = dict(encoder=dict(type="gru", d_model=16, marks=True), head="mix", K=2, treatment="floor", eps=1e-10,
               delta=1.0, delta_model=1.0, mark_head=True, mark_only=True)
    from coarse_tpp.models import build
    m = build(cfg, st, 3)
    b = D.make_batch(v.coarse, np.arange(4), "cpu")
    law, tl, lp = TR.fwd(m, b, st, cfg)
    loss = -(TR.objective(TR.train_ll(law, tl, b.iet[:, 1:], cfg), lp, b, cfg) * b.tmask.double()).sum()
    loss.backward()
    assert m.head.linear.weight.grad is None or float(m.head.linear.weight.grad.abs().sum()) == 0.0


def test_iw_jitter_objective_limits():
    law = lik.LogNormMix(torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64)).expand(50, 2),
                         torch.tensor([-1.0, 1.0], dtype=torch.float64).expand(50, 2),
                         torch.tensor([-0.5, 0.0], dtype=torch.float64).expand(50, 2))
    tau = torch.arange(50, dtype=torch.float64) * 0.3
    one = TR.train_ll(law, None, tau, dict(treatment="iw_jitter", delta_model=0.3, iw_samples=1), torch.Generator().manual_seed(1))
    ju = TR.train_ll(law, None, tau, dict(treatment="jitter_gap", delta_model=0.3), torch.Generator().manual_seed(1))
    assert torch.equal(one, ju)
    big = TR.train_ll(law, None, tau, dict(treatment="iw_jitter", delta_model=0.3, iw_samples=20000), torch.Generator().manual_seed(1))
    ivl = lik.event_loglik(law, tau, "interval", delta=0.3) - math.log(0.3)
    assert float((big - ivl).abs().max()) < 0.02


def test_runner_runs_a_grid_and_skips_finished_cells(datadir, tmp_path):
    run = tmp_path / "grid"
    cs = [cfg_of(run, "synth_SC_r1_d1p0", a, 1.0) for a in ("interval", "jitter_gap")]
    for c in cs:
        c["out_dir"] = str((run / "cells" / c["cell_id"]).resolve())
    X.write("test", run, cells=cs)
    env = dict(__import__("os").environ, COARSE_TPP_DATA=str(datadir), CUDA_VISIBLE_DEVICES="")
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_grid.py"), str(run), "--workers", "2"],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    assert all(R.load(c)[0] == "ok" for c in cs)
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_grid.py"), str(run)], capture_output=True, text=True, env=env)
    assert "0 cells to run" in out.stdout


def test_latent_score_of_coarsened_data(datadir, tmp_path):
    from coarse_tpp import latent_score
    from coarse_tpp.prepare import coarsen
    rng = np.random.default_rng(1)
    for sp, k in (("train", 30), ("dev", 8), ("test", 8)):
        seqs = []
        for _ in range(k):
            t = np.cumsum(rng.exponential(1.0, 25)); t -= t[0]
            seqs.append([dict(time_since_start=float(a), time_since_last_event=float(max(b, 1e-6)), type_event=0)
                         for a, b in zip(t, np.diff(t, prepend=0.0))])
        (datadir / "conttime").mkdir(exist_ok=True)
        pickle.dump({sp: seqs, "dim_process": 1}, open(datadir / "conttime" / f"{sp}.pkl", "wb"))
    row = coarsen.coarsen(datadir, "conttime", 1.0)
    run = tmp_path / "coarse"
    cs = [cfg_of(run, row["dataset"], a, row["delta"]) for a in ("interval", "jitter_time")]
    for c in cs:
        c["out_dir"] = str((run / "cells" / c["cell_id"]).resolve())
        TR.run(c)
    X.write("coarsened_real", run, cells=cs)
    latent_score.main([str(run)])
    rows = [json.loads(x) for x in (run / "latent_scores.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and not any(r["mismatch"] for r in rows) and all(math.isfinite(r["test"]["B"]) for r in rows)


@pytest.mark.parametrize("arm", ["interval", "point_mass"])
def test_evaluate_tie_split_follows_the_score(datadir, tmp_path, arm):
    """Process logs (unit hours) keep exact zeros and genuine 1 ms gaps (2.8e-7 h): at delta > 0 only the recorded zeros
    are ties, at delta = 0 every gap <= TIE0; the overall scores do not depend on the split."""
    from coarse_tpp.models import build
    ms, sec = 1 / 3.6e6, 1 / 3600
    rng = np.random.default_rng(2)
    for sp in ("train", "dev", "test"):
        seqs = []
        for _ in range(6):
            g = np.concatenate([[0.0], np.tile([0.0, ms, sec], 5), rng.exponential(0.5, 5)])
            seqs.append([dict(time_since_start=float(t), time_since_last_event=float(x), type_event=0)
                         for t, x in zip(np.cumsum(g), g)])
        (datadir / "bpi_ties").mkdir(exist_ok=True)
        pickle.dump({sp: seqs, "dim_process": 1}, open(datadir / "bpi_ties" / f"{sp}.pkl", "wb"))
    v = D.load("bpi_ties", "test")
    tau = np.concatenate([x[1:] for x in v.coarse.iet])
    assert (tau == ms).sum() == 30 and (tau == 0).sum() == 30
    for delta, n_ties in ((ms, 30), (0.0, 60)):
        cfg = cfg_of(tmp_path, "bpi_ties", arm if delta else "floor_1e-10", delta)
        st = D.coarse_stats(D.load("bpi_ties", "train"), delta)
        torch.manual_seed(0)
        model = build(cfg, st, 1)
        out, per_seq = TR.evaluate(model, v, st, cfg, torch.device("cpu"))
        assert out["n_events"] == len(tau) and out["n_ties"] == n_ties and out["tie_share"] == n_ties / len(tau)
        for k in ("C", "floor_1e-6"):
            mix = out["tie_share"] * out[f"{k}_tie"] + (1 - out["tie_share"]) * out[f"{k}_nontie"]
            assert abs(mix - out[k]) < 1e-9
        b = D.make_batch(v.coarse, np.arange(len(v.coarse)), "cpu")
        with torch.no_grad():
            law, tl, _ = TR.fwd(model, b, st, cfg)
            c = Ro.C(law, tl, b.iet[:, 1:].double(), delta, TR.rule(cfg))
        assert abs(float(c[b.tmask > 0].mean()) - out["C"]) < 1e-9
