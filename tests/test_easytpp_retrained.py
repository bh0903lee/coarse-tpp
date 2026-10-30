"""Retrained EasyTPP models: vendored training copy, configurations, data files, integration rules, scores of the nine
architectures, cache identity and an end-to-end CPU smoke test (training, scoring, reproduction check, reports)."""
import hashlib
import json
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from coarse_tpp import data as D
from coarse_tpp import easytpp as E
from coarse_tpp import easytpp_train as T
from coarse_tpp import intensity_eval as IE
from coarse_tpp import intensity_exceedance as X
from coarse_tpp import rescore_retrained as R

VENDOR = Path(E.__file__).resolve().parents[1] / "third_party" / "easy_tpp_train"


# ------------------------------------------------------------------ vendored copy
def test_training_copy_files_match_their_recorded_hashes():
    listed = set()
    for line in (VENDOR / "SOURCE_SHA256.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        listed.add(name)
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == digest, name
    patched = dict(re.findall(r"\| `(easy_tpp/[^`]+)` \| `[0-9a-f]{64}` \| `([0-9a-f]{64})` \|", (VENDOR / "PATCHES.md").read_text()))
    assert len(patched) == 16
    for name, digest in patched.items():
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == digest, name
        assert "Patch" in (VENDOR / name).read_text(), name
    files = {str(p.relative_to(VENDOR)) for p in (VENDOR / "easy_tpp").rglob("*.py")}
    assert files == ({n for n in listed if n.startswith("easy_tpp/")} | set(patched))


def test_activate_switches_between_the_two_copies():
    E.activate("models")
    import easy_tpp
    assert Path(easy_tpp.__file__).resolve().is_relative_to((VENDOR.parent / "easy_tpp").resolve())
    mod = E.activate("train")
    from easy_tpp.config_factory import ModelConfig  # noqa: F401  (only in the training copy)
    assert Path(mod.__file__).resolve().is_relative_to(VENDOR.resolve())
    from easy_tpp.model.torch_model.torch_basemodel import mark_loss_on
    assert mark_loss_on() in (True, False)


# ------------------------------------------------------------------ configurations
REF_S2P2 = """pipeline_config_id: runner_config
data:
  data_email_eu:
    data_format: pkl
    train_dir: DATA
    valid_dir: DATA
    test_dir: DATA
    data_specs:
      num_event_types: 1
      pad_token_id: 1
      padding_side: right
S2P2_data_email_eu_train:
  base_config:
    stage: train
    backend: torch
    dataset_id: data_email_eu
    runner_id: std_tpp
    model_id: S2P2
    base_dir: RUN
  trainer_config:
    batch_size: 32
    max_epoch: 100
    shuffle: false
    optimizer: adam
    learning_rate: 0.001
    valid_freq: 1
    use_tfb: false
    metrics:
    - rmse
    - acc
    gpu: 0
    lr_scheduler: true
    weight_decay: 0.0
    warmup_pct: 0.01
    lr_decay_style: cosine
    seed: 0
  model_config:
    hidden_size: 64
    time_emb_size: 16
    num_layers: 4
    num_heads: 2
    dropout_rate: 0.0
    use_ln: false
    loss_integral_num_sample_per_step: 10
    use_mc_samples: true
    model_specs:
      P: 16
      dropout_rate: 0.1
      act_func: gelu
      for_loop: true
      pre_norm: false
      post_norm: true
      int_forward_variant: false
      int_backward_variant: true
      relative_time: true
"""


def test_configuration_is_the_one_the_paper_models_were_trained_with():
    cfg = T.make_config("email_eu", "S2P2", 0, 100, Path("RUN"), Path("DATA"), 1, gpu=0)
    assert yaml.safe_dump(cfg, sort_keys=False) == REF_S2P2
    c = T.make_config("bpi_bpic12", "IntensityFree", 3, 100, Path("RUN"), Path("DATA"), 26, gpu=-1)
    blk = c["IntensityFree_bpi_bpic12_train"]
    assert blk["model_config"]["model_specs"] == {"num_mix_components": 3} and blk["trainer_config"]["seed"] == 3
    assert blk["trainer_config"]["gpu"] == -1 and c["data"]["bpi_bpic12"]["data_specs"]["pad_token_id"] == 26
    assert T.strip_paths(cfg) == T.strip_paths(T.make_config("email_eu", "S2P2", 0, 100, Path("x"), Path("y"), 1))
    assert set(T.SPECS) == set(E.MODELS) and T.EXPERIMENTS["joint"]["mark_loss"] and T.MICROBATCH == {("collegemsg", "AttNHP"): 4}


def test_mark_loss_switch(monkeypatch):
    env = {"EASYTPP_MARK_LOSS": "1"}
    T.set_mark_loss(False, env)
    assert env == {}
    T.set_mark_loss(True, env)
    assert env == {"EASYTPP_MARK_LOSS": "1"}


# ------------------------------------------------------------------ data files
def _write_folder(root, name, splits, dim):
    (root / name).mkdir(parents=True, exist_ok=True)
    for sp, seqs in splits.items():
        evs = [[{"time_since_start": float(t), "time_since_last_event": float(g), "type_event": int(k)}
                for t, g, k in zip(np.cumsum(gs), gs, ks)] for gs, ks in seqs]
        with open(root / name / f"{sp}.pkl", "wb") as f:
            pickle.dump({sp: evs, "dim_process": dim}, f)


def _seqs(rng, n, K, floor=1e-6, zero=False, lengths=(2, 9)):
    out = []
    for _ in range(n):
        L = int(rng.integers(*lengths))
        g = np.floor(rng.exponential(2.0, L)); g[0] = 0.0
        if not zero:
            g = np.where(g == 0, floor, g)
        out.append((g, rng.integers(0, K, L)))
    return out


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "DATA_DIR", tmp_path / "data")
    return tmp_path / "data"


def test_prepared_file_keeps_sequences_and_writes_positive_tie_values(data_root, tmp_path):
    rng = np.random.default_rng(1)
    _write_folder(data_root, "bpi_bpic12", {sp: _seqs(rng, 6, 3, zero=True) + [(np.array([0.0]), np.array([1]))]
                                            for sp in E.SPLITS}, 3)
    p = E.prepare_file("bpi_bpic12", tmp_path / "prep")
    d = E.read_file(p)
    assert list(d) == ["dim_process", "train", "dev", "test"] and len(d["train"]) == 7     # single-event sequence kept
    g = [e["time_since_last_event"] for s in d["train"] for e in s[1:]]
    assert min(g) == 1e-7 and E.tie_threshold("bpi_bpic12") == pytest.approx(1e-7 * (1 + 1e-6))
    assert E.check_file("bpi_bpic12", d) == []
    d["dev"][0][1]["time_since_last_event"] += 1.0
    assert E.check_file("bpi_bpic12", d)


def test_one_based_test_split_of_the_retweet_subsample_is_shifted(data_root, tmp_path):
    rng = np.random.default_rng(2)
    _write_folder(data_root, "retweet_sub", {"train": _seqs(rng, 5, 3), "dev": _seqs(rng, 5, 3)}, 3)
    test = [(g, k + 1) for g, k in _seqs(rng, 5, 3)]
    test[0] = (test[0][0], np.r_[test[0][1][:-1], 3]); test[1] = (test[1][0], np.r_[1, test[1][1][1:]])
    _write_folder(data_root, "retweet_sub", {"test": test}, 3)
    d = E.read_file(E.prepare_file("retweet_sub", tmp_path / "prep"))
    assert min(e["type_event"] for s in d["test"] for e in s) == 0 and max(e["type_event"] for s in d["test"] for e in s) == 2


# ------------------------------------------------------------------ integration rules
def test_rules_are_explicit_and_carry_their_identity():
    r = IE.Rule()
    assert r.nodes_per_panel("email_eu", "AttNHP") == 128 and r.nodes_per_panel("retweet_sub", "AttNHP") == 32
    assert r.identity("x", "S2P2") == dict(per_panel=32, strata=16384, strat_seed=0, cell_panels=1500)
    assert r.identity("x", "NHP") == dict(per_panel=32)
    assert IE.Rule(strat_seed=1).identity("x", "S2P2") != r.identity("x", "S2P2")
    assert IE.Rule(cell_panels=3).identity("x", "S2P2") != r.identity("x", "S2P2")
    assert not IE.Rule(strata=0).stratified("S2P2")
    u, W, C = IE.cell_rule(IE.Rule(cell_panels=3), "x", "S2P2")
    ud, wd = IE.dense_cell_rule(3)
    assert len(u) == 6 and torch.allclose(u, ud) and torch.allclose(W, wd)
    u, W, C = IE.cell_rule(r, "x", "NHP")
    assert len(u) == 320 and torch.allclose(u, IE.graded_rule(32)[0])


def test_quadrature_rules_integrate_smooth_functions():
    u, w = IE.graded_rule(32)
    assert float((w * torch.exp(3 * u)).sum()) == pytest.approx((math.exp(3) - 1) / 3, rel=1e-12)
    u, w = IE.dense_cell_rule(50)
    assert float((w * u ** 3).sum()) == pytest.approx(0.25, rel=1e-12)
    u, W, C = (torch.as_tensor(v) for v in IE.rule_from_panels(IE.panels_graded(8)))
    assert torch.allclose(C @ torch.cos(u), torch.sin(u), atol=1e-12)


# ------------------------------------------------------------------ models and scores
def tiny_cfg(mid, K=2):
    m = dict(hidden_size=8, time_emb_size=4, num_layers=1, num_heads=1, num_event_types=K, num_event_types_pad=K + 1,
             event_pad_index=K, model_id=mid, loss_integral_num_sample_per_step=4, model_specs={})
    m.update({k: v for k, v in T.SPECS[mid].items() if k != "num_layers"})
    if mid == "S2P2":
        m["model_specs"] = dict(m["model_specs"], P=4)
    return {"model_config": m}


def tiny_seqs(K=2):
    rng = np.random.default_rng(0)
    out = []
    for n in (5, 3, 7, 4):
        dt = np.floor(rng.exponential(2.0, n)); dt[0] = 0.0
        dt = np.where(dt == 0, 1e-6, dt)
        out.append((np.cumsum(dt), dt, rng.integers(0, K, n)))
    return out


@pytest.fixture
def train_pkl(tmp_path):
    p = tmp_path / "d.pkl"
    seqs = [[{"time_since_start": float(t), "time_since_last_event": float(g), "type_event": int(k)}
             for t, g, k in zip(*s)] for s in tiny_seqs()]
    with open(p, "wb") as f:
        pickle.dump({"train": seqs, "dev": seqs, "test": seqs, "dim_process": 2}, f)
    return p


@pytest.mark.parametrize("mid", [m for m in E.MODELS])
def test_every_architecture_builds_and_gives_finite_time_and_joint_scores(mid, train_pkl):
    torch.manual_seed(0)
    m = IE.build_model(mid, tiny_cfg(mid), torch.device("cpu"), train_pkl)
    rule = IE.Rule(per_panel=8, strata=64, cell_panels=20)
    r = IE.score_split(m, mid, tiny_seqs(), 2, 1.0, rule, "x", D.TIE_STORED, torch.device("cpu"), joint=True)
    assert r["n"] == 15 and r["n_tie"] > 0
    for k in ("floor_1e-10", "floor_1e-6", "floor_1e-5", "interval", "joint_floor_1e-6", "joint_interval", "mark_rec"):
        assert math.isfinite(r[k]), (mid, k)
    if mid not in ("IntensityFree",):
        assert r["joint_interval"] <= r["interval"] + 1e-9 and r["mark_rec"] <= 1e-9
    if mid == "FullyNN":
        assert r["mark_rec"] == pytest.approx(-math.log(2), abs=1e-6)       # lambda_k = lambda / K
    if mid == "IntensityFree":
        assert r["joint_interval"] == pytest.approx(r["interval"] + r["mark_rec"], abs=1e-9)


def test_mark_probabilities_of_the_cell_sum_to_one_and_the_cell_rule_is_consistent():
    torch.manual_seed(1)
    u_W_C = IE.cell_rule(IE.Rule(per_panel=16), "x", "NHP")
    G = len(u_W_C[0])
    lam = 0.3 + torch.rand(2, 3, 1, 3, dtype=torch.float64) * torch.exp(-u_W_C[0])[None, None, :, None]   # smooth in s
    conds = [IE.mark_cell_terms(lam.expand(2, 3, G, 3), torch.full((2, 3), k), 1.0, u_W_C) for k in range(3)]
    tot = torch.logsumexp(torch.stack([c[0] for c in conds]), 0)
    assert torch.allclose(tot, torch.zeros_like(tot), atol=1e-12)
    assert conds[0][1].abs().max() < 1e-10                                   # cellcheck: exact for a smooth intensity


def test_stratified_compensator_is_seeded_per_batch_and_close_to_the_graded_rule(train_pkl):
    torch.manual_seed(2)
    m = IE.build_model("S2P2", tiny_cfg("S2P2"), torch.device("cpu"), train_pkl)
    args = (m, "S2P2", tiny_seqs(), 2, 1.0)
    a = IE.score_split(*args, IE.Rule(strata=4096), "x", D.TIE_STORED, "cpu")
    b = IE.score_split(*args, IE.Rule(strata=4096), "x", D.TIE_STORED, "cpu")
    c = IE.score_split(*args, IE.Rule(strata=4096, strat_seed=1), "x", D.TIE_STORED, "cpu")
    g = IE.score_split(*args, IE.Rule(strata=0, per_panel=64), "x", D.TIE_STORED, "cpu")
    assert a == b and a["interval"] != c["interval"]
    assert abs(a["interval"] - g["interval"]) < 1e-3 and abs(a["floor_1e-6"] - g["floor_1e-6"]) < 1e-3


def test_intensity_free_interval_tends_to_density_times_delta(train_pkl):
    torch.manual_seed(3)
    m = IE.build_model("IntensityFree", tiny_cfg("IntensityFree"), torch.device("cpu"), train_pkl)
    assert (m.mean_log_inter_time, m.std_log_inter_time) == IE.dt_stats(train_pkl)
    from coarse_tpp.rescore_intensity import make_batch
    b = make_batch(tiny_seqs(), 2, "cpu")
    o = IE.time_scores(IE.Intensity(m, "IntensityFree", b), 1e-5, IE.Rule(), "x", D.TIE_STORED, "cpu")
    sel = o["mask"] & ~o["tie"]
    assert torch.allclose(o["interval"][sel], o["floor_1e-6"][sel] + math.log(1e-5), atol=1e-4)
    xs = torch.full((*o["interval"].shape, 1), 3.0, dtype=torch.float64)
    p = IE.ifl_params(m, b)
    lf = IE.ifl_log_interval(p, torch.zeros_like(xs[..., 0]), xs[..., 0], m.mean_log_inter_time, m.std_log_inter_time)
    ls = IE.ifl_log_survival(p, xs, m.mean_log_inter_time, m.std_log_inter_time)[..., 0]
    assert torch.allclose(torch.exp(lf) + torch.exp(ls), torch.ones_like(lf), atol=1e-12)


def test_exceedance_graded_survival_matches_the_interval_scores(train_pkl):
    torch.manual_seed(4)
    m = IE.build_model("RMTPP", tiny_cfg("RMTPP"), torch.device("cpu"), train_pkl)
    from coarse_tpp.rescore_intensity import make_batch
    b = make_batch(tiny_seqs(), 2, "cpu")
    ev = IE.Intensity(m, "RMTPP", b)
    o = IE.time_scores(ev, 1.0, IE.Rule(per_panel=64), "x", D.TIE_STORED, "cpu")
    xs = torch.full((*o["tie"].shape, 1), 1.0, dtype=torch.float64)
    p0 = X.survival(ev, xs, "graded64", IE.Rule(), "cpu")[..., 0]       # P(s >= delta)
    sel = o["mask"] & o["tie"]
    assert torch.allclose(torch.log1p(-p0[sel]), o["interval"][sel], atol=1e-9)
    s = X.survival(ev, torch.stack([xs[..., 0], 2 * xs[..., 0]], -1), "strat", IE.Rule(strata=8192), "cpu")
    g = X.survival(ev, torch.stack([xs[..., 0], 2 * xs[..., 0]], -1), "graded64", IE.Rule(), "cpu")
    assert torch.allclose(s, g, atol=1e-3)


def test_exceedance_selection_requires_every_seed():
    cells = [dict(dataset="d", model=m, seed=s, dev={R.F6: v + s * 0.01, R.IV: w})
             for m, v, w in (("A", 1.0, -2.0), ("B", 0.5, -1.0)) for s in range(5)]
    cells.append(dict(dataset="d", model="C", seed=0, dev={R.F6: 9.0, R.IV: 9.0}))
    sel, means = X.select(cells, "d")
    assert sel["floor"][0] == "A" and sel["interval"][0] == "B" and means["floor"]["C"] is None


# ------------------------------------------------------------------ end to end (CPU smoke)
@pytest.fixture
def smoke(tmp_path, monkeypatch, data_root):
    rng = np.random.default_rng(5)
    for ds, K in (("email_eu", 1), ("retweet_sub", 3)):
        _write_folder(data_root, ds, {sp: _seqs(rng, 12, K, lengths=(3, 12)) for sp in E.SPLITS}, K)
    prep = tmp_path / "prep"
    for ds in ("email_eu", "retweet_sub"):
        E.prepare_file(ds, prep)
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps([{"dataset": "email_eu", "delta": 1.0}, {"dataset": "retweet_sub", "delta": 1.0}]))
    monkeypatch.setenv("EASYTPP_MARK_LOSS", "0")
    return dict(prep=prep, audit=audit, runs=tmp_path / "runs", out=tmp_path / "out")


def test_train_score_check_and_report_on_cpu(smoke, monkeypatch):
    s = smoke
    rec = T.train_one("joint", "retweet_sub", "RMTPP", 0, s["prep"], s["runs"], epochs=1)
    assert rec["status"] == "ok" and rec["mark_loss"] is True and rec["checkpoint_sha256"]
    rdir = s["runs"] / "retweet_sub" / "RMTPP" / "s0"
    cfg = yaml.safe_load((rdir / "config_in.yaml").read_text())
    assert T.strip_paths(cfg) == T.strip_paths(T.make_config("retweet_sub", "RMTPP", 0, 1, rdir, Path("p"), 3, gpu=-1))
    rule = IE.Rule(per_panel=8)
    R.score("joint", s["runs"], s["prep"], s["out"], models=["RMTPP"], seeds=(0,), rule=rule, audit=s["audit"])
    cell = s["out"] / "scores" / "retweet_sub__RMTPP__s0.json"
    c = json.loads(cell.read_text())
    assert any("full run" in p for p in c["problems"])            # one epoch: not a full run
    assert math.isfinite(c["test"]["joint_interval"]) and c["repro_logged"]["primary"] == "joint"
    # the logged training estimate and ours agree up to the Monte Carlo error of EasyTPP's 4-sample compensator
    assert c["repro_logged"]["time_dev"] < 1.0
    mtime = cell.stat().st_mtime_ns
    R.score("joint", s["runs"], s["prep"], s["out"], models=["RMTPP"], seeds=(0,), rule=rule, audit=s["audit"])
    assert cell.stat().st_mtime_ns == mtime                         # cache reused
    R.score("joint", s["runs"], s["prep"], s["out"], models=["RMTPP"], seeds=(0,), rule=IE.Rule(per_panel=16), audit=s["audit"])
    assert json.loads(cell.read_text())["rule"] == {"per_panel": 16}  # another rule: re-scored
    R.repro("joint", s["runs"], s["prep"], s["out"], models=["RMTPP"], seeds=(0,), repeats=3)
    d = json.loads((s["out"] / "repro" / "retweet_sub__RMTPP__s0.json").read_text())
    assert d["ours_key"] == "joint_floor_1e-6" and d["checkpoint_sha256"] == rec["checkpoint_sha256"]
    cells = R.load_cells("joint", s["out"])
    assert "repro" in cells[0] and cells[0]["quarantined"]
    lines = R.report("joint", s["out"])
    assert (s["out"] / "summary.json").exists() and any("RMTPP" in x for x in lines)
    # a checkpoint replaced at the same path makes the cache stale (report) and is re-scored (score)
    ck = s["runs"] / "retweet_sub" / "RMTPP" / "s0" / rec["checkpoint"]
    sd = torch.load(ck, weights_only=True)
    sd = {k: v + 0.01 for k, v in sd.items()}
    torch.save(sd, ck)
    assert any("stale input: checkpoint" in p for p in R.load_cells("joint", s["out"])[0]["problems"])
    R.score("joint", s["runs"], s["prep"], s["out"], models=["RMTPP"], seeds=(0,), rule=IE.Rule(per_panel=16), audit=s["audit"])
    c2 = json.loads(cell.read_text())
    assert c2["inputs"]["checkpoint_sha256"] != c["inputs"]["checkpoint_sha256"]
    assert any("checkpoint changed after training" in p for p in c2["problems"])
    assert any("not bound" in p for p in R.load_cells("joint", s["out"])[0]["problems"])


def test_time_only_training_smoke_and_exceedance(smoke):
    s = smoke
    for mid in ("NHP", "IntensityFree"):
        rec = T.train_one("time_only", "email_eu", mid, 0, s["prep"], s["runs"], epochs=1, max_seqs=8)
        assert rec["status"] == "ok" and rec["mark_loss"] is False and rec["max_seqs"] == 8
    R.score("time_only", s["runs"], s["prep"], s["out"], models=["NHP", "IntensityFree"], seeds=(0,),
            rule=IE.Rule(per_panel=8), audit=s["audit"])
    c = json.loads((s["out"] / "scores" / "email_eu__NHP__s0.json").read_text())
    assert any("data file differs" in p for p in c["problems"])     # trained on the 8-sequence smoke file
    R.report("time_only", s["out"])
    pkl = s["prep"] / "data_email_eu.pkl"
    hg = X.h_grid(pkl, "email_eu", 1.0)
    kh = [hg[k][0] for k in X.A.H_KEYS]
    out = s["out"] / "exc"; out.mkdir()
    rec = X.score_run("email_eu", "NHP", 0, 1.0, kh, IE.Rule(), s["runs"], pkl, s["out"], out, torch.device("cpu"))
    assert 0 <= rec["brier_mean"] <= 1 and rec["conv_32_128"]["max_abs"] < 1e-6 and rec["identity"]["primary"] == "graded128"
    assert X.cache_ok(rec, "email_eu", "NHP", 0, IE.Rule(), kh, 1.0, s["runs"])
    assert not X.cache_ok(rec, "email_eu", "NHP", 0, IE.Rule(), [k + 1 for k in kh], 1.0, s["runs"])
    rec = X.score_run("email_eu", "IntensityFree", 0, 1.0, kh, IE.Rule(), s["runs"], pkl, s["out"], out, torch.device("cpu"))
    assert rec["methods"] == ["closed"] and "tie_check" in rec
