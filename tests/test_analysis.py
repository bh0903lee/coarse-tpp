"""Summary statistics on fabricated results."""
import json
import math

import numpy as np

from pathlib import Path

from coarse_tpp import analysis as A, experiments as X, synthetic as S


def test_holm_and_t():
    adj = A.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    m, lo, p = A.t_one_sided([0.1, 0.12, 0.08, 0.11, 0.09])
    assert abs(m - 0.1) < 1e-12 and lo < m and p < 0.01


def fabricate(run, cells):
    lines = []
    for c in cells:
        cfg = dict(c["config"], out_dir=str(run / "cells" / c["cell_id"]))
        (run / "configs").mkdir(parents=True, exist_ok=True)
        p = run / "configs" / f"{c['cell_id']}.json"; p.write_text(json.dumps(cfg))
        res = dict(cell_id=c["cell_id"], status="ok", config=cfg, arm=cfg["arm"], seed=cfg["seed"], expected_test_targets=10,
                   test=dict(c["test"], n_events=10, ll=c["test"]["C"]), dev=dict(ll=0.0), wall_sec=1.0, selected_epoch=1,
                   history=[], by_selection={"C": dict(dev_criterion=c["dev"], test=c["test"])})
        (run / "cells" / c["cell_id"]).mkdir(parents=True, exist_ok=True)
        (run / "cells" / c["cell_id"] / "result.json").write_text(json.dumps(res))
        lines.append(json.dumps(dict(cell_id=c["cell_id"], config=str(p))))
    (run / "grid.jsonl").write_text("\n".join(lines) + "\n")


def test_report_real_on_a_fabricated_run(tmp_path):
    run = tmp_path / "run"
    cells = []
    rng = np.random.default_rng(0)
    for arm, base in (("floor_1e-10", -1.0), ("floor_1e-6", -1.02), ("jitter_gap", -0.95), ("interval", -0.90)):
        for lr in ("lr3e4", "lr1e3"):
            for s in range(5):
                cfg = X.cell("t", run, "email_eu", arm, s, lr, 1.0)
                C = base + 0.001 * rng.standard_normal() + (0.0 if lr == "lr1e3" else -0.05)
                e6 = 3.0 if arm == "floor_1e-6" else 1.0
                cells.append(dict(cell_id=cfg["cell_id"], config=cfg, dev=C, test={"C": C, "floor_1e-6": e6}))
    fabricate(run, cells)
    L, s = A.report_real(run, "t", ("floor_1e-10", "floor_1e-6", "jitter_gap", "interval"), {"email_eu": dict(tie_share=0.2)})
    assert s["preservation"]["preserved"] == 1 and s["n_better"] == 1 and s["top_changed"] == 1
    assert s["own_floor_first"] == 1 and math.isfinite(s["tau_median"])


def test_advantage_needs_min_seeds(tmp_path):
    run = tmp_path / "run"
    cells = []
    for arm, base in (("floor_1e-10", -1.0), ("floor_1e-6", -1.02), ("jitter_gap", -0.95), ("interval", -0.75)):
        for s in range(2):
            cfg = X.cell("t", run, "email_eu", arm, s, "lr1e3", 1.0)
            C = base + 0.001 * s
            cells.append(dict(cell_id=cfg["cell_id"], config=cfg, dev=C, test={"C": C, "floor_1e-6": 1.0}))
    fabricate(run, cells)
    L, s = A.report_real(run, "t", ("floor_1e-10", "floor_1e-6", "jitter_gap", "interval"), {"email_eu": dict(tie_share=0.2)})
    assert s["n_better"] == 0 and any("not judged (n = 2)" in x for x in L)


def test_floor_convention_has_no_training_report(tmp_path):
    import pytest
    with pytest.raises(SystemExit, match="rescore_trained"):
        A.run_report("floor_convention", tmp_path)


def _bounds_cells(run, ds, seeds=range(5)):
    """Fabricated component-bounds cells: under the bounds the floor model's self-preference shrinks and C changes."""
    rng = np.random.default_rng(1)
    cells = []
    for bound, shrink in (("none", 1.0), ("sigmin2", 0.1), ("mumin1e-4", 0.5), ("sigmin2_mumin1e-4", 0.05)):
        for arm in X.BOUNDS_ARMS:
            for lr in X.LRS:
                for s in seeds:
                    tag = arm if bound == "none" else X.bounds_tag(arm, bound)
                    cfg = X.cell("component_bounds" if bound != "none" else "rank_change", run, ds, arm, s, lr, 1.0, tag=tag)
                    C = -1.0 - (0.02 if "sigmin2" in bound else 0.0) + 0.001 * rng.standard_normal() + (0.0 if lr == "lr1e3" else -0.05)
                    test = {"C": C, "tie_share": 0.2}
                    for f, e in X.FLOOR_ARMS.items():
                        own = 2.0 * shrink if arm == f else 0.0                 # self-preference at its own floor
                        test[f] = -1.0 + own + 0.001 * rng.standard_normal()
                        test[f + "_tie"] = -2.0 + 5 * own; test[f + "_nontie"] = (test[f] - 0.2 * test[f + "_tie"]) / 0.8
                    cells.append(dict(cell_id=cfg["cell_id"], config=cfg, dev=C, test=test))
    return cells


def test_report_component_bounds_on_fabricated_runs(tmp_path):
    run, base = tmp_path / "bounds", tmp_path / "rank"
    cells = _bounds_cells(run, "email_eu")
    fabricate(run, [c for c in cells if c["config"]["experiment"] == "component_bounds"])
    fabricate(base, [c for c in cells if c["config"]["experiment"] == "rank_change"])
    L, s = A.report_component_bounds(run, [base])
    m = s["email_eu"]
    assert set(m) == {"none", "sigmin2", "mumin1e-4", "sigmin2_mumin1e-4"}
    for b in m:
        for f in X.FLOOR_ARMS:
            mg = m[b]["margins"][f]
            assert math.isclose(mg["tie"] + mg["nontie"], mg["total"], abs_tol=1e-9)
    assert m["none"]["margins"]["floor_1e-6"]["total"] > m["mumin1e-4"]["margins"]["floor_1e-6"]["total"] > \
        m["sigmin2"]["margins"]["floor_1e-6"]["total"] > m["sigmin2_mumin1e-4"]["margins"]["floor_1e-6"]["total"] > 0
    assert m["none"]["tops"]["floor_1e-6"] == "floor_1e-6" and m["none"]["cost"] == {}
    assert all(v < -0.01 for v in m["sigmin2"]["cost"].values()) and all(abs(v) < 0.01 for v in m["mumin1e-4"]["cost"].values())
    assert any("## email_eu" in x for x in L)
    # a missing seed makes the data set pending
    run2 = tmp_path / "bounds2"
    fabricate(run2, [c for c in cells if c["config"]["experiment"] == "component_bounds" and c["config"]["seed"] != 3])
    _, s2 = A.report_component_bounds(run2, [base])
    assert s2["email_eu"] == "pending"


def _latent(run, rows):
    run.mkdir(parents=True, exist_ok=True)
    (run / "latent_scores.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _tri_rows(run, datasets, arms, seed=2, skip=lambda ds, arm: False, b_alt_missing=lambda ds, arm, s: False):
    """Fabricated grid (configs + grid.jsonl) and latent rows of a triangular comparison; returns the rows."""
    rng = np.random.default_rng(seed)
    cs, rows = [], []
    for ds in datasets:
        for arm in arms:
            for lr in X.LRS:
                for s in range(5):
                    cfg = X.cell("triangular_coarsened", run, ds, arm, s, lr, 1.0)
                    cs.append(cfg)
                    if skip(ds, arm):
                        continue
                    good = ds.endswith("r1")
                    b = -1.0 + (0.1 if arm == "triangular" and good else 0.0) + 0.002 * rng.standard_normal() + (0.0 if lr == "lr1e3" else -0.05)
                    b_alt = b + (0.05 if arm == "point_mass" else -0.3)
                    test, dev = dict(C=-1.0, B=b, B_alt=b_alt), dict(B=b, B_alt=b_alt)
                    if b_alt_missing(ds, arm, s):
                        test, dev = dict(C=-1.0, B=b), dict(B=b)                       # a row of the 1.0 schema
                    rows.append(dict(cell_id=cfg["cell_id"], arm=arm, dataset=ds, seed=s, mismatch=False, wall_sec=2.0 if arm == "triangular" else 1.0,
                                     config=dict(dataset=ds, encoder=cfg["encoder"], head="mix", K=8, lr_tag=lr, treatment=cfg["treatment"], delta=1.0),
                                     dev_crit=b, dev=dev, test=test))
    X.write("triangular_coarsened", run, cells=cs)
    _latent(run, rows)
    return rows


def test_report_triangular_on_fabricated_latent_scores(tmp_path):
    """A coarsened unit where the triangular objective beats jitter_time and the best-readout baseline; the point-mass
    baseline is better under its alternative readout, which the best-readout rule must pick."""
    run = tmp_path / "tri"
    _tri_rows(run, ("coarsened_taxi_r1", "coarsened_taxi_r0p1"), ("triangular",) + A.TRI_BASE + X.TRI_IW_ARMS)
    L, s = A.report_triangular(run, [])
    real = {(x["dataset"], x["encoder"]): x for x in s["real"]}
    good, poor = real[("coarsened_taxi_r1", "gru")], real[("coarsened_taxi_r0p1", "gru")]
    assert good["a"] and good["b"] and good["baseline"] == ["point_mass", "B_alt"] and good["a_stat"]["n"] == 5
    assert not poor["a"] and not poor["b"]
    assert s["n_eligible"] == 1 and s["n_both_better"] == 1 and s["c_noloss"] == "pending" and s["verdict"] == "pending"
    assert any("| triangular | 20 | 2.00 | 2.00 | 2.00 |" in x for x in L)   # matched training-time ratio to jitter_gap
    # a mismatching cell is dropped, and a unit missing a baseline is pending
    rows = [json.loads(x) for x in (run / "latent_scores.jsonl").read_text().splitlines()]
    run2 = tmp_path / "tri2"
    X.write("triangular_coarsened", run2, cells=[json.loads(Path(p).read_text()) for p in sorted((run / "configs").glob("*.json"))])
    _latent(run2, [dict(r, mismatch=(r["arm"] == "interval" and r["seed"] == 4)) for r in rows])
    _, s2 = A.report_triangular(run2, [])
    assert all(x["status"] == "pending" for x in s2["real"]) and s2["verdict"] == "pending"


def test_report_triangular_keeps_planned_units_and_old_rows_out(tmp_path):
    """A planned combination without any triangular result stays in the denominator as pending (the verdict cannot
    become true from a truncated population), and rows of the 1.0 schema (no B_alt) are left out and reported."""
    run = tmp_path / "tri"
    _tri_rows(run, ("coarsened_taxi_r1", "coarsened_conttime_r1"), ("triangular",) + A.TRI_BASE,
              skip=lambda ds, arm: ds == "coarsened_conttime_r1" and arm == "triangular")
    L, s = A.report_triangular(run, [])
    st = {x["dataset"]: x["status"] for x in s["real"]}
    assert st == {"coarsened_taxi_r1": "complete", "coarsened_conttime_r1": "pending"}
    assert s["n_eligible"] == 2 and s["n_pending"] == 1 and s["verdict"] == "pending"
    run2 = tmp_path / "tri_old"
    _tri_rows(run2, ("coarsened_taxi_r1",), ("triangular",) + A.TRI_BASE, b_alt_missing=lambda ds, arm, s: arm == "interval" and s == 0)
    L2, s2 = A.report_triangular(run2, [])
    assert s2["real"][0]["status"] == "pending" and any("rows of an older schema" in x for x in L2)
    cells, stale = A.latent_rows([run2 / "latent_scores.jsonl"], with_stale=True)
    assert stale == 2 and all("B_alt" in r["by_selection"]["C"]["test"] for r in cells.values())


def test_recorded_c_table_uses_the_planned_grid_only(tmp_path):
    """The recorded-data C table takes its combinations from the triangular_recorded grid: synthetic and coarsened
    cells of other runs never enter it, and a planned combination without results is pending."""
    rng = np.random.default_rng(3)
    def cells_of(run, ds, arms, seeds=range(5), experiment="triangular_recorded"):
        out = []
        for arm in arms:
            for lr in X.LRS:
                for s in seeds:
                    cfg = X.cell(experiment, run, ds, arm, s, lr, 1.0)
                    C = -1.0 + (0.002 if arm == "triangular" else 0.0) + 0.001 * rng.standard_normal()
                    out.append(dict(cell_id=cfg["cell_id"], config=cfg, dev=C, test={"C": C, "floor_1e-6": 1.0, "tie_share": 0.2}))
        return out
    tri, rank, syn = tmp_path / "tri", tmp_path / "rank", tmp_path / "syn"
    fabricate(tri, cells_of(tri, "email_eu", ("triangular",)) + cells_of(tri, "volcano", ("triangular",), seeds=range(3)))
    fabricate(rank, cells_of(rank, "email_eu", ("interval", "jitter_gap"), experiment="rank_change")
              + cells_of(rank, "volcano", ("interval", "jitter_gap"), experiment="rank_change"))
    fabricate(syn, cells_of(syn, "synth_SP_r1_d1p0", ("triangular", "interval", "jitter_gap"), experiment="synthetic")
              + cells_of(syn, "coarsened_taxi_r1", ("triangular", "interval", "jitter_gap"), experiment="coarsened_more"))
    status, L, n_ok, n_units = A.triangular_c_table([tri, rank, syn])
    assert n_units == 2 and n_ok == 1 and status == "pending"                 # volcano lacks seeds 3, 4
    assert not any("synth_" in x or "coarsened_" in x for x in L)
    status2, L2, n_ok2, n_units2 = A.triangular_c_table([syn, rank])          # no recorded grid at all
    assert status2 == "pending" and n_units2 == 0 and any("pending" in x for x in L2)
    # synthetic settings outside Syn.main_settings() do not enter the synthetic table
    cells = {("synth_SP_r1_d0p3", "gru", "triangular", "lr1e3", 0): dict(by_selection={"C": dict(dev_criterion=0.0)}),
             ("synth_NSP_r1_d0p3", "gru", "triangular", "lr1e3", 0): dict(by_selection={"C": dict(dev_criterion=0.0)})}
    assert set(A.syn_units(cells)) <= set(S.main_settings()) and ("NSP", 0.3) not in A.syn_units(cells)
