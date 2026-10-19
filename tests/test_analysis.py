"""Summary statistics on fabricated results."""
import json
import math

import numpy as np

from coarse_tpp import analysis as A, experiments as X


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
    with pytest.raises(SystemExit, match="rescore_saved"):
        A.run_report("floor_convention", tmp_path)
