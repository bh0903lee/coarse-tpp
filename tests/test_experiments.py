"""Grid sizes and configs of the experiments."""
import json

import pytest

from coarse_tpp import experiments as X, synthetic as S


@pytest.fixture
def audit(tmp_path):
    names = set(X.RANK_DATASETS + X.INTENSITY_DATASETS + X.PROCESS_LOGS + X.TIED_REAL + X.MARK_DATASETS + X.COARSENED
                + X.RESPLIT + X.NEW_DATASETS + X.COARSENED_MORE + X.BOUNDS_DATASETS)
    p = tmp_path / "audit.json"
    p.write_text(json.dumps([dict(dataset=d, delta=1.0, tie_share=0.05) for d in sorted(names)]))
    return p


def n(exp, tmp_path, audit):
    return len(X.BUILDERS[exp](tmp_path, X.audit_table(audit) if exp in X.NEEDS_AUDIT else None))


def test_grid_sizes(tmp_path, audit):
    main = len(S.main_settings())
    assert main == 9
    want = {"synthetic": 7 * 5 * (1 + 5 * 7) * 2 * 2, "rank_change": 9 * 8 * 2 * 2 * 5,
            "intensity_heads": 6 * 5 * 2 * 2 * 5, "process_logs": 7 * 6 * 2 * 2 * 5,
            "delta_misspecification": main * 3 * 2 + 5 * 2 * 5 * 2 * 5, "iw_jitter": main * 5 * 6 * 2 * 2 + 5 * 6 * 2 * 5,
            "iw_jitter_large": 2 * 5 * 6 * 2 * 2, "inbin_smoothing": main * 5 * 5 * 2 * 2 + 5 * 5 * 2 * 5,
            "inbin_smoothing_seeds": 5 * 2 * 2 * 40, "marks": 7 * 6 * 2 * 5, "coarsened_real": 12 * 6 * 2 * 2 * 5,
            "resplit": 8 * 5 * 2 * 5, "new_datasets": 3 * 5 * 2 * 5, "floor_convention": 21 * 2 * 2 * 10,
            "component_bounds": 3 * 3 * 5 * 2 * 5, "triangular_coarsened": 12 * 2 * 2 * 5,
            "triangular_jitter": 6 * 2 * 3 * 2 * 5 + main * 5 * 3 * 2 * 2, "coarsened_more": 6 * 2 * 7 * 2 * 5,
            "triangular_recorded": 9 * 2 * 2 * 5 + 7 * 2 * 2 * 5}
    for exp, k in want.items():
        assert n(exp, tmp_path, audit) == k, exp
    syn = [c for c in X.cells_synthetic(tmp_path) if c["arm"] == "triangular"]
    assert len(syn) == 700
    cb = X.cells_component_bounds(tmp_path, X.audit_table(audit))
    both = [c for c in cb if c["arm"].endswith("sigmin2_mumin1e-4")]
    assert both and all(c["sigma_min"] == 2.0 and c["mu_min"] == X.MU_MIN and c["treatment"] in ("floor", "jitter_gap", "interval") for c in both)
    assert all(("sigma_min" in c) == ("sigmin2" in c["arm"]) and ("mu_min" in c) == ("mumin" in c["arm"]) for c in cb)
    tj = X.cells_triangular_jitter(tmp_path, X.audit_table(audit))
    assert all(c["treatment"] == "iw_jitter_triangular" and c["iw_samples"] in X.TRI_IW_S for c in tj)
    assert {c["seed"] for c in tj if c["dataset"].startswith("synth_")} == set(X.SMOOTH_SYN_SEEDS)
    assert {X.coarsened_r(c["dataset"]) for c in tj if c["dataset"].startswith("coarsened_")} == set(X.TRI_IW_R)
    assert X.COARSENED_MORE == ("coarsened_earthquake_r0p03", "coarsened_earthquake_r0p1", "coarsened_earthquake_r0p3",
                                "coarsened_earthquake_r1", "coarsened_taobao_r0p3", "coarsened_taobao_r1")
    tr = X.cells_triangular_recorded(tmp_path, X.audit_table(audit))
    assert all(c["encoder"].get("marks", False) == c["dataset"].startswith("bpi_") for c in tr)


def test_configs_and_immutability(tmp_path, audit):
    cs = X.write("rank_change", tmp_path / "run", audit)
    ids = [c["cell_id"] for c in cs]
    assert len(set(ids)) == len(ids)
    c = next(x for x in cs if x["arm"] == "floor_1e-10")
    assert c["treatment"] == "floor" and c["eps"] == 1e-10 and c["delta"] == c["delta_model"] == 1.0
    assert X.cells_intensity_heads(tmp_path, X.audit_table(audit))[0]["head"] in ("rmtpp", "thp")
    m = [x for x in X.cells_delta_misspecification(tmp_path, X.audit_table(audit)) if x["arm"] == "interval_dx4"]
    assert m and all(x["delta_model"] == 4 * x["delta"] for x in m)
    iw = [x for x in X.cells_iw_jitter(tmp_path, X.audit_table(audit)) if x["arm"] == "iw_jitter_S16"]
    assert iw and all(x["iw_samples"] == 16 and x["treatment"] == "iw_jitter" for x in iw)
    X.write("rank_change", tmp_path / "run", audit)                       # same configs: allowed
    p = tmp_path / "run" / "configs" / f"{ids[0]}.json"
    p.write_text(p.read_text().replace('"seed": 0', '"seed": 99'))
    with pytest.raises(SystemExit):
        X.write("rank_change", tmp_path / "run", audit)
