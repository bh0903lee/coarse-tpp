"""Report populations and checkpoint-bound score caches."""
import json
from pathlib import Path

import torch

from coarse_tpp import alarms as K, analysis as A, data as D, experiments as X, latent_score as LS
from coarse_tpp import rescore as RS, synthetic as S, train as TR
from test_analysis import fabricate
from test_predictive import toy_run


def test_predictive_cache_follows_current_checkpoint(toy_run):
    from coarse_tpp import predictive as P

    run, audit = toy_run
    P.score(run)
    res = P.selected_cells(run)[0]
    path = run / "predictive" / "cells" / f"{res['cell_id']}.json"
    row = json.loads(path.read_text()); row["mae_median"] = -777.0
    path.write_text(json.dumps(row))
    ck = Path(res["config"]["out_dir"]) / "best_C.pt"
    state = torch.load(ck, weights_only=True)
    state["head.linear.bias"] += 0.01
    torch.save(state, ck)
    lines = P.report(run, audit, arms=("floor_1e-6", "interval"), n_seeds=2)
    assert any("scored: 3; missing: 1" in line for line in lines)
    P.score(run)
    updated = json.loads(path.read_text())
    assert updated["mae_median"] != -777.0
    assert updated["quarantined"]


def test_intensity_cache_follows_current_checkpoint(tmp_path):
    import pickle
    import yaml
    from coarse_tpp import rescore_intensity as W
    from test_rescore_intensity import cfg, seqs

    old = D.data_dir()
    D.set_data_dir(tmp_path / "data")
    try:
        data = tmp_path / "data" / "toy"; data.mkdir(parents=True)
        events = [[dict(time_since_start=float(t), time_since_last_event=float(dt), type_event=int(k))
                   for t, dt, k in zip(*s)] for s in seqs()]
        for split in ("train", "dev", "test"):
            (data / f"{split}.pkl").write_bytes(pickle.dumps({split: events, "dim_process": 2}))
        ckpt, out = tmp_path / "ckpt", tmp_path / "scores"
        cand = ckpt / "toy" / "RMTPP" / "run1"; cand.mkdir(parents=True)
        config = cfg("RMTPP")
        (cand / "config.yaml").write_text(yaml.safe_dump(config))
        model = W.build_model("RMTPP", config, torch.device("cpu"))
        torch.save(model.state_dict(), cand / "saved_model.pt")
        audit = tmp_path / "audit.json"
        audit.write_text(json.dumps([dict(dataset="toy", delta=0.1)]))
        W.score(ckpt, out, datasets=["toy"], nodes=8, audit_path=audit)
        path = out / "cells" / "toy__RMTPP__run1.json"
        row = json.loads(path.read_text())
        original = row["test"]["floor_1e-6"]
        row["test"]["floor_1e-6"] = -777.0; path.write_text(json.dumps(row))
        with torch.no_grad():
            next(model.parameters()).add_(0.1)
        torch.save(model.state_dict(), cand / "saved_model.pt")
        assert W.select([row], ckpt) == {}
        assert any("stale or unidentified: 1" in line for line in W.report(ckpt, out))
        W.score(ckpt, out, datasets=["toy"], nodes=8, audit_path=audit)
        updated = json.loads(path.read_text())
        assert updated["test"]["floor_1e-6"] != -777.0
        assert updated["test"]["floor_1e-6"] != original
        assert len(W.select([updated], ckpt)) == 1
        config["audit_note"] = "config changed"
        (cand / "config.yaml").write_text(yaml.safe_dump(config))
        assert W.select([updated], ckpt) == {}
        W.score(ckpt, out, datasets=["toy"], nodes=8, audit_path=audit)
        assert len(W.select([json.loads(path.read_text())], ckpt)) == 1
    finally:
        D.set_data_dir(old)


def test_missing_planned_group_stays_in_the_family(tmp_path):
    cells = []
    for ds in ("email_eu", "volcano"):
        for arm in ("floor_1e-10", "interval"):
            for lr in X.LRS:
                for s in range(5):
                    c = X.cell("rank_change", tmp_path, ds, arm, s, lr, 1.0)
                    v = -1.0 + (0.1 if arm == "interval" else 0) + 0.001 * s
                    cells.append(dict(cell_id=c["cell_id"], config=c, dev=v, test={"C": v, "floor_1e-6": v}))
    fabricate(tmp_path, cells)
    for c in cells:
        if c["config"]["dataset"] == "volcano":
            (tmp_path / "cells" / c["cell_id"] / "result.json").unlink()
    lines, summary = A.report_real(tmp_path, "test", ("floor_1e-10", "interval"),
                                   {d: dict(tie_share=0.2) for d in ("email_eu", "volcano")})
    assert summary["n_family"] == 2
    assert summary["preservation"]["not judged"] == 1
    assert any("volcano" in x and "pending" in x for x in lines)


def test_rescore_and_latent_cache_follow_current_checkpoint(tmp_path):
    old = D.data_dir()
    D.set_data_dir(tmp_path / "data")
    try:
        S.generate("SP", 1, n_total=15000)
        run = tmp_path / "run"
        c = X.cell("synthetic", run, "synth_SP_r1_d0p3", "interval", 0, "lr1e3", 0.3)
        c.update(cpu=True, optim=dict(c["optim"], max_epochs=1))
        TR.run(c); X.write("synthetic", run, cells=[c])
        RS.score(run, selection="C", audit_path=tmp_path / "none.json")
        LS.main([str(run)])
        rp, lp = run / "rescore.jsonl", run / "latent_scores.jsonl"
        rr, lr = json.loads(rp.read_text()), json.loads(lp.read_text())
        rr["test"]["interval"] = -777.0; lr["test"]["B"] = -777.0
        rp.write_text(json.dumps(rr) + "\n"); lp.write_text(json.dumps(lr) + "\n")
        ck = Path(c["out_dir"]) / "best_C.pt"
        state = torch.load(ck, weights_only=True)
        state["head.linear.bias"] = state["head.linear.bias"] + 0.01
        torch.save(state, ck)
        assert RS.load_rows(run) == []
        assert A.latent_rows([lp]) == {}
        lines, _ = A.report_coarsened(run, {}, datasets=(c["dataset"],), arms=("interval",))
        assert any("0 of 1 cells" in line for line in lines)
        RS.score(run, selection="C", audit_path=tmp_path / "none.json")
        LS.main([str(run)])
        assert json.loads(rp.read_text())["test"]["interval"] != -777.0
        assert json.loads(lp.read_text())["test"]["B"] != -777.0
        result = Path(c["out_dir"]) / "result.json"
        r = json.loads(result.read_text()); r["status"] = "invalid_metrics"
        result.write_text(json.dumps(r))
        assert RS.load_rows(run) == []
        RS.score(run, selection="C", audit_path=tmp_path / "none.json")
        LS.main([str(run)])
        assert not rp.read_text().strip() and not lp.read_text().strip()
    finally:
        D.set_data_dir(old)


def test_rt_cache_must_match_the_verified_checkpoint(tmp_path):
    (tmp_path / "cells").mkdir(); (tmp_path / "rt").mkdir()
    results = {}
    for s in K.PLANNED_SEEDS:
        cid = f"c{s}"; cd = tmp_path / cid; cd.mkdir()
        ck = cd / "best_C.pt"; ck.write_bytes(b"current checkpoint")
        results[s] = dict(cell_id=cid, config=dict(out_dir=str(cd)), by_selection=dict(C=dict(selected_epoch=2)))
        rec = dict(version=K.VERSION, quarantined_sels=[], by_sel=dict(
            C=dict(missing=False, quarantined=False, selected_epoch=2, ckpt_sha256=K.ckpt_sha(ck)), own=dict(missing=True)))
        (tmp_path / "cells" / f"{cid}.json").write_text(json.dumps(rec))
        rt = dict(version=K.VERSION, cell_id=cid, complete=True, sel="C", selected_epoch=1, ckpt_sha256="old", mae_median=123)
        (tmp_path / "rt" / f"{cid}__C.json").write_text(json.dumps(rt))
    rows, pending = K.rt_collect(tmp_path, results, "C")
    assert rows == [] and len(pending) == len(K.PLANNED_SEEDS)
    for s, r in results.items():
        p = tmp_path / "rt" / f"{r['cell_id']}__C.json"
        j = json.loads(p.read_text()); j.update(selected_epoch=2, ckpt_sha256=K.ckpt_sha(Path(r["config"]["out_dir"]) / "best_C.pt"))
        p.write_text(json.dumps(j))
    rows, pending = K.rt_collect(tmp_path, results, "C")
    assert len(rows) == len(K.PLANNED_SEEDS) and not pending


def test_intensity_family_keeps_underseeded_planned_groups(tmp_path):
    cells = []
    for ds, seeds in (("email_eu", range(5)), ("volcano", range(2))):
        for arm in X.INTENSITY_ARMS:
            for lr in X.LRS:
                for s in seeds:
                    c = X.cell("intensity_heads", tmp_path, ds, arm, s, lr, 1.0, head="rmtpp")
                    v = -1.0 + (0.1 if arm == "interval" else 0) + 0.001 * s
                    cells.append(dict(cell_id=c["cell_id"], config=c, dev=v, test={"C": v, "floor_1e-6": v}))
    fabricate(tmp_path, cells)
    lines, summary = A.report_intensity(tmp_path, {d: dict(tie_share=0.2) for d in ("email_eu", "volcano")})
    assert summary["n_family"] == 2 and summary["rmtpp_better_all"] == "1/2"
    assert any("volcano" in x and "pending" in x for x in lines)


def test_deposited_rescore_table_without_grid_remains_reportable(tmp_path):
    scores = {k: -1.0 for k in RS.FLOORS}
    test = dict(scores, interval=-2.0, n=10, n_tie=2)
    row = dict(cell_id="deposited", arm="floor", dataset="toy", seed=0, lr_tag="lr1e3", dev=scores, test=test)
    (tmp_path / "rescore.jsonl").write_text(json.dumps(row) + "\n")
    assert RS.load_rows(tmp_path) == [row]
    lines = RS.report(tmp_path)
    assert any("no training grid" in line for line in lines)
    assert any("| toy |" in line for line in lines)


def test_interrupted_scoring_resumes_from_tmp(tmp_path, monkeypatch):
    old = D.data_dir()
    D.set_data_dir(tmp_path / "data")
    try:
        S.generate("SP", 1, n_total=15000)
        run = tmp_path / "run"
        c = X.cell("synthetic", run, "synth_SP_r1_d0p3", "interval", 0, "lr1e3", 0.3)
        c.update(cpu=True, optim=dict(c["optim"], max_epochs=1))
        TR.run(c); X.write("synthetic", run, cells=[c])
        RS.score(run, selection="C", audit_path=tmp_path / "none.json")
        LS.main([str(run)])
        rp, lp = run / "rescore.jsonl", run / "latent_scores.jsonl"
        rp.replace(rp.with_suffix(".tmp")); lp.replace(lp.with_suffix(".tmp"))   # as left by an interrupted call

        def fail(*a, **k):
            raise AssertionError("a row of the interrupted call was scored again")
        monkeypatch.setattr(RS, "score_cell", fail)
        monkeypatch.setattr(LS, "build", fail)
        RS.score(run, selection="C", audit_path=tmp_path / "none.json")
        LS.main([str(run)])
        assert len(RS.load_rows(run)) == 1 and len(A.latent_rows([lp])) == 1
    finally:
        D.set_data_dir(old)

