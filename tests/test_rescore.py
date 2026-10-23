"""Re-scoring resumes only with the same selection and delta."""
import json

import pytest

from coarse_tpp import rescore as RS


def test_rescore_refuses_rows_of_another_selection(tmp_path):
    run = tmp_path / "run"; run.mkdir()
    cfg = dict(dataset="email_eu", delta=1.0, out_dir=str(run / "cells" / "c0"), head="mix")
    (run / "c0.json").write_text(json.dumps(cfg))
    (run / "grid.jsonl").write_text(json.dumps(dict(cell_id="c0", config=str(run / "c0.json"))) + "\n")
    (run / "rescore.jsonl").write_text(json.dumps(dict(cell_id="c0", selection="C", delta=1.0)) + "\n")
    RS.score(run, selection="C", audit_path=tmp_path / "none.json")          # same setting: resumes, nothing to do
    with pytest.raises(SystemExit, match="selection C"):
        RS.score(run, selection="own", audit_path=tmp_path / "none.json")


def test_rescore_skips_results_that_fail_the_validity_rule(tmp_path, capsys):
    """A cell whose result is not status ok (or does not match its config) is skipped, as in the reports."""
    run = tmp_path / "run"; (run / "cells" / "c0").mkdir(parents=True)
    cfg = dict(cell_id="c0", dataset="email_eu", delta=1.0, out_dir=str(run / "cells" / "c0"), head="mix", arm="x", seed=0, lr_tag="lr1e3")
    (run / "c0.json").write_text(json.dumps(cfg))
    (run / "grid.jsonl").write_text(json.dumps(dict(cell_id="c0", config=str(run / "c0.json"))) + "\n")
    (run / "cells" / "c0" / "best_C.pt").write_bytes(b"x")
    for status in ("ok_walltime", "invalid_metrics"):
        res = dict(cell_id="c0", status=status, config=cfg, test=dict(ll=-1.0, n_events=5), dev=dict(ll=-1.0), expected_test_targets=5)
        (run / "cells" / "c0" / "result.json").write_text(json.dumps(res))
        RS.score(run, selection="C", audit_path=tmp_path / "none.json")
        assert "skipped" in capsys.readouterr().out
    res = dict(cell_id="c0", status="ok", config=dict(cfg, seed=1), test=dict(ll=-1.0, n_events=5), dev=dict(ll=-1.0), expected_test_targets=5)
    (run / "cells" / "c0" / "result.json").write_text(json.dumps(res))
    RS.score(run, selection="C", audit_path=tmp_path / "none.json")
    assert "stored config differs" in capsys.readouterr().out
    assert not (run / "rescore.jsonl").read_text().strip()
