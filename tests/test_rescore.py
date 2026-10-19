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
