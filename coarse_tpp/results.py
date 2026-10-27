"""Validity of a stored cell result, shared by the trainer, the runner and the reports.

States: ok (trained to early stopping or the epoch budget, metrics finite); ok_walltime (stopped by the wall-clock
limit; reported separately and excluded from comparisons); invalid_metrics (a required metric is not finite;
terminal, a rerun of the same deterministic config gives the same value). Anything else (missing or unreadable
file, other cell, different config, wrong event count) is invalid and the cell is not finished.
"""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

REQUIRED = ("ll",)
FINISHED = ("ok", "ok_walltime")
TERMINAL_BAD = ("invalid_metrics",)


def checkpoint_identity(cfg, res, selection):
    """Identity of a current checkpoint and its configuration; None if unavailable."""
    ck = Path(cfg["out_dir"]) / f"best_{selection}.pt"
    selected = res.get("by_selection", {}).get(selection)
    if not ck.exists() or selected is None:
        return None
    return dict(config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                ckpt_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),
                selected_epoch=selected.get("selected_epoch"), selection=selection)


def nonfinite_metrics(res: dict) -> list:
    bad = []
    for split in ("test", "dev"):
        for k in REQUIRED:
            v = (res.get(split) or {}).get(k)
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                bad.append(f"{split}.{k}")
    return bad


def check(res: dict, cfg: dict) -> tuple[str, str]:
    """(state, reason); state is one of FINISHED, TERMINAL_BAD or "invalid"."""
    if res.get("cell_id") != cfg.get("cell_id"):
        return "invalid", f"cell_id {res.get('cell_id')!r} != {cfg.get('cell_id')!r}"
    if res.get("config") != cfg:
        return "invalid", "stored config differs from the cell config"
    st = res.get("status")
    if st in TERMINAL_BAD:
        return st, "non-finite metrics: " + ",".join(res.get("nonfinite", []))
    if st not in FINISHED:
        return "invalid", f"unexpected status {st!r}"
    bad = nonfinite_metrics(res)
    if bad:
        return "invalid", "non-finite metrics: " + ",".join(bad)
    exp = res.get("expected_test_targets")
    if exp is None or (res.get("test") or {}).get("n_events") != exp:
        return "invalid", f"test n_events {(res.get('test') or {}).get('n_events')} != expected {exp}"
    return st, ""


def load(cfg: dict, out_dir: Path | None = None) -> tuple[str, str, dict | None]:
    """(state, reason, result) for a cell config; state "missing" if there is no result."""
    p = Path(out_dir or cfg["out_dir"]) / "result.json"
    if not p.exists():
        return "missing", "", None
    try:
        res = json.loads(p.read_text(), parse_constant=_reject_constant)
    except Exception as e:
        return "invalid", f"unreadable result: {e}", None
    st, why = check(res, cfg)
    return st, why, res


def _reject_constant(name):
    raise ValueError(f"non-standard JSON constant {name}")


def sanitize(obj):
    """Replace non-finite floats by None so the result can be written as strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj
