"""Run the cells of a grid, sequentially or in parallel over a list of GPUs. Finished cells are skipped.

    python -m coarse_tpp.runner RUN_DIR [--gpus 0,1] [--per-gpu 2] [--workers 4] [--only SUBSTRING] [--retries 1]

Each cell runs `python -m coarse_tpp.train RUN_DIR/configs/<cell_id>.json` in its own process with
CUDA_VISIBLE_DEVICES set to one GPU of the list (or empty without --gpus: CPU), and writes
RUN_DIR/cells/<cell_id>/result.json. A cell whose result is valid (or terminally invalid) is not started again,
so the command can be re-run after an interruption.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import results as R

ROOT = Path(__file__).resolve().parents[1]


def pending(run: Path, only: str | None = None):
    out = []
    for line in (run / "grid.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        if only and only not in m["cell_id"]:
            continue
        cfg = json.loads(Path(m["config"]).read_text())
        st, _, _ = R.load(cfg)
        if st in R.FINISHED or st in R.TERMINAL_BAD:
            continue
        out.append((m["cell_id"], m["config"]))
    return out


def run_cell(cfg_path: str, gpu: str, retries: int) -> int:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    rc = 1
    for _ in range(retries + 1):
        rc = subprocess.call([sys.executable, "-m", "coarse_tpp.train", cfg_path], env=env)
        if rc == 0:
            break
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--gpus", default="", help="comma-separated GPU indices; empty = CPU")
    ap.add_argument("--per-gpu", type=int, default=1, help="concurrent cells per GPU")
    ap.add_argument("--workers", type=int, default=1, help="concurrent cells on CPU (without --gpus)")
    ap.add_argument("--only", default=None, help="run only cells whose id contains this string")
    ap.add_argument("--retries", type=int, default=1)
    a = ap.parse_args(argv)
    run = Path(a.run).resolve()
    todo = pending(run, a.only)
    slots: queue.Queue = queue.Queue()
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    for g in gpus:
        for _ in range(a.per_gpu):
            slots.put(g)
    if not gpus:
        for _ in range(a.workers):
            slots.put("")
    n_slots = slots.qsize()
    print(f"{len(todo)} cells to run on {n_slots} slot(s)", flush=True)
    t0, failed = time.time(), []

    def job(item):
        cid, path = item
        g = slots.get()
        try:
            rc = run_cell(path, g, a.retries)
        finally:
            slots.put(g)
        print(f"[{time.time() - t0:8.0f}s] {cid}: {'ok' if rc == 0 else f'failed (exit {rc})'}", flush=True)
        if rc != 0:
            failed.append(cid)

    with ThreadPoolExecutor(max_workers=n_slots) as ex:
        list(ex.map(job, todo))
    print(f"done: {len(todo) - len(failed)} finished, {len(failed)} failed", flush=True)
    if failed:
        (run / "failed.txt").write_text("\n".join(failed) + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
