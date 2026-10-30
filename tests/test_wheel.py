"""Exercise a built wheel without allowing imports from the source checkout.

Set COARSE_TPP_TEST_WHEEL to the wheel path when running this packaging check.
"""
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


def test_wheel_rebuilds_and_scores_vendored_intensity_models(tmp_path):
    wheel = os.environ.get("COARSE_TPP_TEST_WHEEL")
    if not wheel:
        pytest.skip("set COARSE_TPP_TEST_WHEEL to test a built wheel")
    installed = tmp_path / "installed"
    with zipfile.ZipFile(Path(wheel).resolve()) as archive:
        archive.extractall(installed)
    code = """
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from coarse_tpp import rescore_intensity as W

seqs = [(np.array([0., 1., 2.]), np.array([0., 1., 1.]), np.array([0, 1, 0]))]
for mid in W.MODELS:
    torch.manual_seed(0)
    cfg = {'model_config': dict(hidden_size=8, time_emb_size=4, num_layers=1,
        num_heads=1, num_event_types=2, num_event_types_pad=3, event_pad_index=2)}
    model = W.build_model(mid, cfg, torch.device('cpu'))
    scores = W.score_split(model, mid, seqs, 2, 0.1, 8, 'cpu')
    assert scores['n'] == 2
    assert math.isfinite(scores['floor_1e-6']) and math.isfinite(scores['interval']), mid
vendor = Path(W.__file__).resolve().parents[1] / 'third_party' / 'easy_tpp'
for line in (vendor / 'SOURCE_SHA256.txt').read_text().splitlines():
    digest, name = line.split(maxsplit=1)
    assert hashlib.sha256((vendor / name).read_bytes()).hexdigest() == digest, name
"""
    env = dict(os.environ, PYTHONPATH=str(installed), PYTHONDONTWRITEBYTECODE="1",
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path,
                            env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
