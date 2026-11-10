import os
import sys
from pathlib import Path

# The test suite runs on the CPU, also where a GPU is visible: hide CUDA before torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    """Report CUDA as unavailable in every test, so that code choosing a device selects the CPU."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
