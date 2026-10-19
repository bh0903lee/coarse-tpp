#!/usr/bin/env python
"""Run the cells of a grid sequentially or over GPUs, skipping finished cells (see coarse_tpp.runner)."""
import _bootstrap  # noqa: F401
from coarse_tpp.runner import main

if __name__ == "__main__":
    main()
