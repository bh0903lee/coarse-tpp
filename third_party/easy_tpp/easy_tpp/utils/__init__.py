"""Shim for the parts of easy_tpp.utils that the vendored models import (the original module pulls in omegaconf,
tensorboard and datasets, which are not needed here)."""
import logging

import torch

logger = logging.getLogger("easy_tpp")


def set_device(gpu=-1):
    """Same semantics as easy_tpp.utils.torch_utils.set_device without the MPS branch."""
    return torch.device("cuda:" + str(gpu)) if gpu >= 0 and torch.cuda.is_available() else torch.device("cpu")
