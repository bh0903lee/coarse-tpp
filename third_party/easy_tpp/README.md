# Vendored EasyTPP models

EasyTPP model code as distributed in the S2P2 repository
(https://github.com/UCIDataLab/state_space_point_process, a fork of
[EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess)), commit
`c3933240f16a22b43d80d09bda272475526ff24b` (`c393324`), Apache License 2.0 (see `LICENCE` and `NOTICE`).
It is used only by `coarse_tpp.rescore_intensity` to rebuild saved RMTPP, NHP, SAHP and AttNHP checkpoints.

The files `easy_tpp/model/torch_model/torch_*.py`, `LICENCE`, `NOTICE` and `easy_tpp_version.py` (`version.py` in
the source repository) are unmodified copies; their SHA-256 sums are listed in `SOURCE_SHA256.txt` (check with
`sha256sum -c SOURCE_SHA256.txt` in this directory).

The remaining files are not copies. `easy_tpp/utils/__init__.py` is a minimal import shim that provides the two
names the models import from `easy_tpp.utils` (`logger` and `set_device`), so that the original module and its
dependencies (omegaconf, tensorboard, datasets) are not needed. The package files `easy_tpp/__init__.py`,
`easy_tpp/model/__init__.py` and `easy_tpp/model/torch_model/__init__.py` are short replacements that import only
the four models above.
