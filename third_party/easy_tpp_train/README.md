# Vendored EasyTPP (training copy)

EasyTPP (https://github.com/ant-research/EasyTemporalPointProcess) as distributed in the S2P2 repository
(https://github.com/UCIDataLab/state_space_point_process), commit `c3933240f16a22b43d80d09bda272475526ff24b`
(`c393324`), package version 0.1.3 (`easy_tpp_version.py`), Apache License 2.0 (`LICENCE`, `NOTICE`).

This directory holds the complete `easy_tpp` package of that commit, so that the retrained intensity-based models of
Section 5.2 and Supplementary Tables S3-S5 can be trained (`coarse_tpp.easytpp_train`), rebuilt and scored
(`coarse_tpp.rescore_retrained`, `coarse_tpp.intensity_exceedance`) with the code they were trained with. Runtime
dependencies are those of `coarse_tpp` (torch, numpy, PyYAML); `tensorboard` is optional and `optuna` is needed only by
`easy_tpp.hpo`, which is not used. The unmodified model subset in `third_party/easy_tpp` is unchanged and is still used
by `coarse_tpp.rescore_intensity`; the two copies have the same top-level package name, and `coarse_tpp.easytpp`
switches between them (`activate`).

- `LICENCE`, `NOTICE`, `easy_tpp_version.py` (`version.py` in the source repository) and 39 files of `easy_tpp/` are
  unmodified copies; their SHA-256 sums are in `SOURCE_SHA256.txt` (check with `sha256sum -c SOURCE_SHA256.txt` in
  this directory).
- 16 files carry the patches listed in `PATCHES.md` (marked `Patch` in the source), with the SHA-256 of the source
  file and of the patched file. None of the patches changes a forward computation of a model.
