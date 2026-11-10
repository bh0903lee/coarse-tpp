# Third-party notices

The directory third_party/easy_tpp contains EasyTPP model code as distributed
in the S2P2 repository (https://github.com/UCIDataLab/state_space_point_process,
a fork of EasyTPP), commit c393324, licensed under the Apache License,
Version 2.0 (see third_party/easy_tpp/LICENCE and third_party/easy_tpp/NOTICE).
The model files are unmodified; the minimal import shim
third_party/easy_tpp/easy_tpp/utils/__init__.py and the short package
__init__ files are not copies (see third_party/easy_tpp/README.md). The files
in third_party/easy_tpp are not covered by the MIT License of this repository (LICENSE).

The directory third_party/easy_tpp_train contains the complete EasyTPP package
of the same commit (c393324 of the S2P2 repository), licensed under the Apache
License, Version 2.0 (see third_party/easy_tpp_train/LICENCE and
third_party/easy_tpp_train/NOTICE). It is used to train, rebuild and score the
retrained intensity-based models. 39 of its package files are unmodified
(SOURCE_SHA256.txt); 16 files carry the patches listed, with the SHA-256 of
the source and of the patched file, in third_party/easy_tpp_train/PATCHES.md,
and each patch is marked "Patch" in the source. The files in
third_party/easy_tpp_train are not covered by the MIT License of this
repository (LICENSE).

Parts of the state-space layers of S2P2 in third_party/easy_tpp_train/easy_tpp/ssm
are taken or modified from S5 (https://github.com/lindermanlab/S5), as stated in
the source: discretize_zoh and apply_ssm in ssm/ssm_util.py (S5 commit
3c18fdb6b06414da35e77b94b9cd855f6a95ef17, s5/ssm.py), init_VinvB in
ssm/initializers.py (S5 commit 52cc7e22d6963459ad99a8674e4d3cfb0a480008,
s5/ssm.py), and the initialization of C and the "half_glu" activation in
ssm/models.py, which cite the same S5 files. S5 is distributed under the MIT
License, Copyright (c) 2022 Linderman Lab; its copyright and permission notice,
identical at both commits, is reproduced in third_party/easy_tpp_train/LICENSE_S5.

The data sets are not distributed with this repository, except the simulated data sets in
`data/synthetic_extra/` (MIT License); the sources and terms of the others are listed in `data/README.md`.
