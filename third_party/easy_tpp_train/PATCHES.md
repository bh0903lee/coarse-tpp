# Patches of the vendored EasyTPP training copy

Each patch is marked `Patch` in the source. The models of Section 5.2 and Supplementary Tables S3-S5 were trained
with exactly this code. No patch changes the forward computation of a model: the intensities, the cumulative hazard
of FullyNN and the mixture of IntensityFree are those of the source files, and the files differ from the source only
in the training objective, the training loop and the data pipeline listed here.

| File | SHA-256 of the source file | SHA-256 of the patched file |
|---|---|---|
| `easy_tpp/config_factory/runner_config.py` | `7690df0ae6b69313f9c0f60c8c7f16fe63f2e879388abeef104dbc408f0686dc` | `bf120f8e9e56f68dacfc603e98e3fd5b942c5089b1b991dd9ab4314fb6639293` |
| `easy_tpp/model/torch_model/torch_attnhp.py` | `efce8d0fb5b8fcf9818544bbb3e97047892a9f98a4959eb24ea8bfdbbb6431b9` | `bcf4ad4c2b072f6e952295c32bfc37b5e134a0bcd2d0e7ba3b47d06df328cbc3` |
| `easy_tpp/model/torch_model/torch_basemodel.py` | `c6acec407edbb96938eb0a7061f4e7d5d45b7a882876ee2ac955d405591e32ae` | `d84c3bb254b552a664a71b79464f8b92848604f3712b8da22ac4c1d5a2ce96c8` |
| `easy_tpp/model/torch_model/torch_fullynn.py` | `5747b9274f64cfeba8e0c57327868efe483881403e78520ecc58ed88ed2bccd6` | `80084a16ee7a552a7cdc7bbcabb7b63eeae917e2cb7272da8ca8e1caf984a664` |
| `easy_tpp/model/torch_model/torch_intensity_free.py` | `46757b4b45dca46a7e61fa3f5338020a5b3aa8ecc5b41e086220d821c7cd433b` | `c23fd3bd6a502f96a01598b5376e778e7775282260db4ee65bd299b12bfd8fd1` |
| `easy_tpp/model/torch_model/torch_mamba.py` | `ae6b5d9f3c259e2ef8a18cee87334916c99fd10ed134e5926398e68509279373` | `d627b0705b45af1e6f6969027c7e0cae666f16b8f79a1b1ac50331c6f1c1dc9f` |
| `easy_tpp/model/torch_model/torch_nhp.py` | `fdb7646191bc921a96591739596a3777fa58ea376d0dfc0cf89940f7303edb66` | `aa31e0a16d5b0246375c1e04a37a30140a06fe1b40adc82f58b76901a67a41dc` |
| `easy_tpp/model/torch_model/torch_ode_tpp.py` | `96a67938c267d4a7a7a905aa1444ced43291be81c236432b6d7449270ea6290e` | `fdd56f963b1a1ff81be2306f2649176055114c5963029aae05c1460c1d574b89` |
| `easy_tpp/model/torch_model/torch_rmtpp.py` | `b14f260019b55a5d583d1c349c04481add7bdbd42305d975309d06cb3b702fe9` | `0c7ad9ae48d1fd4deedd2a4d8997eabac095d2ac16c1a1399cb4505f28f049b9` |
| `easy_tpp/model/torch_model/torch_s2p2.py` | `a2f34bffedcf19b83245d59ff8d6f7c760865345dcffd32b2248f5b324f40eb3` | `655017b3e83ba0d406797af534f5e4a15923a3d2d0423411ea432ea5af642b04` |
| `easy_tpp/model/torch_model/torch_sahp.py` | `f4d719a8896fe92a840728298eaf34dab298e0a78f19f2e434970260d47a5244` | `4f7da6961d720c7c09a6437a6d45c6b73bc67239d258ae52aaee3d320881727f` |
| `easy_tpp/model/torch_model/torch_thp.py` | `9887ad95e1ffc2471650122fd7fc1bf7e2ca209c5af4040b57b2944d1aede90f` | `1e589a168dd2d3f75abbc15b516dfe56619504382f9c207bf50d9c295efeec24` |
| `easy_tpp/preprocess/data_loader.py` | `fe04fcf5a55fb919bb95d28a15b22ec723ed0613ec00be27f0e81c2588355bed` | `74fcaba1b82795e80806b5a5c9f7d006c9cb09f22661ff4dad314f583e8afdb0` |
| `easy_tpp/preprocess/dataset.py` | `a58c3b0e677c6260fe41c5a84fc5d9b0920466be2251965a680a364ba45d4d9d` | `c2f40fc590484a1fe81bfa9200d6dad56a311abfea45a9d7c34d1238bdc4f398` |
| `easy_tpp/preprocess/event_tokenizer.py` | `1c6f77497bfe194d9345572997959aac7865e405378b6f26a835fff8087251ad` | `e0c87c0f801635f2c210bd47a838449ca97d8984fe051f484a04b659942da53c` |
| `easy_tpp/torch_wrapper.py` | `c9c31da755a6573749b61170442053d24fb3e3fffadef8218d25c8a540d2611f` | `32a5c4da57585dfc2bdb1fe606de2e634488306a260a666fd461131f1e2e5b2c` |

## Training objective

1. `torch_basemodel.py` gains `mark_loss_on()`, true only when the environment variable `EASYTPP_MARK_LOSS` is `1`.
   In `loglike_loss` of `torch_rmtpp.py`, `torch_nhp.py`, `torch_sahp.py`, `torch_attnhp.py`, `torch_thp.py`,
   `torch_s2p2.py` and `torch_mamba.py` the loss is `-(time_ll_pos - non_event_ll).sum()` (log total intensity at the
   event minus the compensator, without the mark term) unless `mark_loss_on()`, in which case it is the source's
   `-(event_ll - non_event_ll).sum()`; `torch_intensity_free.py` likewise drops the categorical mark term unless
   `mark_loss_on()`. On single-mark data the two objectives coincide. The time-only experiments (Supplementary Tables
   S3 and S5) leave the variable unset; the joint experiment (Table S4) sets it to `1`.
2. `torch_fullynn.py`, `loglike_loss`: the source returned `(loss, num_events)`, while the training wrapper of this
   version expects five values, so the model could not be trained. The patched method returns
   `(loss, num_events, mark_ll, time_ll, None)` with the objective of item 1; the event term is the log of the total
   intensity `sum_k dPhi_k / dtau` (each term with the float32 eps of the base class), the compensator is the network
   output `Phi(tau)` itself, as in the source.
3. `torch_ode_tpp.py`: `NeuralODE.forward` differentiates the forward RK4 solve directly (the source's adjoint
   backward reshapes a scalar into each parameter shape and fails for every multi-element parameter; the source path
   is kept as `forward_adjoint_upstream`); `loglike_loss` returns the five values of item 2 with the objective of
   item 1 (the source unpacked three values from `compute_loglikelihood`).

## Model construction

4. `torch_intensity_free.py`, `__init__`: `model_config.get(...)` raises `KeyError` for an absent key with the
   configuration class of this version, so the model could not be constructed; the same two attributes
   (`mean_log_inter_time`, `std_log_inter_time`, defaults 0 and 1) are read with `getattr`. The runner sets them from
   the train split (`TPPDataset.get_dt_stats`).
5. `config_factory/runner_config.py`: the source compares the backend with `Backend.TF`, which the enum of this version
   does not define, so building an ODETPP runner raised `AttributeError`; `getattr(Backend, "TF", object())` keeps the
   TensorFlow-only check without failing.

## Data pipeline and training loop

6. `preprocess/dataset.py`, `get_dt_stats`: sequences with a single event (no gap) are skipped; the source failed on
   the empty slice (Volcano has such sequences).
7. `preprocess/event_tokenizer.py`: when no padding is requested, list inputs are converted to arrays (float32 times
   and gaps, int64 types) as in the padded branch, and `make_pad_sequence` accepts lists.
8. `preprocess/data_loader.py`: the number of loader worker processes is read from `EASYTPP_WORKERS` (default 2;
   `coarse_tpp.easytpp_train` sets 0, as in the paper's runs), with pinned memory. The sampler runs in the main
   process, so the batch order for a given seed does not depend on the number of workers.
9. `torch_wrapper.py`: tensorboard is imported only when installed (it is used only with `use_tfb`, which is off). With
   `EASYTPP_MICROBATCH=k > 1`, a batch is evaluated in `k` row chunks: in training, the gradients of the chunks'
   summed losses are accumulated and divided by the batch's event count before clipping and the optimizer step (the
   same gradient as for the whole batch); in validation the sums are added. Unset or 1: the source path. It was used,
   with `k = 4`, only for AttNHP on CollegeMsg (memory).
