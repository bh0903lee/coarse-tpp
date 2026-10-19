# coarse-tpp

Code for the paper **"Ties, Time Grids and the Likelihood: Evaluating Neural Temporal Point Processes on Coarsely
Recorded Data"**.

Event data are often recorded on a time grid (seconds, days, ...), so that several events share a time stamp and
gaps are multiples of a resolution Δ. This repository lets you

1. **audit** the recording resolution of an event-sequence data set (tie share, lattice detection, hidden ties, the
   assigned Δ; Section 2 and Appendix B of the paper);
2. **score** any model by the **interval likelihood** of the recorded cell, `log P(τ̃ ≤ s < τ̃ + Δ)`, next to the
   conventional ε-floor scores, including the conversion rules (readouts) of models trained on jittered or
   zero-inflated data;
3. **train** a GRU or Transformer encoder with a log-normal mixture head, or RMTPP / THP intensity heads, under every
   treatment of ties studied in the paper (floors, jitter of gaps or times, multi-sample jitter, a point mass at zero,
   the interval likelihood, a triangular-kernel likelihood, interval / jitter mixtures);
4. **generate** the synthetic processes with known latent times (SP, SR, SC, H1, H2, and two non-stationary ones) and
   their coarsened recordings;
5. **prepare** the derived data sets (quantized ContTime, coarsened real data, new random splits, MOOC / Wikipedia,
   business process logs);
6. **run** the experiment grids of the paper and compute their summary statistics.

## Installation

Python 3.10. CPU is enough for the quick start; the full grids were run on GPUs.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch 2.6.0; pick the CUDA or CPU wheel for your machine
pip install -e .                         # optional: installs the package coarse_tpp
python -m pytest -q tests                # 72 tests, CPU, about a minute
```

The scripts in `scripts/` work from the repository root without installation. Data live in `./data` (or the
directory given by the environment variable `COARSE_TPP_DATA`); see [data/README.md](data/README.md).

## Quick start

**Audit a data set.** Every data set is a directory `data/<name>/{train,dev,test}.pkl` in the NHP / EasyTPP layout
(`{split: [sequence], "dim_process": n}`, events `{"time_since_start", "time_since_last_event", "type_event"}`).

```bash
python scripts/audit.py --datasets email_eu volcano --out results/audit     # writes audit.json and audit.md
python scripts/score.py audit --pkl any_file.pkl                             # the same rule on any pickle file(s)
```

**Score your own model.** The interval score needs only the model's distribution function of the time since the
last recorded event, at the two edges of the recorded cell:

```python
import numpy as np
from coarse_tpp import score

info = score.detect_resolution(gaps)                       # recorded gaps, all splits pooled, first events excluded
ll, tie = score.interval_loglik_from_cdf(tau, F_lo, F_hi)  # F_lo = F(tau~) (0 for ties), F_hi = F(tau~ + delta)
print(info["delta"], score.summarize(ll, tie))
```

or from a CSV with columns `tau, F_lo, F_hi`: `python scripts/score.py score --csv preds.csv --delta 1`. Models
built from `coarse_tpp.lik` (log-normal mixture, RMTPP, THP) can be scored directly with
`score.interval_loglik(law, tau, delta)` and `score.floor_loglik(law, tau, eps)`.

**Train and score one cell (CPU smoke test, under a minute).**

```bash
export COARSE_TPP_DATA="$PWD/data_smoke"   # keep the small realization apart from data/synthetic
python scripts/generate_synthetic.py --processes SP --realizations 1 --n-total 15000   # a small realization
python scripts/make_grid.py synthetic --out runs/smoke --datasets synth_SP_r1_d0p3 --max-epochs 2
python scripts/run_grid.py runs/smoke --only interval__mix_gru_K8__synth_SP_r1_d0p3__lr1e3__s0
cat runs/smoke/cells/synthetic__interval__mix_gru_K8__synth_SP_r1_d0p3__lr1e3__s0/result.json   # test C, B, floors
python scripts/rescore_saved.py score runs/smoke --selection C       # re-score the saved checkpoint
python scripts/rescore_saved.py report runs/smoke
unset COARSE_TPP_DATA
```

`result.json` holds, for each checkpoint selection (`C`: dev interval score, `own`: the treatment's own objective,
`B`: dev latent-time score on synthetic data), the test scores `C`, `B`, `floor_1e-10`, `floor_1e-6`, `floor_1e-5`,
the true-law scores `oracle_C`, `oracle_B` (synthetic data) and their tie / non-tie means.

## Reproducing the paper

Each experiment is a grid of cells (one cell = data set × encoder × head × treatment × learning rate × seed). The
same three commands apply to every training experiment; `floor_convention` is reported by the re-scoring commands
given in its row of the table below instead of `scripts/report.py`:

```bash
python scripts/make_grid.py EXPERIMENT --out runs/EXPERIMENT          # needs results/audit/audit.json for recorded data
python scripts/run_grid.py runs/EXPERIMENT --gpus 0,1,2 --per-gpu 2  # or --workers N on CPU; re-run to resume
python scripts/report.py EXPERIMENT runs/EXPERIMENT                   # writes runs/EXPERIMENT/report.md, summary.json
```

| Paper | Experiment / command | Cells |
|---|---|---:|
| Section 2, Table 1 and Appendix B (audit of 23 data sets) | `scripts/audit.py --out results/audit` | — |
| Section 5.1, Table 3 and S2 (re-scoring saved mixture models; see "Reproduction scope") | `floor_convention`, then `scripts/rescore_saved.py score/report runs/floor_convention --compare floor_1e-6_sigma_sigmoid floor_1e-6_standard` | 840 |
| Section 5.2, Supplementary Table S1 and S2 (re-scoring intensity-based models; see "Reproduction scope") | `scripts/rescore_intensity.py score/report --ckpt CKPT --out results/intensity` (EasyTPP runs) | — |
| Section 5.3, Table 4 and Supplementary Table S2; Section 6.2, Table 5; S2 (mixture head) | `rank_change` | 1,440 |
| Sections 5.3, 6.2 and S2 (new splits, further data sets) | `resplit`, `new_datasets` | 400, 150 |
| Section 6.1, S1 and S2 (synthetic data) | `synthetic` | 5,040 |
| Section 6.1 and S3 (coarsened real data) | `coarsened_real`, then `scripts/score_latent.py runs/coarsened_real` before the report | 1,440 |
| Section 6.2, Table 5; S1 and S2 (RMTPP, THP) | `intensity_heads` | 600 |
| Sections 5.3 and 6.2, Table 5; S2 (process logs) | `process_logs` | 840 |
| Section 6.2 and S3 (multi-sample jitter) | `iw_jitter` | 1,380 |
| Section 7 and S3 (self-correcting process at Δ = 1) | `iw_jitter_large`; `scripts/inbin_diagnostic.py true_law` and `models --run runs/synthetic --cells '*synth_SC_r*_d1p0*'` | 240 |
| Section 6.2 and S3 (smoothing objectives) | `inbin_smoothing`, `inbin_smoothing_seeds` | 1,150, 800 |
| Section 7 and S3 (misspecified resolution) | `delta_misspecification` (report with `--base runs/synthetic`) | 554 |
| Section 7 and S3 (marks) | `marks` | 420 |
| S3 and Supplementary Table S3 (predictive accuracy of the selected models; no training) | after `rank_change`: `scripts/predictive_accuracy.py score runs/rank_change`, then `scripts/predictive_accuracy.py report runs/rank_change` | — |

Statistics: within a unit (data set, encoder, head, treatment) the learning rate is chosen by the dev mean over seeds
of the selection criterion, differences are paired by seed (synthetic data: averaged within each realization), and
intervals are one-sided 95% t intervals. *Preserved*: lower bound of [interval − ε = 10⁻¹⁰] ≥ −0.01 nats;
*violated*: upper bound < −0.01; *advantage*: Holm-adjusted p < 0.05 and a mean gain ≥ 0.01 nats over the strongest
non-interval treatment chosen by dev C. These rules are implemented in `coarse_tpp/analysis.py`.

Results are deterministic for a given device and library version (seeded data order, initialization and jitter
draws). Evaluating a saved model on another device changes the scores in the 6th to 8th decimal (float32 encoders;
floor scores of very narrow mixture components can move by up to about 1e-4).

## Reproduction scope

The audit (Section 2) and the training experiments (Sections 5.3, 6 and 7 and the Supplementary Material) are run by
the commands above from the data listed in `data/README.md`. The two re-scoring analyses of Sections 5.1 and 5.2
evaluate saved models that are not distributed with this repository:

- **Section 5.1, Table 3 (saved mixture-density models).** The re-scoring code (`coarse_tpp/rescore.py`,
  `scripts/rescore_saved.py`) is the procedure used in the paper. The 1,020 saved checkpoints it was applied to are
  not distributed. The experiment `floor_convention` retrains comparable models with this code (840 models: two
  parameterizations of the mixture head, 21 data sets, two learning rates, ten seeds, trained and selected under the
  floor 10⁻⁶); it approximates the saved models of the paper but is not identical to them (their number and the
  training code differ), so its scores are expected to be close to, not equal to, Table 3. The StackOverflow subsample
  is not reconstructible (see `data/README.md`).
- **Section 5.2, Supplementary Table S1 (intensity-based models).** The EasyTPP checkpoints are not distributed. The
  models were trained with the EasyTPP implementation of RMTPP, NHP, SAHP and AttNHP with these settings: hidden
  size 64, 100 epochs, learning rate 10⁻³, batch size 32, seed 42, 10 Monte Carlo samples per interval for the
  compensator, with a time-only objective (the log-likelihood of the event times without the mark term, i.e.
  `-time_ll` instead of `loss` in each model's `loglike_loss`). Models trained in this way with EasyTPP can be
  re-scored with `scripts/rescore_intensity.py`, which expects one EasyTPP run directory (`saved_model.pt`,
  `config.yaml`, optionally the training log) per data set, model and run.

## Names in the paper and in the code

| Paper | Code |
|---|---|
| ε-floor log-likelihood, ε = 10⁻¹⁰, 10⁻⁶, 10⁻⁵ | score keys `floor_1e-10`, `floor_1e-6`, `floor_1e-5`; training treatment `floor` (arms `floor_1e-10`, `floor_1e-6`, `floor_1e-5`) |
| interval likelihood (objective) / interval score (evaluation) | treatment `interval`; `lik.event_loglik(..., "interval")`; `score.interval_loglik` |
| observed-cell score C | `C` |
| latent-time score B | `B` |
| conversion rules (readouts) | `readout.RULE`: `raw`, `smooth`, `point_mass` |
| smoothed density q_B | readout `smooth` |
| uniform jitter of the gaps | `jitter_gap` |
| uniform jitter of the recorded times | `jitter_time` |
| Gaussian jitter of the recorded times | `jitter_time_gauss` |
| multi-sample jitter with S samples | `iw_jitter` with `iw_samples` = S (arms `iw_jitter_S{S}`) |
| point mass at zero (zero-inflated model) | `point_mass` |
| triangular-kernel likelihood | `triangular` |
| mixtures of the interval and jitter objectives | `interval_jitter_mix` with `mix_lambda` |
| model trained on the latent times | `latent` |
| control trained on marks alone | arm `mark_only` |
| log-normal mixture head | head `mix` (`mix_param: standard`) |
| mixture head with component scales bounded through a sigmoid | `mix_param: sigma_sigmoid` |
| RMTPP, THP heads | heads `rmtpp`, `thp` |
| GRU / Transformer encoder | encoders `gru`, `transformer` |
| setting (process, Δ) and realization r | data set `synth_<process>_r<r>_d<Δ>`, e.g. `synth_SC_r1_d1p0` |
| ratio of the largest to the smallest density inside a cell | `diagnostics.py`, `ratio` |
| re-scoring saved models | `floor_convention` + `rescore.py` |
| re-scoring intensity-based models | `rescore_intensity.py` |
| rankings of training treatments | experiment `rank_change` |
| predictive accuracy of the selected models (law q_B, median, truncated CRPS twCRPS) | `predictive.py` (`cdf`, `median`, `cell_crps`), `scripts/predictive_accuracy.py` |
| new random splits of the same data sets | experiment `resplit`, data sets `<name>_resplit` |
| data sets not used elsewhere (MemeTracker, MOOC; Wikipedia as control) | experiment `new_datasets` |
| ContTime, Taxi, StackOverflow coarsened to Δ = r × median gap | experiment `coarsened_real`, data sets `coarsened_<base>_r<r>` |
| misspecified resolution | experiment `delta_misspecification`, arms `<treatment>_dx<m>` |
| smoothing objectives | experiments `inbin_smoothing`, `inbin_smoothing_seeds` |
| business process logs | experiment `process_logs` |

| Data set (paper) | Folder | Data set (paper) | Folder |
|---|---|---|---|
| Retweet | `retweet` | Retweet (subsample) | `retweet_sub` |
| StackOverflow | `stackoverflow` | StackOverflow (subsample) | `stackoverflow_sub` |
| Taxi | `taxi` | Email-EU | `email_eu` |
| CollegeMsg | `collegemsg` | Volcano | `volcano` |
| Earthquake | `earthquake` | BookOrder | `bookorder` |
| Amazon | `amazon` | Taobao | `taobao` |
| MIMIC-II | `mimic` | ContTime | `conttime` |
| ContTime-q0.02 / -q0.1 | `conttime_q02` / `conttime_q10` | Synthetic-Exp | `synthetic_exp` |
| MemeTracker | `memetracker` | MOOC / Wikipedia | `mooc` / `wikipedia` |
| BPIC12, BPIC13-inc. | `bpi_bpic12`, `bpi_bpic13i` | BPIC20-DD, -PTC, -RfP | `bpi_bpic20dd`, `bpi_bpic20ptc`, `bpi_bpic20rfp` |
| Helpdesk, Sepsis | `bpi_helpdesk`, `bpi_sepsis` | further simulated data sets | `synthetic_powerlaw_0{1,2}`, `synthetic_corr_pareto_0{1,2}` |

## Repository layout

```
coarse_tpp/           the package
  lik.py              predictive laws (log-normal mixture, RMTPP, THP) and per-event scores
  readout.py          conversion rules and the scores C and B
  score.py            audit rule and model-agnostic interval / floor scores
  audit.py            audit of data directories
  data.py, models.py, train.py, results.py      data, encoders and heads, training of one cell
  synthetic.py        synthetic processes, true-law quantities, recorded views
  experiments.py      the cell grids of the paper
  runner.py           runs a grid sequentially or on several GPUs
  analysis.py         summary statistics and reports
  rescore.py, rescore_intensity.py, latent_score.py, diagnostics.py, predictive.py
  prepare/            conversion of public data and derived data sets
  resources/          reference tie shares of the synthetic data; the Retweet subsample index
scripts/              command-line entry points
tests/                unit and end-to-end tests (CPU)
third_party/easy_tpp/ vendored EasyTPP models (S2P2 repository, commit c393324, Apache-2.0) for re-scoring saved
                      intensity models
data/README.md        sources and preparation of every data set
data/synthetic_extra/ four simulated data sets that are not available elsewhere (about 23 MB)
```

## Citation

Please cite the article (reference will be added upon publication). `CITATION.cff` holds the title.

## License

MIT (see `LICENSE`). `third_party/easy_tpp` is EasyTPP model code as distributed in the S2P2 repository
(https://github.com/UCIDataLab/state_space_point_process, a fork of EasyTPP), commit c393324, under the Apache
License 2.0 (`third_party/easy_tpp/LICENCE`, `NOTICE`); the model files are unmodified, and the only other files are
the minimal import shim `easy_tpp/utils/__init__.py` and short package `__init__` files (see
`third_party/easy_tpp/README.md`). The simulated data sets in `data/synthetic_extra/` are distributed under the MIT
License; the other data sets have their own licences and terms (see `data/README.md`).
