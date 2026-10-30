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
   treatment of ties studied in the paper (floors, jitter of gaps or times, multi-sample uniform or triangular
   jitter, a point mass at zero, the interval likelihood, a triangular-kernel likelihood, interval / jitter mixtures),
   optionally with lower bounds on the mixture components;
4. **generate** the synthetic processes with known latent times (SP, SR, SC, H1, H2, and two non-stationary ones) and
   their coarsened recordings;
5. **prepare** the derived data sets (quantized ContTime, coarsened real data, new random splits, MOOC / Wikipedia,
   business process logs);
6. **run** the experiment grids of the paper and compute their summary statistics; each analysis writes a report
   (`report.md`, `summary.json`) with the values given in the paper.

## Installation

Python 3.10. CPU is enough for the quick start; the full grids were run on GPUs.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch 2.6.0; pick the CUDA or CPU wheel for your machine
pip install -e .                         # optional: installs the package coarse_tpp
python -m pytest -q tests                # unit and integration tests, CPU, about two minutes
```

The scripts in `scripts/` work from the repository root without installation. Data live in `./data` (or the
directory given by the environment variable `COARSE_TPP_DATA`); see [data/README.md](data/README.md).

A regular installation (`pip install .`) or an installed wheel includes the Python package and the vendored
EasyTPP models used by `coarse_tpp.rescore_intensity`, together with their license and provenance files. The
`scripts/` wrappers, example configurations and data sets remain in the source checkout. Outside that checkout,
set `COARSE_TPP_DATA` to your data directory and use module entry points, for example
`python -m coarse_tpp.rescore_intensity --help`.

To check the built distribution, run `pip wheel --no-deps --wheel-dir dist .`, then
`COARSE_TPP_TEST_WHEEL="$PWD/dist/coarse_tpp-1.2.2-py3-none-any.whl" python -m pytest -q tests/test_wheel.py`.
This check rebuilds and scores all four supported intensity models from the wheel in a separate process, without
source-checkout imports, and verifies the vendored files against their recorded hashes.

## Quick start

**Audit a data set.** Every data set is a directory `data/<name>/{train,dev,test}.pkl` in the NHP / EasyTPP layout
(`{split: [sequence], "dim_process": n}`, events `{"time_since_start", "time_since_last_event", "type_event"}`).

```bash
python scripts/audit.py --datasets email_eu volcano --out results/audit     # writes audit.json and audit.md
python scripts/score.py audit --pkl any_file.pkl   # the same rule on any pickle file(s); loading a pickle runs code, use trusted files only
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

`F_hi - F_lo` is the probability of the half-open cell `[τ̃, τ̃ + Δ)` when the law has no atom at a positive time (an
atom at 0 counts toward the tie cell); for a law with atoms at positive times, pass the left limits of the
distribution function at both edges, or give the law a `log_interval(a, b)` method. Distribution function values
outside [0, 1] or decreasing across the cell (beyond a roundoff of 1e-9) raise an error, and a cell of exact zero
probability scores −∞. `summarize` returns `mean` over all events (−∞ if any cell has zero probability), its tie /
non-tie split, `all_finite`, the counts `n_neg_inf` and `n_nonfinite`, and `mean_finite`, the mean over the
finite scores only (`n_finite` events), a diagnostic that is not the score over all events.

**Train and score one cell (CPU smoke test, under a minute).**

```bash
export COARSE_TPP_DATA="$PWD/data_smoke"   # keep the small realization apart from data/synthetic
python scripts/generate_synthetic.py --processes SP --realizations 1 --n-total 15000   # a small realization
python scripts/make_grid.py synthetic --out runs/smoke --datasets synth_SP_r1_d0p3 --max-epochs 2
python scripts/run_grid.py runs/smoke --only interval__mix_gru_K8__synth_SP_r1_d0p3__lr1e3__s0
cat runs/smoke/cells/synthetic__interval__mix_gru_K8__synth_SP_r1_d0p3__lr1e3__s0/result.json   # test C, B, floors
python scripts/rescore_trained.py score runs/smoke --selection C       # re-score the trained checkpoint
python scripts/rescore_trained.py report runs/smoke
unset COARSE_TPP_DATA
```

`result.json` holds, for each checkpoint selection (`C`: dev interval score, `own`: the treatment's own objective,
`B`: dev latent-time score on synthetic data), the test scores `C`, `B`, `floor_1e-10`, `floor_1e-6`, `floor_1e-5`,
the true-law scores `oracle_C`, `oracle_B` (synthetic data) and their tie / non-tie means. A tie is a recorded gap
≤ 10⁻⁶ when Δ = 0 and a recorded zero gap after loading when Δ > 0, as in the score `C`.

## Experiments implemented in this package

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
| Section 2, Table 1 and Appendix B (audit of 23 data sets; 22 are reconstructible from public files, the StackOverflow subsample is not) | `scripts/audit.py --out results/audit` | — |
| Section 5.1, Table 3 and Supplementary Sections S2 and S3 (re-scoring trained mixture models; see "Reproduction scope") | `floor_convention`, then `scripts/rescore_trained.py score/report runs/floor_convention --compare floor_1e-6_sigma_sigmoid floor_1e-6_standard` | 840 |
| Section 5.2 and Supplementary Table S2 (re-scoring the seed-42 intensity-based models; see "Reproduction scope") | `rescore_intensity_models.py` of the trained-models archive; `scripts/rescore_intensity.py score/report --ckpt CKPT --out results/intensity` gives single-rule scores of RMTPP, NHP, SAHP and AttNHP runs | — |
| Section 5.2, Supplementary Tables S3, S4 and S5 and Fig. S3 (EasyTPP models retrained with seeds 0–4: time-only objective on Email-EU, the Retweet subsample and CollegeMsg; the mark term in the objective on the Retweet subsample; time-only objective on Volcano, Sepsis and BPIC12) | `scripts/easytpp_train.py prepare --out runs/easytpp/data`; `scripts/easytpp_train.py train --experiment {time_only,joint,further} --data runs/easytpp/data --out runs/easytpp/EXP`; then `scripts/rescore_retrained.py score`, `repro` (joint, further) and `report --experiment EXP --ckpt runs/easytpp/EXP --data runs/easytpp/data --out results/EXP` | 135, 45, 135 runs |
| Section S4 (threshold-exceedance comparison of the retrained intensity-based models; no training) | after `rescore_retrained.py score --experiment time_only`: `scripts/intensity_exceedance.py score`, then `report --scores results/time_only --ckpt CKPT --data runs/easytpp/data --out results/exceedance` | — |
| Section 5.3, Table 4 and Supplementary Table S6; Section 6.2, Supplementary Table S1; S2 and S3 (mixture head) | `rank_change` | 1,440 |
| Sections 5.3, 6.2 and S3 (new splits, further data sets) | `resplit`, `new_datasets` | 400, 150 |
| Section 6.1, S2 and S3 (synthetic data) | `synthetic` | 5,040 |
| Section 6.1 and S4 (coarsened real data) | `coarsened_real`, then `scripts/score_latent.py runs/coarsened_real` before the report | 1,440 |
| Section 6.2, Supplementary Table S1; S2 and S3 (RMTPP, THP) | `intensity_heads` | 600 |
| Sections 5.3 and 6.2, Supplementary Table S1; S2 and S3 (process logs) | `process_logs` | 840 |
| Section 6.2 and S4 (multi-sample jitter) | `iw_jitter` | 1,380 |
| Section 8 and S4 (self-correcting process at Δ = 1) | `iw_jitter_large`; `scripts/inbin_diagnostic.py true_law` and `models --run runs/synthetic --cells '*synth_SC_r*_d1p0*'` | 240 |
| Section 8 and S4 (smoothing objectives) | `inbin_smoothing`, `inbin_smoothing_seeds` | 1,150, 800 |
| Section 8 and S4 (misspecified resolution) | `delta_misspecification` (report with `--base runs/synthetic`) | 554 |
| Section 8 and S4 (marks) | `marks` | 420 |
| Section S4, Supplementary Table S13 and Fig. S6 (joint rankings of time and mark; no training) | after `marks`: `scripts/joint_rankings.py runs/marks` (writes `runs/marks/joint_rankings/`) | — |
| Section 5.3, S3 and Supplementary Table S8 (other density heads: Weibull mixture, spline flow) | `density_heads` (report with `--base runs/rank_change`) | 400 |
| Section S4 and Fig. S4 (floor score on a grid of 37 floors; no training) | after `rank_change`: `scripts/floor_grid.py score runs/rank_change` (`--cpu`, `--shard i/n`), then `scripts/floor_grid.py report runs/rank_change` (writes `runs/rank_change/floor_grid/`) | — |
| Section S4 and Supplementary Table S7 (bootstrap of the rankings; no training) | after `rank_change`: `scripts/rank_bootstrap.py run runs/rank_change` (writes `runs/rank_change/rank_bootstrap/`) | — |
| Section 5.3, S4 and Supplementary Table S14 (predictive accuracy of the selected models; no training) | after `rank_change`: `scripts/predictive_accuracy.py score runs/rank_change`, then `scripts/predictive_accuracy.py report runs/rank_change` | — |
| Section 5.3 and Supplementary Table S9 (location and scale of the mixture component at the tie: edits of the saved checkpoints; no training) | after `rank_change`: `scripts/component_edits.py score runs/rank_change`, then `scripts/component_edits.py report runs/rank_change` (writes `runs/rank_change/component_edits/`) | — |
| Section 5.3 and Supplementary Tables S10 and S11 (lower bounds on the mixture components) | `component_bounds` (report with `--base runs/rank_change`, the unbounded arm) | 450 |
| Section 5.3 and Supplementary Table S12 (threshold-exceedance forecasts and remaining time on the process logs; no training) | after `process_logs`: `scripts/sla_alarms.py score runs/process_logs --tasks sla,rt`, then `scripts/sla_alarms.py report runs/process_logs` (writes `runs/process_logs/alarms/`; `--base runs/triangular_recorded` adds the triangular candidate) | — |
| Section 8 and S4 (triangular kernel on coarsened and recorded data, multi-sample triangular jitter; the recorded-data table of the report takes its combinations from the `triangular_recorded` grid only) | `triangular_coarsened`, `coarsened_more`, `triangular_jitter`, each followed by `scripts/score_latent.py RUN` (also on `runs/coarsened_real`, `runs/synthetic`, `runs/inbin_smoothing`); `triangular_recorded`; one report: `scripts/report.py triangular_coarsened runs/triangular_coarsened --base runs/coarsened_real runs/coarsened_more runs/triangular_jitter runs/synthetic runs/inbin_smoothing runs/triangular_recorded runs/rank_change runs/process_logs` | 240, 840, 900, 320 |

Supplementary Table S14 is derived from the rows of `predictive/summary.json`: for `twcrps` and `mae_median` (the other
measures, `rmse`, `rmse_median` and `crps`, are in the same file), the table reports the seed-paired difference `d_m` (interval-score
choice minus floor-score choice) in percent of the floor-score choice's value, `100 × d_m / m_floor`; the same scaling
applies to the upper bound `d_m_up95`.

Statistics: within a unit (data set, encoder, head, treatment) the learning rate is chosen by the dev mean over seeds
of the selection criterion, differences are paired by seed (synthetic data: averaged within each realization), and
intervals are one-sided 95% t intervals. *Preserved*: lower bound of [interval − ε = 10⁻¹⁰] ≥ −0.01 nats;
*violated*: upper bound < −0.01; *advantage*: Holm-adjusted p < 0.05 and a mean gain ≥ 0.01 nats over the strongest
non-interval treatment chosen by dev C. These rules are implemented in `coarse_tpp/analysis.py`.

The component and triangular-kernel analyses (Tables S9–S12, the triangular comparison of Section S4) are implemented in this
release; their full training and checkpoint-scoring pipelines have not been rerun end to end with this public package.
Unit tests and small integration tests validate the implementation, and the aggregation functions reproduce the
published tables from the original result files; matching those tables requires the specified data, grids, selection
rules and score checks.

Results are deterministic for a given device and library version (seeded data order, initialization and jitter
draws). Evaluating a trained model on another device or library version changes the scores in the 6th to 8th decimal
(float32 encoders); floor scores of very narrow mixture components, as in IntensityFree, can move by up to about 1e-4.
These figures are for means over all events of a split. Means over the tied events alone average far fewer events and
can differ more (up to about 2e-3 nats in a CPU re-scoring of IntensityFree models trained on a GPU).

Cached scores (`rescore.jsonl`, `latent_scores.jsonl`, predictive, remaining-time and intensity score files) record
the configuration and checkpoint they were computed from and are reused or reported only while these still match;
files without this record are shown as stale or pending until re-scored (latent-time rows stay readable). An
interrupted scoring call resumes from its `.tmp` file.

## Reproduction scope

The commands above implement every analysis of the paper and its Supplementary Material. The reports contain the
values given in the paper; tables and figures are typeset from them. Supplementary Table S7 lists the data sets in the
order and with the tie shares of Table 1 (the audit); its report gives the tie shares of the test split. Trained models
that are distributed are listed below; all other results are reproduced by training with the commands above. The
trained models, their configurations and scores, and the StackOverflow subsample are in the archive
`coarse-tpp-trained-models.zip` attached to the release `trained-models-v1` of this repository
(https://github.com/bh0903lee/coarse-tpp/releases); its README describes every directory.

- **Section 5.1, Table 3 (trained mixture-density models).** The re-scoring code (`coarse_tpp/rescore.py`,
  `scripts/rescore_trained.py`) is the procedure used in the paper. The 1,020 trained checkpoints it was applied to are
  in that archive, not in the source tree, except the 60 trained on MIMIC-II (the original database requires a data
  use agreement), whose scores alone are included. The experiment `floor_convention` retrains comparable models with this code (840 models: two
  parameterizations of the mixture head, 21 data sets, two learning rates, ten seeds, trained and selected under the
  floor 10⁻⁶); it approximates the trained models of the paper but is not identical to them (their number and the
  training code differ), so its scores are expected to be close to, not equal to, Table 3. The StackOverflow subsample
  is not reconstructible from public files and is included in the archive (see `data/README.md`).
- **Section 5.2, Supplementary Tables S2 and S3 (intensity-based models).** The selected EasyTPP checkpoints of
  Table S2 (RMTPP, NHP, SAHP and AttNHP, seed 42, settings of `configs/easytpp_intensity.yaml`: hidden size 64, 100
  epochs, learning rate 10⁻³, batch size 32, 10 Monte Carlo samples per interval for the compensator) and the models of
  Table S3, retrained with seeds 0–4 on Email-EU, the Retweet subsample and CollegeMsg (RMTPP, NHP, SAHP, AttNHP, THP,
  S2P2, FullyNN, IntensityFree; same settings), are in that archive. All were trained with a time-only objective (the
  log-likelihood of the event times without the mark term; not a configuration option). The archive also holds the
  patched EasyTPP model code needed to rebuild them and `rescore_intensity_models.py`, which reproduces the archive's
  original scores: compensators are integrated by a composite Gauss–Legendre rule graded toward both ends of each gap, for S2P2
  additionally with fine panels at both ends of each of the first 64 one-second recording cells and of the target cell
  and the graded rule over the remaining tail. The values of the paper use the rules of `scripts/rescore_retrained.py`
  instead: the graded rule for every model, and for S2P2 a stratified Monte Carlo estimate of the compensator (16,384
  strata per gap) with a dense rule for the recorded cell; `--ckpt` accepts the archive's
  `intensity_models_five_seeds/` directly, and `scripts/intensity_exceedance.py` uses the same checkpoints. The S2P2
  estimate depends on the device's random generator: on CUDA it reproduces the published values, on CPU it differs by
  Monte Carlo error (about 0.003 nats per event).
  `scripts/rescore_intensity.py` of this repository re-scores RMTPP, NHP,
  SAHP and AttNHP runs (one EasyTPP run directory with `saved_model.pt` and `config.yaml` per data set, model and run)
  with a single Gauss–Legendre rule per gap (`--nodes`, 32 by default); its values can differ from the paper's by up to
  about 0.7 nats per event (retrained RMTPP on Email-EU) for intensities that rise steeply after an event.
- **Supplementary Tables S4 and S5 (intensity-based models retrained with the mark term on the Retweet subsample, and
  on Volcano, Sepsis and BPIC12).** The checkpoints, their scores and the reproduction references are in the same
  archive (`intensity_models_mark_term/`, `intensity_models_further_data/`). `scripts/rescore_retrained.py import
  --experiment {joint,further} --ckpt <archive directory> --out OUT` turns the deposited scores into the input of
  `report`; `score` recomputes them from the checkpoints, and `report --repro-dir <archive directory>/repro` uses the
  deposited reproduction references. `scripts/easytpp_train.py` retrains the models with the vendored, patched EasyTPP
  in `third_party/easy_tpp_train/` (configurations identical to those of the paper apart from paths; `PATCHES.md`
  lists the changes); retrained models can differ slightly from the deposited ones because GPU training is not
  bit-for-bit deterministic, and `scripts/rescore_retrained.py repro` checks each new run against EasyTPP's own Monte
  Carlo evaluation before it enters the comparison.
- **Section 5.3, Table 4 (treatment rankings) and the analyses that re-score these models (Fig. S4, Supplementary
  Tables S7, S9 and S14).** The 1,440 trained models of `rank_change` are in the archive (`rank_change/`) in the run
  layout of this package; paths in it are relative, so run the commands of the table above from inside a copy of that
  directory (see the archive's README).

## Names in the paper and in the code

| Paper | Code |
|---|---|
| ε-floor log-likelihood, ε = 10⁻¹⁰, 10⁻⁶, 10⁻⁵ | score keys `floor_1e-10`, `floor_1e-6`, `floor_1e-5`; training treatment `floor` (arms `floor_1e-10`, `floor_1e-6`, `floor_1e-5`) |
| interval likelihood (objective) / interval score (evaluation) | treatment `interval`; `lik.event_loglik(..., "interval")`; `score.interval_loglik` |
| recorded-cell score C | `C` |
| latent-time score B | `B` |
| conversion rules (readouts) | `readout.RULE`: `raw`, `smooth`, `point_mass` |
| alternative readout of a treatment (raw ↔ smooth; density part of the point-mass model) | `readout.ALT`; key `B_alt` of `latent_score.py` |
| smoothed density q_B | readout `smooth` |
| uniform jitter of the gaps | `jitter_gap` |
| uniform jitter of the recorded times | `jitter_time` |
| Gaussian jitter of the recorded times | `jitter_time_gauss` |
| multi-sample jitter with S samples | `iw_jitter` with `iw_samples` = S (arms `iw_jitter_S{S}`) |
| multi-sample triangular jitter with S samples (target gap drawn from the triangular kernel of its cell) | `iw_jitter_triangular` with `iw_samples` = S (arms `iw_jitter_triangular_S{S}`) |
| point mass at zero (zero-inflated model) | `point_mass` |
| triangular-kernel likelihood | `triangular` |
| mixtures of the interval and jitter objectives | `interval_jitter_mix` with `mix_lambda` |
| model trained on the latent times | `latent` |
| control trained on marks alone | arm `mark_only` |
| log-normal mixture head | head `mix` (`mix_param: standard`) |
| mixture head with component scales bounded through a sigmoid | `mix_param: sigma_sigmoid` |
| lower bounds on the mixture components, σ_k ≥ 2 and μ_k ≥ log 10⁻⁴ | config keys `sigma_min`, `mu_min`; experiment `component_bounds`, arms `<treatment>_sigmin2`, `<treatment>_mumin1e-4`, `<treatment>_sigmin2_mumin1e-4` |
| self-preference margin of a floor-trained model; relocation / rescaling of the component at the tie | `component_edits.py` (edits `location`, `scale`, `both`; `scale_event` as a secondary contrast) |
| RMTPP, THP heads | heads `rmtpp`, `thp` |
| GRU / Transformer encoder | encoders `gru`, `transformer` |
| setting (process, Δ) and realization r | data set `synth_<process>_r<r>_d<Δ>`, e.g. `synth_SC_r1_d1p0` |
| ratio of the largest to the smallest density inside a cell | `diagnostics.py`, `ratio` |
| re-scoring trained models | `floor_convention` + `rescore.py` |
| re-scoring intensity-based models | `rescore_intensity.py` |
| rankings of training treatments | experiment `rank_change` |
| predictive accuracy of the selected models (law q_B, median, threshold-weighted ranked probability score twCRPS) | `predictive.py` (`cdf`, `median`, `cell_crps`), `scripts/predictive_accuracy.py` |
| new random splits of the same data sets | experiment `resplit`, data sets `<name>_resplit` |
| data sets not used elsewhere (MemeTracker, MOOC; Wikipedia as control) | experiment `new_datasets` |
| ContTime, Taxi, StackOverflow coarsened to Δ = r × median gap | experiment `coarsened_real`, data sets `coarsened_<base>_r<r>` |
| Earthquake and Taobao coarsened to Δ = r × median gap (storage-grain rule) | experiment `coarsened_more`, data sets `coarsened_earthquake_r<r>`, `coarsened_taobao_r<r>` (`scripts/prepare_coarsened.py --more`) |
| triangular kernel on the coarsened and the recorded data sets | experiments `triangular_coarsened`, `triangular_jitter`, `triangular_recorded`; report `triangular_coarsened` with `--base` |
| misspecified resolution | experiment `delta_misspecification`, arms `<treatment>_dx<m>` |
| smoothing objectives | experiments `inbin_smoothing`, `inbin_smoothing_seeds` |
| business process logs | experiment `process_logs` |
| floor score on a grid of floors | `floor_grid.py` |
| bootstrap of the rankings | `rank_bootstrap.py` |
| joint rankings of time and mark | `joint_rankings.py` |
| Weibull-mixture and spline-flow heads | heads `weib`, `flow`; experiment `density_heads` |
| retrained EasyTPP models (time-only objective; mark term; further data sets) | `easytpp_train.py`, experiments `time_only`, `joint`, `further`; scoring and reports `rescore_retrained.py` |
| threshold-exceedance comparison of the retrained intensity-based models | `intensity_exceedance.py` |
| threshold-exceedance forecasts, P(next activity not recorded within a horizon h) (illustrative deadline monitoring; no alarm policy evaluated), and remaining time of a case | `alarms.py`, `scripts/sla_alarms.py` (rules `F`: dev floor 10⁻⁶, `C`: dev interval score, `F_own`) |

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
  lik.py              predictive laws (log-normal and Weibull mixtures, spline flow, RMTPP, THP) and per-event scores
  readout.py          conversion rules and the scores C and B
  score.py            audit rule and model-agnostic interval / floor scores
  audit.py            audit of data directories
  data.py, models.py, train.py, results.py      data, encoders and heads, training of one cell
  synthetic.py        synthetic processes, true-law quantities, recorded views
  experiments.py      the cell grids of the paper
  runner.py           runs a grid sequentially or on several GPUs
  analysis.py         summary statistics and reports
  rescore.py, rescore_intensity.py, latent_score.py, diagnostics.py, predictive.py
  component_edits.py  edits of the mixture component at the tie on saved checkpoints (Table S9)
  alarms.py           threshold-exceedance forecasts and remaining time on the process logs (Table S12)
  floor_grid.py, rank_bootstrap.py, joint_rankings.py     floor grid (Fig. S4), ranking bootstrap (Table S7),
                      joint rankings of time and mark (Table S13)
  easytpp.py, easytpp_train.py    data preparation and training of the EasyTPP models (Tables S3-S5)
  intensity_eval.py, rescore_retrained.py, intensity_exceedance.py   scoring rules, scores, reports and the
                      threshold-exceedance comparison of the retrained intensity-based models
  prepare/            conversion of public data and derived data sets
  resources/          reference tie shares of the synthetic data; the Retweet subsample index
scripts/              command-line entry points
configs/              EasyTPP runner configuration of the re-scored intensity models (Section 5.2)
tests/                unit and end-to-end tests (CPU)
third_party/easy_tpp/ vendored EasyTPP models (S2P2 repository, commit c393324, Apache-2.0) for re-scoring trained
                      intensity models
third_party/easy_tpp_train/  the full EasyTPP package of the same commit, patched for training (PATCHES.md)
data/README.md        sources and preparation of every data set
data/synthetic_extra/ four simulated data sets that are not available elsewhere (about 23 MB)
THIRD_PARTY_NOTICES.md  license notices for the vendored code
```

## Citation

Please cite the article (reference will be added upon publication). `CITATION.cff` holds the title.

## License

MIT (see `LICENSE`); notices for third-party code are in `THIRD_PARTY_NOTICES.md`. `third_party/easy_tpp` is
EasyTPP model code as distributed in the S2P2 repository
(https://github.com/UCIDataLab/state_space_point_process, a fork of EasyTPP), commit c393324, under the Apache
License 2.0 (`third_party/easy_tpp/LICENCE`, `NOTICE`); the model files are unmodified, and the only other files are
the minimal import shim `easy_tpp/utils/__init__.py` and short package `__init__` files (see
`third_party/easy_tpp/README.md`). The simulated data sets in `data/synthetic_extra/` are distributed under the MIT
License; the other data sets have their own licenses and terms (see `data/README.md`).
