# Data

Most data sets used in the paper are public and are not distributed with this repository; this file lists where to
obtain each of them, how to convert it into the layout read by `coarse_tpp.data`, and the original source to cite.
Two exceptions: four simulated data sets (power-law and correlated Pareto gaps, used only in the re-scoring of saved
models) are not available elsewhere and are distributed here in `data/synthetic_extra/`, and one StackOverflow
subsample cannot be reconstructed (see "Not reconstructible" below).

**Layout.** `data/<name>/{train,dev,test}.pkl`, each `{split: [sequence] or {key: sequence}, "dim_process": n}`;
a sequence is a list of events `{"time_since_start": float, "time_since_last_event": float, "type_event": int}`
(types 0-based). Recorded zero gaps are stored as `1e-6` (the NHP / EasyTPP convention) and read back as exact zeros;
the reading threshold `1.000001e-6` also maps a genuinely positive gap below it to zero (one gap in ContTime, two in
Earthquake). Business process logs (`bpi_*`) store exact zeros. A file `resolution.json` (`{"delta": ...}`) marks a resolution
known by construction. The directory can be changed with the environment variable `COARSE_TPP_DATA` or `--data`.

After all data sets are in place, run the audit once; the experiment grids read the resolution Δ from its output:

```bash
python scripts/audit.py --out results/audit
```

## Recorded data sets

| Data set | Folder | Original source | Distribution used | Preparation |
|---|---|---|---|---|
| Retweet | `retweet` | Zhao et al., SEISMIC, KDD 2015 | NHP data release (Mei and Eisner 2017) | NHP pickles: `scripts/prepare_nhp.py --src DIR --name retweet` |
| Retweet (subsample) | `retweet_sub` | as Retweet | 900 / 100 / 200 sequences of the NHP files | `scripts/prepare_subsample.py` (index in `coarse_tpp/resources/retweet_subsample.json`) |
| StackOverflow | `stackoverflow` | Du et al., RMTPP, KDD 2016 (preprocessing) | EasyTPP (Hugging Face `easytpp/stackoverflow`) | `scripts/prepare_easytpp.py --json-dir RAW/stackoverflow --name stackoverflow` |
| StackOverflow (subsample) | `stackoverflow_sub` | as StackOverflow | a 900 / 100 / 200-sequence subsample whose sequence list was not retained | not reconstructible; used in the audit (Table 1) and in the re-scoring of saved models and of intensity models (see below) |
| Taxi | `taxi` | Whong, FOILing NYC's Taxi Trip Data, 2014 | EasyTPP (`easytpp/taxi`) | `scripts/prepare_easytpp.py --json-dir RAW/taxi --name taxi --max-seqs 2000 0 500 --min-len 5` |
| Earthquake | `earthquake` | U.S. Geological Survey, ANSS ComCat, 2017 | EasyTPP (`easytpp/earthquake`) | as Taxi, `--name earthquake` |
| Volcano | `volcano` | Global Volcanism Program, Volcanoes of the World v. 4, 2013 | EasyTPP (`easytpp/volcano`) | `scripts/prepare_easytpp.py --json-dir RAW/volcano --name volcano` |
| Amazon | `amazon` | Ni, Li and McAuley, EMNLP-IJCNLP 2019 | EasyTPP (`easytpp/amazon`) | as Volcano |
| Taobao | `taobao` | Xue et al., HYPRO, NeurIPS 2022 | EasyTPP (`easytpp/taobao`) | as Volcano |
| Email-EU | `email_eu` | Paranjape, Benson and Leskovec, WSDM 2017 | SNAP `email-Eu-core-temporal` | `scripts/prepare_snap.py --email RAW/email-Eu-core-temporal.txt --college RAW/CollegeMsg.txt` |
| CollegeMsg | `collegemsg` | Panzarasa, Opsahl and Carley, JASIST 2009 | SNAP `CollegeMsg` | same command (both in one call) |
| BookOrder | `bookorder` | Du et al., RMTPP, KDD 2016 | NHP data release, fold 3 | NHP pickles |
| MIMIC-II | `mimic` | Saeed et al., Crit. Care Med. 2011; Du et al. 2016 (preprocessing) | NHP data release, fold 1 | NHP pickles (the original database requires a data use agreement) |
| ContTime | `conttime` | Mei and Eisner, NeurIPS 2017 (simulated) | NHP data release | NHP pickles |
| ContTime-q0.02, -q0.1 | `conttime_q02`, `conttime_q10` | ContTime with times floored to 0.02 / 0.1 | built here | `scripts/prepare_quantized.py` |
| MemeTracker | `memetracker` | Leskovec, Backstrom and Kleinberg, KDD 2009 | NHP data release | NHP pickles |
| MOOC, Wikipedia | `mooc`, `wikipedia` | Kumar, Zhang and Leskovec, JODIE, KDD 2019 | SNAP JODIE CSVs | `scripts/download_jodie.sh`, then `scripts/prepare_jodie.py --csv RAW/mooc.csv --name mooc` (and `wikipedia`) |
| Synthetic-Exp | `synthetic_exp` | Lee et al., Front. Comput. Neurosci. 2023 | `syn_exp_01` of github.com/bh0903lee/BMT-synthetic-datasets | `scripts/download_synthetic_exp.sh` |
| further simulated sets | `synthetic_powerlaw_01/02`, `synthetic_corr_pareto_01/02` | simulations generated for this study | this repository, `data/synthetic_extra/` | same script |
| Sepsis, BPIC12, BPIC13-inc., Helpdesk, BPIC20-DD, -PTC, -RfP | `bpi_sepsis`, `bpi_bpic12`, `bpi_bpic13i`, `bpi_helpdesk`, `bpi_bpic20dd`, `bpi_bpic20ptc`, `bpi_bpic20rfp` | 4TU.ResearchData deposits (DOIs in `coarse_tpp/prepare/bpi.py`) | official 4TU files | `scripts/download_bpi.sh` (download and conversion; preprocessing rules in `coarse_tpp/prepare/bpi.py`) |

Sources of the raw files. The NHP data release (Retweet, BookOrder, MIMIC-II, ContTime, MemeTracker) and the data
sets derived from it (`retweet_sub`, `conttime_q02`, `conttime_q10`, `coarsened_conttime_*` and their `_resplit`
versions) require a manual download of the NHP pickles; the other sources are fetched by the scripts named in the
table.
- NHP data release: the Google Drive folder linked from the data page of the neural Hawkes process code of Mei and
  Eisner, https://github.com/hongyuanmei/neurawkes/blob/master/data/README.md (Retweet, BookOrder, MIMIC-II,
  ContTime, MemeTracker; for BookOrder and MIMIC-II, which come in several folds, fold 3 and fold 1 are used). One
  pickle per split, already in the layout above; pass the folder of the data set (or fold) as `--src`.
- EasyTPP: the Hugging Face dataset repositories `easytpp/<name>` (files `train.json`, `dev.json`, `test.json`);
  `scripts/download_easytpp.sh NAME` fetches them.
- SNAP: https://snap.stanford.edu/data/email-Eu-core-temporal.html, https://snap.stanford.edu/data/CollegeMsg.html
  (`scripts/download_snap.sh`) and https://snap.stanford.edu/jodie/ (`scripts/download_jodie.sh`). The SNAP pages ask
  for citation of the original papers.
- 4TU.ResearchData: BPIC 2020 logs under CC BY-NC 4.0, the other logs under the 4TU.ResearchData General Terms of Use.
- Synthetic-Exp: the repository github.com/bh0903lee/BMT-synthetic-datasets is released under the MIT License;
  please cite its article (Lee et al., Front. Comput. Neurosci. 2023).

Simulated data distributed here: `data/synthetic_extra/{syn_powerlaw_01,syn_powerlaw_02,syn_corr_pareto_01,
syn_corr_pareto_02}/{train,dev,test}.pkl` (about 23 MB in total, SHA-256 in `data/synthetic_extra/SHA256SUMS`),
single-mark sequences with power-law or correlated Pareto-distributed gaps, used only in the re-scoring of saved
models (`floor_convention`). `scripts/download_synthetic_exp.sh` copies them, together with Synthetic-Exp, into the
data directory and stops with an error if a source folder is missing. They are distributed under the MIT License of
this repository. As with every pickle file, loading executes code; load only files from the sources listed here.

Not reconstructible from the public files here: the StackOverflow subsample (`stackoverflow_sub`), used in the audit
(Table 1) and in the re-scoring of saved models and of intensity models, is a subsample whose sequence list is not
recorded in this repository; it is included in the Zenodo deposit cited in the data availability statement of the paper. The other 22 audited data sets are reconstructible from the sources above.

## Derived data sets

```bash
python scripts/prepare_quantized.py         # conttime_q02, conttime_q10 (with resolution.json)
python scripts/prepare_coarsened.py         # coarsened_{conttime,taxi,stackoverflow}_r{0p03,0p1,0p3,1}
python scripts/prepare_resplit.py           # <name>_resplit for the eight data sets of the ranking experiment
python scripts/generate_synthetic.py        # data/synthetic/<process>_r<r>.npz, seven processes x five realizations
python scripts/audit.py --out results/audit # after every change of the data directory
```

The synthetic realizations are generated from fixed seeds and checked against the reference tie shares in
`coarse_tpp/resources/synthetic_tie_table.json`; about 1 GB in total.
