#!/usr/bin/env bash
# Synthetic-Exp (syn_exp_01 of github.com/bh0903lee/BMT-synthetic-datasets) and the four further simulated data sets
# distributed with this repository in data/synthetic_extra/ (syn_powerlaw_01/02, syn_corr_pareto_01/02). Both are
# NHP-style split pickles; they are copied to data/ with zero gaps stored as 1e-6. Exits with an error if any
# expected folder is missing.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW="${RAW:-data/raw/synthetic_datasets}"
EXTRA="data/synthetic_extra"
SRC_COMMIT=de7d3bd5bc1e54b79295112fb68db29bb50d800f   # pinned revision of BMT-synthetic-datasets
if [ ! -d "$RAW" ]; then
  git init -q "$RAW"
  git -C "$RAW" fetch -q --depth 1 https://github.com/bh0903lee/BMT-synthetic-datasets "$SRC_COMMIT"
  git -C "$RAW" checkout -q FETCH_HEAD
fi
missing=0
for pair in "$RAW/syn_exp_01:synthetic_exp" \
            "$EXTRA/syn_powerlaw_01:synthetic_powerlaw_01" "$EXTRA/syn_powerlaw_02:synthetic_powerlaw_02" \
            "$EXTRA/syn_corr_pareto_01:synthetic_corr_pareto_01" \
            "$EXTRA/syn_corr_pareto_02:synthetic_corr_pareto_02"; do
  src="${pair%%:*}"; dst="${pair##*:}"
  if [ ! -d "$src" ]; then echo "missing: $src (needed for $dst)" >&2; missing=1; continue; fi
  python scripts/prepare_nhp.py --src "$src" --name "$dst"
done
if [ "$missing" -ne 0 ]; then echo "some simulated data sets could not be prepared" >&2; exit 1; fi
