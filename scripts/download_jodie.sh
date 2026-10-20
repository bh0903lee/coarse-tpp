#!/usr/bin/env bash
# MOOC and Wikipedia interaction logs of JODIE (Kumar, Zhang and Leskovec, KDD 2019) from SNAP, converted to data/.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW="${RAW:-data/raw/jodie}"
mkdir -p "$RAW"
for name in mooc wikipedia; do
  [ -f "$RAW/$name.csv" ] || curl -fL -o "$RAW/$name.csv" "https://snap.stanford.edu/jodie/$name.csv"
  python scripts/prepare_jodie.py --csv "$RAW/$name.csv" --name "$name"
done
