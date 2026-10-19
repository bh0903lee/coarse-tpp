#!/usr/bin/env bash
# Email-EU (email-Eu-core-temporal) and CollegeMsg from SNAP, converted to data/email_eu and data/collegemsg.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW="${RAW:-data/raw/snap}"
mkdir -p "$RAW"
for f in email-Eu-core-temporal CollegeMsg; do
  [ -f "$RAW/$f.txt" ] || { curl -fL -o "$RAW/$f.txt.gz" "https://snap.stanford.edu/data/$f.txt.gz" && gunzip -f "$RAW/$f.txt.gz"; }
done
python scripts/prepare_snap.py --email "$RAW/email-Eu-core-temporal.txt" --college "$RAW/CollegeMsg.txt"
