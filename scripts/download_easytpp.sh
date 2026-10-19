#!/usr/bin/env bash
# EasyTPP data sets from their Hugging Face repositories (train.json, dev.json, test.json), converted to data/<name>.
# Usage: scripts/download_easytpp.sh volcano amazon taobao stackoverflow taxi earthquake
set -euo pipefail
cd "$(dirname "$0")/.."
RAW="${RAW:-data/raw/easytpp}"
for name in "$@"; do
  mkdir -p "$RAW/$name"
  for split in train dev test; do
    [ -f "$RAW/$name/$split.json" ] || curl -fL -o "$RAW/$name/$split.json" \
      "https://huggingface.co/datasets/easytpp/$name/resolve/main/$split.json"
  done
  case "$name" in
    taxi|earthquake) python scripts/prepare_easytpp.py --json-dir "$RAW/$name" --name "$name" --max-seqs 2000 0 500 --min-len 5 ;;
    *) python scripts/prepare_easytpp.py --json-dir "$RAW/$name" --name "$name" ;;
  esac
done
