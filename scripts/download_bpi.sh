#!/usr/bin/env bash
# The seven business process logs from 4TU.ResearchData (official files), converted to data/bpi_<short>.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/prepare_bpi.py all sepsis bpic12 bpic13i helpdesk bpic20dd bpic20ptc bpic20rfp
