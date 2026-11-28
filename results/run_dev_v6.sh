#!/bin/bash
# Development-stage tuning of the look-ahead weight (lambda), on the five development census blocks only
# (leave-one-block-out fitting). The weights were chosen on results/f1_dev6_lam.csv (earlier development version
# of the simulator); this script repeats the run with the final code (dev6_lam_rerun), which selects the same weights.
set -e
cd "$(dirname "$0")/.."
G=""
for pol in static geometry adaptive learning; do
  for lam in 0 0.05 0.1 0.2 0.3 0.5; do G="$G $pol=$lam"; done
done
python -X utf8 src/f1/experiment4.py --tag dev6_lam_rerun --dev --splits 1 --seeds 3 --thr 50 10 --kit on --days 20 \
  --lams $G --policies static geometry adaptive learning --learners gp > results/dev6_lam_rerun.log 2>&1
python -X utf8 src/f1/select_lambda.py
echo DEV_DONE
