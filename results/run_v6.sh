#!/bin/bash
# v6 final runs (see docs/f1_final_config.md)
set -e
cd "$(dirname "$0")/.."
L="static=0.05 geometry=0.1 adaptive=0.1 learning=0.1 oracle=0.1 sweep=0 uncertainty=0"
E="python -X utf8 src/f1/experiment4.py --days 20 --lams $L"
$E --tag v6_main     --splits 5 --seeds 4 --thr 50 10 --kit on                                   > results/v6_main.log 2>&1
$E --tag v6_bd2012   --area bd2012 --impute --splits 2 --seeds 3 --thr 50 10                      > results/v6_bd2012.log 2>&1
$E --tag v6_learners --splits 2 --seeds 4 --thr 50 10 --kit on --policies learning --learners gp rf lgbm knn > results/v6_learners.log 2>&1
$E --tag v6_field    --splits 2 --seeds 4 --thr 50 --kit on --daily_return --absent 0.15          > results/v6_field.log 2>&1
$E --tag v6_advise   --splits 2 --seeds 4 --thr 50 10 --kit on --advise 0.5 --policies sweep static adaptive learning > results/v6_advise.log 2>&1
$E --tag v6_radius   --splits 2 --seeds 4 --thr 50 --kit on --radius 50 200 --policies sweep static adaptive learning > results/v6_radius.log 2>&1
$E --tag v6_nokit    --splits 2 --seeds 4 --thr 50 10 --kit off --policies sweep static adaptive learning > results/v6_nokit.log 2>&1
$E --tag v6_thin     --splits 2 --seeds 4 --thr 50 --kit on --thin 0.5 0.25 --policies sweep static adaptive learning > results/v6_thin.log 2>&1
$E --tag v6_t10      --splits 2 --seeds 4 --thr 50 --kit on --test_min 10 --policies sweep static adaptive learning > results/v6_t10.log 2>&1
$E --tag v6_bd2012_old --area bd2012 --impute --splits 2 --seeds 3 --thr 50 10 --known -1 --min_known 20 --policies sweep static adaptive learning > results/v6_bd2012_old.log 2>&1
echo ALL_DONE
# sensitivity analyses added after review (2026-11-28): kit readings of 0 set to 5 ug/L; kit-error variance added to the GP nugget
$E --tag v6_zero5      --splits 5 --seeds 4 --thr 50 10 --kit on --zero_as 5 --policies sweep static adaptive learning > results/v6_zero5.log 2>&1
$E --tag v6_bd2012_zero5 --area bd2012 --impute --splits 2 --seeds 3 --thr 50 10 --zero_as 5 --policies sweep static adaptive learning > results/v6_bd2012_zero5.log 2>&1
$E --tag v6_kitnug     --splits 5 --seeds 4 --thr 50 10 --kit on --kit_nugget --policies sweep static adaptive learning > results/v6_kitnug.log 2>&1
echo SENS_DONE
