# Design log

The design history is summarised in Table S4 of the article. This file records the settings of each version.

## v1-v2 (2026-11-19, fixed on development blocks before any test-block run)
- Simulator: src/f1/sim2.py; 3 km/h walking, detour factor 1.3, 360 min per team-day.
- Learners compared: Gaussian process, kNN, LightGBM, random forest (all reported).
- Sensitivity: radius 50/200 m; 10 µg/L; no kit error; 5 and 20 min per test; thinning to 50% and 25% of wells.
- Amendment before any test-block result: van Geen et al. (2014) report about 12 wells per tester per day and the
  Econo-Quick kit needs about 15 min per sample, so the headline calibration became 20 min per test plus walking
  (about 12-15 wells per team-day) with a 10-day horizon. The 10-min and 5-min runs were kept as sensitivity analyses.

## v3 (2026-11-19)
- Seeding error corrected: seeds now use zlib.crc32 of the block id, with separate streams for the campaign and the
  policy, so runs are paired across policies and reproducible. All test-block results were re-run.
- Added comparators and outcomes: oracle upper bound; static plan from a risk map built on an earlier survey;
  capacity of 5 households per safe well; distance-decay switching.
- Earlier-survey scenario: static and adaptive policies also use the earlier results.

## v5 (2026-11-20, fixed on development blocks)
- Primary outcome: share of exposed households (own well truly unsafe) that are protected, i.e. own well tested
  (kit unsafe) and a tested-safe, truly safe well within the radius.
- Policy score: one-step expected gain per minute with look-ahead weight lam (src/f1/sim3.py); lam = 0.1 chosen on
  development blocks. Gaussian process chosen as the headline learner; all four learners reported.
- Comparator 're-planning only' (geometry, no safety information) to decompose the gain.
- Evaluation: cross-fitting over all 10 census blocks; 20 team-days per block; 10 runs.
- Replication on the Araihazar upazila 2012-13 survey (46,976 wells, HydroShare 48e779d1, CC BY 4.0): 52 blocks,
  cross-fitted, same design, no re-tuning.

## v6 (2026-11-20, fixed before the final runs; used in the article)
- Simulator src/f1/sim4.py, runner src/f1/experiment4.py. Per-policy look-ahead weight chosen on development blocks
  (results/f1_dev6_lam.csv): static 0.05; re-planning, adaptive and learning 0.1 (oracle 0.1).
  Uncertainty-sampling policy included as a comparator.
- Primary endpoint: share of exposed households protected after 5 team-days per 1,000 wells (interpolated per
  block), 50 µg/L, 100 m; learning (Gaussian process) vs door-to-door sweep, static plan and adaptive policy in both
  surveys; 6 comparisons, Holm-adjusted p from t-tests over blocks. All other analyses are secondary.
- Inference: repeated random splits (census 5, upazila survey 2) with cross-fitting; block means averaged over splits
  and runs; t-intervals over blocks; well-weighted pooled shares reported alongside block averages.
- 2012-13 survey: laboratory truth imputed per run from the 944 kit-laboratory pairs with the same kit reading; the
  recorded kit reading is the observation. Results also reported for the 38 blocks outside the census area.
- Earlier survey: links within 100 m only (1,076); old status used as calibrated P(safe now | old status).
- Further outcomes: misled households, false alarms, capacity, distance-decay switching; field realism with a daily
  return to a base well and 15% of owners absent.
