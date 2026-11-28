# Design log

The design history is summarised in Table A4 of the article. This file records the settings of each version.

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

## Additional analyses (2026-11-27, no new simulation runs)
- src/f1/inference.py on the v6 results: sign-flip permutation (exact for 10 census blocks) and Wilcoxon tests of the
  primary family, Holm-adjusted; the five confirmatory census blocks reported as rows of Table 2; leave-one-block-out
  range; 17 upazila super-blocks of 4 km; Monte Carlo SE of block means against the SD between blocks; Figure A1.
- Costing clarified: time-based (US$28.80 per team-day = 12 tests x US$2.40 all-in), time plus kit (US$0.17 per test)
  and per test (US$2.40 x tests performed); tests per day from the runs (Table A6).

## Sensitivity analyses (2026-11-28)
- experiment4.py --zero_as 5: kit readings of 0 set to 5 ug/L before log10 (fitting data and campaign observations),
  census and upazila survey (v6_zero5, v6_bd2012_zero5). --kit_nugget: kit-error variance of the 944 pairs (0.113)
  added to the GP nugget in the census (v6_kitnug). Defaults reproduce v6_main exactly. Table A7 (inference.py).

## Reproducibility audit (2026-11-28)
- src/f1/audit.py recomputes Table 2, Holm, permutation/Wilcoxon (Table A5), costs (Table A6), Table A7 and the
  kit-error variance independently from the campaign results; 116/116 checks match (results/f1_v6_audit.csv).

## Corrections and checks after the fifth review (2026-11-28)
- sim4.run4: under --daily_return a well is chosen only if the walk to it, the test and the walk back to the base fit in the
  working day; otherwise the team walks back and the next day starts at the base (previously the team was moved to the base
  without travel time). v6_field re-run. Main runs unchanged (identical outputs checked).
- sim4.run4: the learner is conditioned only on wells tested in the campaign; earlier-survey wells enter only through the
  calibrated P(safe now | old status). v6_bd2012_old re-run (lead over the static plan in 11 blocks 4.7 -> 3.3 points).
- inference.signflip_p: Monte Carlo p-values use (b + 1)/(B + 1).
- New metric prot_cap_opt (maximum-flow capacity assignment); v6_main and v6_bd2012 re-run with identical existing columns.
- New runs: v6_bd2012_boot (bootstrap of the 944 pairs, 20 replicates), v6_buffer and v6_bd2012_buffer (600 m exclusion
  buffer), v6_t29 and v6_bd2012_t29 (29 min per test, about 12 tests per day). calibration.py: probability calibration.
- Development tuning: results/run_dev_v6.sh; the rerun with the final code (f1_dev6_lam_rerun.csv) selects the same weights
  as the original f1_dev6_lam.csv (earlier development version of the simulator).
- audit.py rewritten: every cell of Table 2 and Tables A1-A8, A10 plus abstract and selected text numbers; rounding-only
  tolerance; exits with status 1 on any mismatch (3,495 checks OK, 2 skipped). Tests: tests/test_sim4.py, test_inference_audit.py.
- Pinned environment (requirements.txt), src/verify_data.py, CI workflow, CC BY licence file, S->A table naming.

## Final verification (2026-11-28, sixth review)
- audit.py also recomputes Table A9 from the saved per-well calibration predictions (results/f1_v6_calibration_predictions.csv.gz)
  and the Table A8 share of safe wells from the saved bootstrap draws (results/f1_v6_boot_truth.csv); 3,655 checks, none skipped
  with the data present. The audit found one wrong number in the text (under-prediction 0.15 -> 0.22), now corrected.
- Tests: GP probabilities equal scikit-learn's GaussianProcessRegressor with the same fixed kernel; counts at 360/720/1,080 min
  with and without daily return; hand-computed walk-home case (50 vs 52 wells). 29 tests.
- Scopus check: results/scopus_citation_check.csv (EID, DOI, year, match flags by title and by author names; 0 citations;
  one author-name match is a different Chipeta, a 2005 Malawi water-sector report).
- 2012-13 workbook provenance stated in Methods and data/README.md.
- manuscript/check_submission.py: submission PDF, zip source and figures match the current audited sources.
