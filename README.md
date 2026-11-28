# Learning where to test: adaptive arsenic testing of tube wells

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24578227.svg)](https://doi.org/10.5281/zenodo.24578227)

Code and results for the article

> Islam, M.M.; Yerkos, A.; Kunicina, N. Learning Where to Test: Adaptive Arsenic Testing of Tube Wells Could Protect
> More Exposed Households for the Same Field Effort in Bangladesh. Submitted to *Sustainability* (MDPI), 2026.

The study replays well-testing campaigns on two open surveys of tube wells in Araihazar upazila, Bangladesh, and compares
door-to-door testing, a depth-based advance plan and adaptive policies that choose the next well to test after every
field-kit result, with and without a spatial model of arsenic (Gaussian process). The outcome is the share of exposed
households protected, that is, households whose own unsafe well has been tested and that have a truly safe tested well
within 100 m.

## Data (not redistributed)

See [`data/README.md`](data/README.md) for sources, licences and checksums:

- 2000-2001 laboratory census of 6,595 wells and 944 kit-laboratory pairs, HydroShare doi:10.4211/hs.8e1373d87419447c945625af13f0a2ea (CC BY 4.0)
- 2012-2013 upazila field-kit survey of 46,976 wells, HydroShare doi:10.4211/hs.48e779d1aeac49609e3077625092e31b (CC BY 4.0)
- national village summary of the 2000-2005 survey, Supporting Information of Jamil et al. 2019, doi:10.1021/acs.est.9b01375 (context only)

## Code (`src/f1/`)

| File | Purpose |
|---|---|
| `data.py` | loads the 2000-2001 census, cleans it, builds 2 km blocks |
| `data_ext.py` | loads the 2012-2013 upazila survey and links wells to the census |
| `sim.py` | depth prior and Gaussian-process hyperparameters |
| `sim2.py` | campaign state, field-kit error model and learners (Gaussian process, kNN, random forest, LightGBM) |
| `sim4.py` | campaign simulator used in the article: policies, protection outcome, misled households, capacity (greedy and maximum-flow), distance decay, field realism (daily return to a base well, absent owners) |
| `experiment4.py` | runs campaigns with cross-fitting over repeated random splits; options for the sensitivity analyses (zero readings, kit-error nugget, bootstrap of the kit-laboratory pairs, spatial exclusion buffer) |
| `select_lambda.py` | selection of the look-ahead weights from the development run |
| `figures6.py` | primary endpoint (Holm-adjusted), Tables A1-A3 and Figures 2-6 |
| `inference.py` | permutation and Wilcoxon tests, confirmatory blocks, super-blocks, Monte Carlo error, costings, sensitivity and bootstrap analyses (Tables A5-A8, A10) and Figure A1 |
| `calibration.py` | calibration of the predicted probabilities on held-out blocks (Table A9, Figure A2) |
| `figures.py` | Figure 1 (map) |
| `audit.py` | independent recomputation of every cell of Table 2 and Tables A1-A10 and of the numbers in the abstract and selected sentences (Table A9 from the saved per-well predictions; the Gaussian-process predictions are checked against scikit-learn in the tests); exits with an error if any value differs beyond rounding (`results/f1_v6_audit.csv`). Checks that need the raw data are marked SKIPPED when the data are absent, so a pass without the data means that every check that could be run passed; with the data, 3,655 checks pass and none is skipped. |

Also: `src/verify_data.py` (checks the input files against their SHA-256 checksums); `src/scopus_citations.py` (the
citation check behind the Scopus statement in the Introduction; the export cannot be redistributed, so record
identifiers and match flags are in `results/scopus_citation_check.csv`); `manuscript/check_submission.py` (checks that
the files in `submission/` were generated from the current, audited sources and figures).
Earlier development versions of the simulator and runner (`sim3.py`, `experiment.py`, `experiment3.py`) are kept
because `sim4.py` and `experiment4.py` import parts of them; the design history is in Table A4 of the article.

## Tests

`tests/test_sim4.py` tests the simulator used in the article (bounds and monotonicity of protection, outcome
definitions, capacity rules, daily return, absent owners, reproducibility, no information from untested wells, no use of
earlier-survey values by the Gaussian process); `tests/test_inference_audit.py` tests the permutation p-values and the
audit's comparison rules; `tests/test_sim2.py` covers the earlier simulator. Run `python -X utf8 -m pytest tests -q`.
GitHub Actions runs the tests and the audit on every push (`.github/workflows/ci.yml`).

## Reproduce

Python 3.14.0 with the exact versions in `requirements.txt` (`pip install -r requirements.txt`).

```
python -X utf8 src/verify_data.py       # input files present and unchanged (see data/README.md)
bash results/run_dev_v6.sh              # development-stage tuning of the look-ahead weights (5 census blocks)
bash results/run_v6.sh                  # all campaigns used in the article (about 1.5 hours on 14 cores)
python -X utf8 src/f1/figures6.py       # primary endpoint, Tables A1-A3, Figures 2-6
python -X utf8 src/f1/inference.py      # further tests, costings, sensitivity and bootstrap, Tables A5-A8, A10, Figure A1
python -X utf8 src/f1/calibration.py    # Table A9, Figure A2
python -X utf8 -c "import sys; sys.path.insert(0, 'src/f1'); import figures; figures.fig_map()"   # Figure 1
python -X utf8 manuscript/make_appendix.py  # Appendices A-B (Figures A1-A2, Tables A1-A10) from the results
python -X utf8 src/f1/audit.py          # independent check of the published numbers
python -X utf8 manuscript/build_latex.py   # LaTeX manuscript (MDPI template) and PDF
python -X utf8 manuscript/check_submission.py  # submitted PDF, LaTeX source and figures match the audited sources
```

The results of the runs reported in the article are included in `results/` (`f1_v6_*.csv`; `f1_v6_primary.csv` holds
the primary endpoint), so the tables and figures can be regenerated without re-running the campaigns. The weights were
chosen on `results/f1_dev6_lam.csv`, produced by an earlier development version of the simulator; `run_dev_v6.sh`
repeats that run with the final code (`f1_dev6_lam_rerun.csv`), which gives slightly different numbers and selects the
same weights.

## Licence

Code (`src/`, `tests/`, `manuscript/*.py`, `results/*.sh`): MIT, see `LICENSE`. Results, manuscript text and figures:
CC BY 4.0, see `LICENSE-CC-BY-4.0.md`. Input data: the licences of their sources, see `data/README.md`.

## Citation

See `CITATION.cff`.
