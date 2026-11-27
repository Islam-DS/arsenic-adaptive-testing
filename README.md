# Learning where to test: adaptive arsenic testing of tube wells

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24566075.svg)](https://doi.org/10.5281/zenodo.24566075)

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
| `sim4.py` | campaign simulator used in the article: policies, protection outcome, misled households, capacity, distance decay, field realism |
| `experiment4.py` | runs campaigns with cross-fitting over repeated random splits |
| `figures6.py` | primary endpoint (Holm-adjusted), Tables S1-S3 and Figures 2-6 |
| `figures.py` | Figure 1 (map) |

Earlier development versions of the simulator and runner (`sim3.py`, `experiment.py`, `experiment3.py`) are
kept because `sim4.py` and `experiment4.py` import parts of them; the design history is summarised in Table S4 of the
article. `tests/test_sim2.py` contains sanity tests on synthetic blocks.

## Reproduce

Python 3.14 with numpy, scipy, pandas, scikit-learn 1.8, lightgbm 4.6, joblib, matplotlib, pymupdf, python-docx and markdown.

```
bash results/run_v6.sh                  # all campaigns used in the article (several hours on 14 cores)
python -X utf8 src/f1/figures6.py       # primary endpoint, Tables S1-S3, Figures 2-6
python -X utf8 -c "import sys; sys.path.insert(0, 'src/f1'); import figures; figures.fig_map()"   # Figure 1
python -X utf8 manuscript/build_latex.py   # LaTeX manuscript (MDPI template) and PDF
```

The results of the runs reported in the article are included in `results/` (`f1_v6_*.csv`; `f1_v6_primary.csv` holds
the primary endpoint), so the tables and figures can be regenerated without re-running the campaigns.

## Licence

Code: MIT (see `LICENSE`). Results, manuscript text and figures: CC BY 4.0.

## Citation

See `CITATION.cff`.
