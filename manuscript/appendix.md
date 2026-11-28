# Appendix A. Details of the Score, the Gaussian Process, the Costings and the Inference

## Score

The score $g_j$ of Equation (1) is computed for every untested well *j*. $N(j)$ is the set of wells within the radius *r* of *j*, including *j*. For each household *i* (one per well), $\mathrm{inf}_i = 1$ if its well has been tested and read unsafe, $\mathrm{unt}_i = 1$ if its well is untested, and $c_i$ is the probability that a tested well within *r* of *i* has read safe. For the adaptive, re-planning and learning policies, $c_i = 1$ as soon as such a well exists and 0 otherwise. The static plan never looks at results: it treats a tested well *k* as informing its household with probability $1 - p_k$ and sets $c_i = 1 - \prod_k (1 - p_k)$ over the tested wells *k* within *r* of *i*. The terms of Equation (1) are $E_j = \sum_{i \in N(j)} \mathrm{inf}_i (1 - c_i)$, $S_j = c_j$, $U_j = \sum_{i \in N(j), i \neq j} \mathrm{unt}_i (1 - c_i)(1 - p_i)$ and $Q_j = 1 - \prod_{i \in N(j), i \neq j, \mathrm{unt}_i = 1} (1 - p_i)$. The expected gain is divided by the test time (20 min) plus the walking time from the team's position (straight-line distance × 1.3 at 3 km/h). Probabilities are clipped to $[10^{-9}, 1 - 10^{-9}]$ and ties are broken at random. Door-to-door testing goes to the nearest untested well. Uncertainty sampling maximises $p_j (1 - p_j)$ times the number of untested wells within *r*, per minute. The oracle uses $p_j = 1$ for truly safe wells and 0 otherwise. Outcomes are evaluated per household, so a household with several safe wells nearby is counted once.

## Depth Prior, Gaussian Process and Kit Readings

Depth classes are <30, 30-50, 50-70, 70-100, 100-150 and ≥150 ft, and unknown depth. For each class the prior mean $m$ and standard deviation $\sigma$ of $y = \log_{10} \max(\mathrm{As}, 1)$ are estimated on the fitting half. The residual $z = y - m$ is modelled as a zero-mean Gaussian process with covariance $C(h) = \sigma_s^2 \exp(-h/\rho)$ and nugget $\tau^2$, on planar coordinates in metres (local projection, not standardised). Hyperparameters are fitted by maximum likelihood (scikit-learn GaussianProcessRegressor with a constant kernel times a Matérn kernel with $\nu = 0.5$, plus a white-noise kernel; bounds 0.001-5, 10-5,000 m and 0.0001-2) on 1,500 shallow wells drawn at random from the fitting half (one subsample per fitting half and threshold); the fitted hyperparameters are reused across the campaign replicates of that split. Deep (>100 ft) and shallow wells form separate layers with the same hyperparameters, and an untested well is conditioned only on tested wells of its own layer (wells of unknown depth are shallow). The posterior is computed by Cholesky decomposition.

Kit readings are ordered categories (0, 10, 25, 50, 100, 200, 300, 500 and about 1000 µg/L) and enter the model as $y = \log_{10} \max(\mathrm{reading}, 1)$, so a reading of 0 gives $y = 0$. In the upazila survey the model is fitted to kit readings, so its nugget includes kit error. In the census it is fitted to laboratory values, which the simulator treats as the truth, so its nugget contains no kit error. In the sensitivity analysis of Table A7, the kit-error variance $\kappa^2 = 0.113$ (variance of $\log_{10}$ kit minus $\log_{10}$ laboratory value over the 944 pairs) is added once to the census nugget. Because $\kappa^2$ is measured against the laboratory value, which is the truth by construction, it contains no laboratory error and nothing is counted twice. The covariance of the observed readings becomes $\mathbf{K} + (\tau^2 + \kappa^2)\mathbf{I}$, and the predictive variance then refers to the kit reading the team will see. In the census, the kit reading for a well is drawn at random from the 25 of the 944 kit-laboratory pairs whose laboratory values are closest (in $\log_{10}$) to the well's laboratory value. In the upazila survey, the recorded kit reading is observed and the true laboratory value of each well is drawn for each run from the pairs with the same kit reading.

## Field Realism, Productivity and Capacity

Under field realism the team starts each day at a base well (the well nearest the block centroid). A well is chosen only if the walk to it, the test and the walk back to the base all fit in the remaining working day; when no well fits, the team walks back and the next day starts at the base. A visit fails with probability 0.15 (owner absent); the time is spent and the well can be retried on a later day. The field-calibrated scenario lengthens each test to 29 minutes, which gives about 12 tests per team-day, as reported in the field. Without these options the working time is continuous, so a team that ends a day far from where it started begins the next day there. Capacity-limited protection assigns informed exposed households to truly safe sources within *r*, each source serving at most five households, nearest pair first; a maximum-flow assignment, which maximises the number assigned, is reported alongside (Table A1). In the earlier-survey scenario, wells with a 2000-2001 value enter the score only through the calibrated probability $P(\text{safe now} \mid \text{old status})$ until they are tested; their old values are not used as Gaussian-process observations.

## Costings

Persons protected are exposed households protected × 8 persons per well. The time-based costing charges US$28.80 per team-day, the all-in cost of the 12 tests one tester completed per day in Araihazar in 2012-2013 (US$2.40 per test, including labour, supervision, supplies and a placard). The time-plus-kit costing charges 12 × (US$2.40 - US$0.17) per team-day plus US$0.17 for each test performed, separating kit consumables from the daily cost of the team. The per-test costing charges US$2.40 for every test performed. Costs are pooled over blocks at the primary effort (Table A6).

## Inference

For a comparison, $d_b$ is the mean over the runs of block *b* of the paired difference (learning minus comparator). The *t*-interval and *t*-test use the $d_b$. The sign-flip permutation test compares $|\bar{d}|$ with its distribution when the signs of the $d_b$ are flipped: by exact enumeration of all $2^n$ patterns for up to 16 units, and otherwise with $B = 200{,}000$ random patterns and $p = (b + 1)/(B + 1)$, where $b$ is the number of patterns at least as extreme as the observed one. The Wilcoxon signed-rank test is two-sided. Leave-one-block-out gives the mean of the $d_b$ with each block removed in turn. For the super-blocks, the 2 km grid indices of the upazila blocks are halved, giving 17 cells of 4 km × 4 km within which the $d_b$ are averaged. The Monte Carlo standard error is the standard deviation of the run-level differences within a block divided by the square root of the number of runs, averaged over blocks (Table A5, Figure A1).

## Further Checks of the Evaluation

*Imputed truth.* The 944 kit-laboratory pairs were resampled with replacement; each of 20 replicates of the upazila campaigns (2 splits × 10 runs) imputed the laboratory truth from its own resample, shared by all blocks and policies (Table A8). The distribution of laboratory values within each kit category is in Table A10. *Exclusion buffer.* Fitting wells within 600 m (about twice the fitted range) of any evaluated well were removed before the depth prior and the Gaussian process were fitted (Table A7). During a campaign the Gaussian process is conditioned only on wells tested in the evaluated block, so the fitting half supplies only the prior and the hyperparameters. *Calibration.* On held-out blocks, a random 5% or 20% of the wells were treated as tested and the probability that each remaining well is truly safe was predicted (Table A9, Figure A2).

# Appendix B. Further Results

![**Figure A1.** Paired difference in exposed households protected (learning minus comparator, percentage points) after 5 team-days per 1,000 wells (50 µg/L, 100 m) in every block. Open circles: the five confirmatory census blocks; diamonds: mean over blocks with 95% *t*-interval.](figures/figS1_blocks.png)

<!-- spec: L{3.0cm}L{2.0cm}L{1.0cm}L{1.6cm}R{1.0cm}R{1.4cm}L{2.4cm}R{0.9cm}R{0.8cm}R{0.9cm} -->
**Table A1.** All comparisons of the learning policy (Gaussian process) with other policies. Effort 'primary' is 5 team-days per 1,000 wells; the capacity and expected-switching outcomes are after ten team-days. Difference in percentage points of exposed households protected (mean over blocks, 95% *t*-interval); *p*, paired *t*-test over blocks (unadjusted); blocks favouring learning; relative gain in the well-weighted share.

| Analysis | Setting | Effort | Comparator | Learning (%) | Comparator (%) | Difference (95% CI) | *p* | Blocks | Relative gain (%) |
|---|---|---|---|---|---|---|---|---|---|
| 2000-01 census | 10 µg/L, 100 m | primary | Door-to-door | 8.2 | 4.4 | 3.8 (2.0, 5.5) | <0.001 | 10/10 | +74 |
|  | 10 µg/L, 100 m | primary | Static plan | 8.2 | 5.9 | 2.2 (0.6, 3.8) | 0.011 | 9/10 | +37 |
|  | 10 µg/L, 100 m | primary | Re-planning | 8.2 | 5.7 | 2.4 (0.5, 4.3) | 0.017 | 8/10 | +36 |
|  | 10 µg/L, 100 m | primary | Uncertainty | 8.2 | 2.4 | 5.8 (3.7, 7.8) | <0.001 | 10/10 | +218 |
|  | 10 µg/L, 100 m | primary | Adaptive | 8.2 | 7.6 | 0.5 (-0.1, 1.1) | 0.081 | 8/10 | +8 |
|  | 10 µg/L, 100 m | primary | Oracle | 8.2 | 8.8 | -0.6 (-1.7, 0.5) | 0.237 | 4/10 | -7 |
|  | 50 µg/L, 100 m | primary | Door-to-door | 10.3 | 6.3 | 4.0 (1.5, 6.5) | 0.006 | 10/10 | +61 |
|  | 50 µg/L, 100 m | primary | Static plan | 10.3 | 7.4 | 2.8 (0.9, 4.8) | 0.008 | 9/10 | +41 |
|  | 50 µg/L, 100 m | primary | Re-planning | 10.3 | 7.4 | 2.9 (0.6, 5.3) | 0.019 | 10/10 | +42 |
|  | 50 µg/L, 100 m | primary | Uncertainty | 10.3 | 4.8 | 5.5 (3.2, 7.8) | <0.001 | 10/10 | +123 |
|  | 50 µg/L, 100 m | primary | Adaptive | 10.3 | 9.0 | 1.3 (0.9, 1.8) | <0.001 | 10/10 | +17 |
|  | 50 µg/L, 100 m | primary | Oracle | 10.3 | 12.5 | -2.2 (-3.7, -0.6) | 0.011 | 1/10 | -17 |
|  | 10 µg/L, 100 m | 5 days | Door-to-door | 14.2 | 7.6 | 6.7 (2.9, 10.4) | 0.003 | 10/10 | +76 |
|  | 10 µg/L, 100 m | 5 days | Static plan | 14.2 | 9.8 | 4.4 (1.9, 6.9) | 0.003 | 10/10 | +40 |
|  | 10 µg/L, 100 m | 5 days | Adaptive | 14.2 | 13.9 | 0.4 (-0.8, 1.5) | 0.499 | 6/10 | +3 |
|  | 50 µg/L, 100 m | 5 days | Door-to-door | 17.8 | 10.8 | 7.0 (3.2, 10.8) | 0.002 | 10/10 | +61 |
|  | 50 µg/L, 100 m | 5 days | Static plan | 17.8 | 12.3 | 5.5 (2.3, 8.7) | 0.004 | 10/10 | +45 |
|  | 50 µg/L, 100 m | 5 days | Adaptive | 17.8 | 14.6 | 3.3 (1.6, 4.9) | 0.002 | 10/10 | +22 |
|  | 10 µg/L, 100 m | 10 days | Door-to-door | 25.4 | 16.2 | 9.2 (5.2, 13.2) | <0.001 | 10/10 | +53 |
|  | 10 µg/L, 100 m | 10 days | Static plan | 25.4 | 20.0 | 5.4 (2.3, 8.5) | 0.003 | 10/10 | +26 |
|  | 10 µg/L, 100 m | 10 days | Adaptive | 25.4 | 24.0 | 1.4 (-0.3, 3.1) | 0.094 | 9/10 | +5 |
|  | 50 µg/L, 100 m | 10 days | Door-to-door | 33.6 | 22.0 | 11.5 (5.6, 17.5) | 0.002 | 10/10 | +50 |
|  | 50 µg/L, 100 m | 10 days | Static plan | 33.6 | 24.0 | 9.6 (4.5, 14.6) | 0.002 | 10/10 | +41 |
|  | 50 µg/L, 100 m | 10 days | Adaptive | 33.6 | 28.8 | 4.8 (1.9, 7.7) | 0.004 | 10/10 | +17 |
|  | 10 µg/L, 100 m | 20 days | Door-to-door | 40.3 | 33.3 | 7.0 (3.9, 10.2) | <0.001 | 10/10 | +24 |
|  | 10 µg/L, 100 m | 20 days | Static plan | 40.3 | 34.6 | 5.6 (2.0, 9.2) | 0.007 | 10/10 | +16 |
|  | 10 µg/L, 100 m | 20 days | Adaptive | 40.3 | 37.1 | 3.1 (1.1, 5.1) | 0.006 | 9/10 | +9 |
|  | 50 µg/L, 100 m | 20 days | Door-to-door | 57.0 | 44.4 | 12.7 (6.8, 18.5) | <0.001 | 10/10 | +30 |
|  | 50 µg/L, 100 m | 20 days | Static plan | 57.0 | 43.9 | 13.1 (7.4, 18.9) | <0.001 | 10/10 | +34 |
|  | 50 µg/L, 100 m | 20 days | Adaptive | 57.0 | 49.7 | 7.3 (4.8, 9.8) | <0.001 | 10/10 | +16 |
| 2012-13 upazila survey | 10 µg/L, 100 m | primary | Door-to-door | 10.0 | 5.9 | 4.1 (2.4, 5.8) | <0.001 | 45/52 | +37 |
|  | 10 µg/L, 100 m | primary | Static plan | 10.0 | 7.7 | 2.3 (1.3, 3.4) | <0.001 | 43/52 | +23 |
|  | 10 µg/L, 100 m | primary | Re-planning | 10.0 | 6.6 | 3.3 (1.7, 5.0) | <0.001 | 38/52 | +21 |
|  | 10 µg/L, 100 m | primary | Uncertainty | 10.0 | 5.2 | 4.8 (3.6, 5.9) | <0.001 | 49/52 | +80 |
|  | 10 µg/L, 100 m | primary | Adaptive | 10.0 | 8.6 | 1.4 (0.5, 2.2) | 0.002 | 40/52 | +19 |
|  | 10 µg/L, 100 m | primary | Oracle | 10.0 | 13.8 | -3.8 (-5.7, -1.9) | <0.001 | 17/52 | -12 |
|  | 50 µg/L, 100 m | primary | Door-to-door | 15.6 | 7.2 | 8.4 (4.4, 12.4) | <0.001 | 44/52 | +53 |
|  | 50 µg/L, 100 m | primary | Static plan | 15.6 | 12.1 | 3.5 (1.7, 5.2) | <0.001 | 39/52 | +33 |
|  | 50 µg/L, 100 m | primary | Re-planning | 15.6 | 8.7 | 6.9 (3.1, 10.7) | <0.001 | 40/52 | +29 |
|  | 50 µg/L, 100 m | primary | Uncertainty | 15.6 | 9.9 | 5.7 (2.5, 8.9) | <0.001 | 44/52 | +49 |
|  | 50 µg/L, 100 m | primary | Adaptive | 15.6 | 13.4 | 2.2 (0.8, 3.6) | 0.003 | 40/52 | +20 |
|  | 50 µg/L, 100 m | primary | Oracle | 15.6 | 26.0 | -10.4 (-15.3, -5.5) | <0.001 | 11/52 | -18 |
|  | 10 µg/L, 100 m | 5 days | Door-to-door | 16.5 | 9.8 | 6.6 (3.9, 9.3) | <0.001 | 45/52 | +37 |
|  | 10 µg/L, 100 m | 5 days | Static plan | 16.5 | 12.1 | 4.4 (2.6, 6.2) | <0.001 | 43/52 | +26 |
|  | 10 µg/L, 100 m | 5 days | Adaptive | 16.5 | 13.6 | 2.8 (1.4, 4.3) | <0.001 | 42/52 | +22 |
|  | 50 µg/L, 100 m | 5 days | Door-to-door | 23.0 | 11.7 | 11.3 (6.4, 16.2) | <0.001 | 46/52 | +45 |
|  | 50 µg/L, 100 m | 5 days | Static plan | 23.0 | 18.3 | 4.7 (2.4, 6.9) | <0.001 | 44/52 | +29 |
|  | 50 µg/L, 100 m | 5 days | Adaptive | 23.0 | 20.1 | 2.9 (1.0, 4.8) | 0.004 | 39/52 | +16 |
|  | 10 µg/L, 100 m | 10 days | Door-to-door | 28.4 | 19.0 | 9.4 (6.2, 12.6) | <0.001 | 47/52 | +36 |
|  | 10 µg/L, 100 m | 10 days | Static plan | 28.4 | 21.9 | 6.5 (4.9, 8.2) | <0.001 | 49/52 | +32 |
|  | 10 µg/L, 100 m | 10 days | Adaptive | 28.4 | 23.7 | 4.7 (3.1, 6.3) | <0.001 | 45/52 | +21 |
|  | 50 µg/L, 100 m | 10 days | Door-to-door | 37.0 | 21.5 | 15.5 (10.0, 20.9) | <0.001 | 50/52 | +46 |
|  | 50 µg/L, 100 m | 10 days | Static plan | 37.0 | 28.6 | 8.3 (5.8, 10.8) | <0.001 | 47/52 | +37 |
|  | 50 µg/L, 100 m | 10 days | Adaptive | 37.0 | 32.0 | 4.9 (3.2, 6.7) | <0.001 | 45/52 | +17 |
|  | 10 µg/L, 100 m | 20 days | Door-to-door | 42.2 | 32.4 | 9.7 (7.0, 12.5) | <0.001 | 40/52 | +34 |
|  | 10 µg/L, 100 m | 20 days | Static plan | 42.2 | 34.0 | 8.1 (6.1, 10.2) | <0.001 | 40/52 | +34 |
|  | 10 µg/L, 100 m | 20 days | Adaptive | 42.2 | 37.5 | 4.6 (2.9, 6.4) | <0.001 | 38/52 | +17 |
|  | 50 µg/L, 100 m | 20 days | Door-to-door | 53.5 | 38.6 | 14.9 (10.7, 19.1) | <0.001 | 41/52 | +43 |
|  | 50 µg/L, 100 m | 20 days | Static plan | 53.5 | 42.8 | 10.7 (8.0, 13.5) | <0.001 | 40/52 | +41 |
|  | 50 µg/L, 100 m | 20 days | Adaptive | 53.5 | 48.5 | 5.0 (3.1, 6.9) | <0.001 | 39/52 | +13 |
| 2012-13, 38 blocks outside census area | 10 µg/L, 100 m | primary | Door-to-door | 9.9 | 5.5 | 4.4 (2.6, 6.3) | <0.001 | 34/38 | +51 |
|  | 10 µg/L, 100 m | primary | Static plan | 9.9 | 7.5 | 2.5 (1.2, 3.8) | <0.001 | 33/38 | +27 |
|  | 10 µg/L, 100 m | primary | Re-planning | 9.9 | 6.8 | 3.2 (1.6, 4.7) | <0.001 | 27/38 | +17 |
|  | 10 µg/L, 100 m | primary | Uncertainty | 9.9 | 5.1 | 4.9 (3.6, 6.2) | <0.001 | 36/38 | +86 |
|  | 10 µg/L, 100 m | primary | Adaptive | 9.9 | 8.8 | 1.2 (0.1, 2.2) | 0.029 | 28/38 | +18 |
|  | 10 µg/L, 100 m | primary | Oracle | 9.9 | 14.9 | -4.9 (-7.4, -2.4) | <0.001 | 10/38 | -15 |
|  | 50 µg/L, 100 m | primary | Door-to-door | 15.5 | 7.2 | 8.3 (3.6, 13.0) | <0.001 | 31/38 | +50 |
|  | 50 µg/L, 100 m | primary | Static plan | 15.5 | 12.5 | 3.1 (0.8, 5.4) | 0.011 | 27/38 | +24 |
|  | 50 µg/L, 100 m | primary | Re-planning | 15.5 | 8.6 | 6.9 (2.5, 11.3) | 0.003 | 28/38 | +22 |
|  | 50 µg/L, 100 m | primary | Uncertainty | 15.5 | 10.1 | 5.4 (1.2, 9.6) | 0.014 | 31/38 | +38 |
|  | 50 µg/L, 100 m | primary | Adaptive | 15.5 | 13.3 | 2.2 (0.3, 4.1) | 0.024 | 28/38 | +22 |
|  | 50 µg/L, 100 m | primary | Oracle | 15.5 | 28.4 | -12.9 (-19.5, -6.3) | <0.001 | 7/38 | -21 |
|  | 10 µg/L, 100 m | 5 days | Door-to-door | 18.4 | 10.7 | 7.8 (4.9, 10.7) | <0.001 | 35/38 | +50 |
|  | 10 µg/L, 100 m | 5 days | Static plan | 18.4 | 13.0 | 5.4 (3.2, 7.7) | <0.001 | 35/38 | +32 |
|  | 10 µg/L, 100 m | 5 days | Adaptive | 18.4 | 15.1 | 3.3 (1.4, 5.3) | 0.001 | 30/38 | +25 |
|  | 50 µg/L, 100 m | 5 days | Door-to-door | 24.9 | 12.9 | 12.0 (6.4, 17.5) | <0.001 | 32/38 | +41 |
|  | 50 µg/L, 100 m | 5 days | Static plan | 24.9 | 19.7 | 5.1 (2.1, 8.2) | 0.002 | 31/38 | +26 |
|  | 50 µg/L, 100 m | 5 days | Adaptive | 24.9 | 21.8 | 3.1 (0.5, 5.6) | 0.019 | 28/38 | +17 |
|  | 10 µg/L, 100 m | 10 days | Door-to-door | 30.5 | 20.5 | 10.0 (7.0, 13.0) | <0.001 | 36/38 | +41 |
|  | 10 µg/L, 100 m | 10 days | Static plan | 30.5 | 23.6 | 6.9 (5.0, 8.8) | <0.001 | 37/38 | +33 |
|  | 10 µg/L, 100 m | 10 days | Adaptive | 30.5 | 25.7 | 4.8 (2.8, 6.8) | <0.001 | 33/38 | +22 |
|  | 50 µg/L, 100 m | 10 days | Door-to-door | 39.4 | 23.5 | 15.9 (9.5, 22.2) | <0.001 | 36/38 | +42 |
|  | 50 µg/L, 100 m | 10 days | Static plan | 39.4 | 31.0 | 8.4 (5.1, 11.7) | <0.001 | 34/38 | +32 |
|  | 50 µg/L, 100 m | 10 days | Adaptive | 39.4 | 34.6 | 4.8 (2.7, 6.8) | <0.001 | 32/38 | +17 |
|  | 10 µg/L, 100 m | 20 days | Door-to-door | 43.1 | 33.9 | 9.1 (6.3, 12.0) | <0.001 | 27/38 | +36 |
|  | 10 µg/L, 100 m | 20 days | Static plan | 43.1 | 35.9 | 7.2 (4.8, 9.6) | <0.001 | 27/38 | +30 |
|  | 10 µg/L, 100 m | 20 days | Adaptive | 43.1 | 38.8 | 4.3 (2.3, 6.4) | <0.001 | 26/38 | +18 |
|  | 50 µg/L, 100 m | 20 days | Door-to-door | 54.9 | 41.1 | 13.8 (8.8, 18.9) | <0.001 | 28/38 | +39 |
|  | 50 µg/L, 100 m | 20 days | Static plan | 54.9 | 45.3 | 9.6 (6.3, 13.0) | <0.001 | 27/38 | +34 |
|  | 50 µg/L, 100 m | 20 days | Adaptive | 54.9 | 50.4 | 4.5 (2.3, 6.8) | <0.001 | 26/38 | +12 |
| Census, field realism | 50 µg/L, 100 m | primary | Door-to-door | 7.5 | 5.1 | 2.3 (0.2, 4.4) | 0.037 | 6/10 | +36 |
|  | 50 µg/L, 100 m | primary | Static plan | 7.5 | 6.4 | 1.1 (-0.8, 2.9) | 0.216 | 6/10 | +20 |
|  | 50 µg/L, 100 m | primary | Re-planning | 7.5 | 5.8 | 1.7 (0.2, 3.2) | 0.028 | 9/10 | +26 |
|  | 50 µg/L, 100 m | primary | Uncertainty | 7.5 | 3.8 | 3.7 (2.2, 5.1) | <0.001 | 10/10 | +98 |
|  | 50 µg/L, 100 m | primary | Adaptive | 7.5 | 6.8 | 0.7 (0.1, 1.2) | 0.024 | 9/10 | +9 |
|  | 50 µg/L, 100 m | primary | Oracle | 7.5 | 9.2 | -1.7 (-3.0, -0.4) | 0.016 | 1/10 | -19 |
|  | 50 µg/L, 100 m | 5 days | Door-to-door | 13.0 | 9.0 | 3.9 (0.7, 7.1) | 0.022 | 7/10 | +38 |
|  | 50 µg/L, 100 m | 5 days | Static plan | 13.0 | 10.7 | 2.3 (-0.2, 4.8) | 0.070 | 8/10 | +22 |
|  | 50 µg/L, 100 m | 5 days | Adaptive | 13.0 | 11.2 | 1.7 (0.3, 3.2) | 0.022 | 10/10 | +15 |
|  | 50 µg/L, 100 m | 10 days | Door-to-door | 26.3 | 18.3 | 8.0 (2.8, 13.2) | 0.007 | 9/10 | +41 |
|  | 50 µg/L, 100 m | 10 days | Static plan | 26.3 | 20.2 | 6.1 (1.7, 10.5) | 0.012 | 8/10 | +32 |
|  | 50 µg/L, 100 m | 10 days | Adaptive | 26.3 | 22.7 | 3.6 (1.2, 6.0) | 0.008 | 8/10 | +18 |
|  | 50 µg/L, 100 m | 20 days | Door-to-door | 47.9 | 33.9 | 14.0 (6.6, 21.3) | 0.002 | 10/10 | +39 |
|  | 50 µg/L, 100 m | 20 days | Static plan | 47.9 | 38.1 | 9.7 (4.9, 14.5) | 0.001 | 10/10 | +30 |
|  | 50 µg/L, 100 m | 20 days | Adaptive | 47.9 | 42.1 | 5.8 (2.9, 8.6) | 0.001 | 9/10 | +17 |
| Census, safe if kit ≤ half threshold | 10 µg/L, 100 m | primary | Door-to-door | 7.0 | 3.5 | 3.4 (1.8, 5.1) | 0.001 | 10/10 | +80 |
|  | 10 µg/L, 100 m | primary | Static plan | 7.0 | 4.4 | 2.6 (0.5, 4.6) | 0.019 | 10/10 | +54 |
|  | 10 µg/L, 100 m | primary | Adaptive | 7.0 | 6.8 | 0.2 (-0.4, 0.8) | 0.471 | 5/10 | +3 |
|  | 50 µg/L, 100 m | primary | Door-to-door | 10.4 | 5.6 | 4.8 (1.8, 7.8) | 0.006 | 10/10 | +77 |
|  | 50 µg/L, 100 m | primary | Static plan | 10.4 | 7.0 | 3.4 (1.2, 5.5) | 0.006 | 9/10 | +56 |
|  | 50 µg/L, 100 m | primary | Adaptive | 10.4 | 8.6 | 1.8 (0.8, 2.8) | 0.003 | 8/10 | +22 |
|  | 10 µg/L, 100 m | 5 days | Door-to-door | 12.2 | 6.2 | 6.0 (2.9, 9.1) | 0.002 | 10/10 | +84 |
|  | 10 µg/L, 100 m | 5 days | Static plan | 12.2 | 8.2 | 4.0 (1.2, 6.8) | 0.010 | 9/10 | +45 |
|  | 10 µg/L, 100 m | 5 days | Adaptive | 12.2 | 12.4 | -0.3 (-1.1, 0.6) | 0.485 | 4/10 | -1 |
|  | 50 µg/L, 100 m | 5 days | Door-to-door | 18.0 | 9.6 | 8.4 (4.0, 12.8) | 0.002 | 10/10 | +81 |
|  | 50 µg/L, 100 m | 5 days | Static plan | 18.0 | 11.7 | 6.3 (3.1, 9.6) | 0.002 | 10/10 | +59 |
|  | 50 µg/L, 100 m | 5 days | Adaptive | 18.0 | 13.7 | 4.3 (1.2, 7.3) | 0.012 | 9/10 | +29 |
|  | 10 µg/L, 100 m | 10 days | Door-to-door | 21.5 | 13.1 | 8.4 (4.8, 12.0) | <0.001 | 10/10 | +63 |
|  | 10 µg/L, 100 m | 10 days | Static plan | 21.5 | 16.1 | 5.4 (1.6, 9.1) | 0.010 | 10/10 | +33 |
|  | 10 µg/L, 100 m | 10 days | Adaptive | 21.5 | 20.9 | 0.6 (-0.6, 1.9) | 0.290 | 6/10 | +3 |
|  | 50 µg/L, 100 m | 10 days | Door-to-door | 32.3 | 20.0 | 12.3 (6.0, 18.7) | 0.002 | 10/10 | +59 |
|  | 50 µg/L, 100 m | 10 days | Static plan | 32.3 | 21.8 | 10.5 (5.7, 15.3) | <0.001 | 10/10 | +53 |
|  | 50 µg/L, 100 m | 10 days | Adaptive | 32.3 | 27.3 | 5.0 (2.1, 7.8) | 0.003 | 10/10 | +20 |
|  | 10 µg/L, 100 m | 20 days | Door-to-door | 33.1 | 26.6 | 6.6 (3.4, 9.7) | 0.001 | 10/10 | +29 |
|  | 10 µg/L, 100 m | 20 days | Static plan | 33.1 | 28.7 | 4.5 (3.0, 6.0) | <0.001 | 10/10 | +17 |
|  | 10 µg/L, 100 m | 20 days | Adaptive | 33.1 | 31.5 | 1.6 (0.5, 2.8) | 0.009 | 9/10 | +7 |
|  | 50 µg/L, 100 m | 20 days | Door-to-door | 53.9 | 41.1 | 12.8 (7.3, 18.4) | <0.001 | 10/10 | +33 |
|  | 50 µg/L, 100 m | 20 days | Static plan | 53.9 | 40.1 | 13.8 (8.0, 19.7) | <0.001 | 10/10 | +41 |
|  | 50 µg/L, 100 m | 20 days | Adaptive | 53.9 | 48.0 | 5.9 (3.4, 8.4) | <0.001 | 10/10 | +16 |
| Census, other radius | 50 µg/L, 50 m | primary | Door-to-door | 8.6 | 4.1 | 4.4 (1.7, 7.2) | 0.006 | 9/10 | +101 |
|  | 50 µg/L, 50 m | primary | Static plan | 8.6 | 5.6 | 3.0 (1.5, 4.5) | 0.002 | 9/10 | +61 |
|  | 50 µg/L, 50 m | primary | Adaptive | 8.6 | 7.0 | 1.6 (0.2, 3.0) | 0.027 | 8/10 | +31 |
|  | 50 µg/L, 200 m | primary | Door-to-door | 10.1 | 7.2 | 2.9 (0.3, 5.4) | 0.031 | 9/10 | +31 |
|  | 50 µg/L, 200 m | primary | Static plan | 10.1 | 8.8 | 1.2 (-1.1, 3.5) | 0.257 | 5/10 | +14 |
|  | 50 µg/L, 200 m | primary | Adaptive | 10.1 | 9.0 | 1.1 (0.3, 1.9) | 0.011 | 9/10 | +12 |
|  | 50 µg/L, 50 m | 5 days | Door-to-door | 13.6 | 7.1 | 6.5 (2.5, 10.5) | 0.005 | 9/10 | +90 |
|  | 50 µg/L, 50 m | 5 days | Static plan | 13.6 | 8.9 | 4.7 (2.2, 7.2) | 0.002 | 9/10 | +58 |
|  | 50 µg/L, 50 m | 5 days | Adaptive | 13.6 | 10.5 | 3.0 (1.0, 5.0) | 0.009 | 9/10 | +32 |
|  | 50 µg/L, 200 m | 5 days | Door-to-door | 18.4 | 12.8 | 5.6 (1.7, 9.6) | 0.010 | 10/10 | +37 |
|  | 50 µg/L, 200 m | 5 days | Static plan | 18.4 | 15.0 | 3.4 (-0.3, 7.2) | 0.068 | 8/10 | +21 |
|  | 50 µg/L, 200 m | 5 days | Adaptive | 18.4 | 16.2 | 2.2 (1.1, 3.4) | 0.001 | 9/10 | +13 |
|  | 50 µg/L, 50 m | 10 days | Door-to-door | 24.0 | 14.8 | 9.2 (3.8, 14.6) | 0.004 | 9/10 | +62 |
|  | 50 µg/L, 50 m | 10 days | Static plan | 24.0 | 18.7 | 5.3 (2.5, 8.2) | 0.002 | 9/10 | +34 |
|  | 50 µg/L, 50 m | 10 days | Adaptive | 24.0 | 19.7 | 4.4 (2.3, 6.4) | <0.001 | 10/10 | +27 |
|  | 50 µg/L, 200 m | 10 days | Door-to-door | 38.4 | 25.6 | 12.8 (5.1, 20.4) | 0.004 | 10/10 | +43 |
|  | 50 µg/L, 200 m | 10 days | Static plan | 38.4 | 31.8 | 6.6 (2.4, 10.8) | 0.006 | 10/10 | +22 |
|  | 50 µg/L, 200 m | 10 days | Adaptive | 38.4 | 33.4 | 5.0 (1.8, 8.1) | 0.006 | 10/10 | +16 |
|  | 50 µg/L, 50 m | 20 days | Door-to-door | 38.7 | 30.1 | 8.6 (4.2, 13.0) | 0.002 | 10/10 | +31 |
|  | 50 µg/L, 50 m | 20 days | Static plan | 38.7 | 33.2 | 5.5 (2.6, 8.3) | 0.002 | 10/10 | +21 |
|  | 50 µg/L, 50 m | 20 days | Adaptive | 38.7 | 33.5 | 5.2 (1.5, 8.9) | 0.012 | 10/10 | +20 |
|  | 50 µg/L, 200 m | 20 days | Door-to-door | 67.4 | 52.8 | 14.7 (7.0, 22.3) | 0.002 | 10/10 | +27 |
|  | 50 µg/L, 200 m | 20 days | Static plan | 67.4 | 57.4 | 10.0 (4.2, 15.8) | 0.004 | 10/10 | +22 |
|  | 50 µg/L, 200 m | 20 days | Adaptive | 67.4 | 61.3 | 6.1 (2.5, 9.8) | 0.004 | 10/10 | +11 |
| Census, no kit error | 10 µg/L, 100 m | primary | Door-to-door | 9.2 | 5.0 | 4.2 (1.8, 6.5) | 0.003 | 10/10 | +66 |
|  | 10 µg/L, 100 m | primary | Static plan | 9.2 | 6.5 | 2.7 (0.5, 4.9) | 0.021 | 9/10 | +35 |
|  | 10 µg/L, 100 m | primary | Adaptive | 9.2 | 8.8 | 0.4 (-0.2, 0.9) | 0.174 | 7/10 | +4 |
|  | 50 µg/L, 100 m | primary | Door-to-door | 12.3 | 6.9 | 5.4 (2.2, 8.7) | 0.005 | 10/10 | +71 |
|  | 50 µg/L, 100 m | primary | Static plan | 12.3 | 8.6 | 3.6 (1.5, 5.7) | 0.004 | 10/10 | +48 |
|  | 50 µg/L, 100 m | primary | Adaptive | 12.3 | 10.6 | 1.7 (0.6, 2.8) | 0.006 | 9/10 | +18 |
|  | 10 µg/L, 100 m | 5 days | Door-to-door | 16.0 | 8.3 | 7.7 (3.5, 11.9) | 0.003 | 9/10 | +77 |
|  | 10 µg/L, 100 m | 5 days | Static plan | 16.0 | 10.7 | 5.2 (2.2, 8.3) | 0.004 | 10/10 | +41 |
|  | 10 µg/L, 100 m | 5 days | Adaptive | 16.0 | 15.6 | 0.4 (-0.1, 0.9) | 0.077 | 7/10 | +3 |
|  | 50 µg/L, 100 m | 5 days | Door-to-door | 20.9 | 12.1 | 8.8 (4.3, 13.3) | 0.002 | 10/10 | +67 |
|  | 50 µg/L, 100 m | 5 days | Static plan | 20.9 | 14.6 | 6.3 (3.2, 9.4) | 0.001 | 10/10 | +47 |
|  | 50 µg/L, 100 m | 5 days | Adaptive | 20.9 | 17.5 | 3.4 (1.8, 5.0) | <0.001 | 10/10 | +21 |
|  | 10 µg/L, 100 m | 10 days | Door-to-door | 28.9 | 17.8 | 11.1 (5.0, 17.3) | 0.003 | 10/10 | +55 |
|  | 10 µg/L, 100 m | 10 days | Static plan | 28.9 | 21.9 | 7.0 (2.9, 11.1) | 0.004 | 10/10 | +29 |
|  | 10 µg/L, 100 m | 10 days | Adaptive | 28.9 | 27.1 | 1.8 (-0.0, 3.7) | 0.051 | 6/10 | +6 |
|  | 50 µg/L, 100 m | 10 days | Door-to-door | 39.3 | 24.6 | 14.7 (7.5, 21.9) | 0.001 | 10/10 | +55 |
|  | 50 µg/L, 100 m | 10 days | Static plan | 39.3 | 28.7 | 10.7 (6.2, 15.1) | <0.001 | 10/10 | +40 |
|  | 50 µg/L, 100 m | 10 days | Adaptive | 39.3 | 34.3 | 5.0 (2.9, 7.0) | <0.001 | 10/10 | +15 |
|  | 10 µg/L, 100 m | 20 days | Door-to-door | 44.7 | 36.8 | 7.9 (3.8, 12.1) | 0.002 | 10/10 | +23 |
|  | 10 µg/L, 100 m | 20 days | Static plan | 44.7 | 39.4 | 5.3 (2.9, 7.8) | <0.001 | 10/10 | +14 |
|  | 10 µg/L, 100 m | 20 days | Adaptive | 44.7 | 41.4 | 3.2 (0.4, 6.1) | 0.030 | 6/10 | +8 |
|  | 50 µg/L, 100 m | 20 days | Door-to-door | 65.5 | 50.6 | 14.9 (8.3, 21.5) | <0.001 | 10/10 | +29 |
|  | 50 µg/L, 100 m | 20 days | Static plan | 65.5 | 52.2 | 13.3 (7.3, 19.3) | <0.001 | 10/10 | +29 |
|  | 50 µg/L, 100 m | 20 days | Adaptive | 65.5 | 58.8 | 6.7 (3.9, 9.6) | <0.001 | 10/10 | +13 |
| Census, thinned wells | 50 µg/L, 100 m, 25% of wells | primary | Door-to-door | 6.9 | 4.4 | 2.5 (0.3, 4.7) | 0.030 | 7/10 | +43 |
|  | 50 µg/L, 100 m, 25% of wells | primary | Static plan | 6.9 | 5.6 | 1.2 (0.3, 2.2) | 0.013 | 9/10 | +19 |
|  | 50 µg/L, 100 m, 25% of wells | primary | Adaptive | 6.9 | 6.4 | 0.5 (-0.2, 1.2) | 0.145 | 7/10 | +8 |
|  | 50 µg/L, 100 m, 50% of wells | primary | Door-to-door | 8.5 | 4.1 | 4.4 (1.1, 7.7) | 0.015 | 9/10 | +90 |
|  | 50 µg/L, 100 m, 50% of wells | primary | Static plan | 8.5 | 6.1 | 2.5 (0.9, 4.0) | 0.005 | 7/10 | +47 |
|  | 50 µg/L, 100 m, 50% of wells | primary | Adaptive | 8.5 | 7.5 | 1.1 (0.2, 1.9) | 0.019 | 8/10 | +18 |
|  | 50 µg/L, 100 m, 25% of wells | 5 days | Door-to-door | 35.7 | 28.9 | 6.8 (3.9, 9.7) | <0.001 | 10/10 | +23 |
|  | 50 µg/L, 100 m, 25% of wells | 5 days | Static plan | 35.7 | 32.6 | 3.1 (1.4, 4.8) | 0.003 | 9/10 | +10 |
|  | 50 µg/L, 100 m, 25% of wells | 5 days | Adaptive | 35.7 | 33.3 | 2.4 (0.8, 4.0) | 0.009 | 9/10 | +7 |
|  | 50 µg/L, 100 m, 50% of wells | 5 days | Door-to-door | 27.4 | 16.6 | 10.8 (2.8, 18.7) | 0.014 | 9/10 | +63 |
|  | 50 µg/L, 100 m, 50% of wells | 5 days | Static plan | 27.4 | 20.4 | 6.9 (3.3, 10.5) | 0.002 | 10/10 | +38 |
|  | 50 µg/L, 100 m, 50% of wells | 5 days | Adaptive | 27.4 | 24.7 | 2.7 (0.5, 4.9) | 0.021 | 8/10 | +14 |
|  | 50 µg/L, 100 m, 25% of wells | 10 days | Door-to-door | 46.6 | 44.7 | 1.9 (-0.2, 4.0) | 0.070 | 4/10 | +6 |
|  | 50 µg/L, 100 m, 25% of wells | 10 days | Static plan | 46.6 | 45.4 | 1.2 (-1.2, 3.6) | 0.295 | 3/10 | +4 |
|  | 50 µg/L, 100 m, 25% of wells | 10 days | Adaptive | 46.6 | 45.5 | 1.1 (-0.8, 3.0) | 0.234 | 3/10 | +4 |
|  | 50 µg/L, 100 m, 50% of wells | 10 days | Door-to-door | 45.5 | 35.1 | 10.4 (2.1, 18.8) | 0.020 | 10/10 | +30 |
|  | 50 µg/L, 100 m, 50% of wells | 10 days | Static plan | 45.5 | 38.1 | 7.4 (3.5, 11.4) | 0.002 | 9/10 | +24 |
|  | 50 µg/L, 100 m, 50% of wells | 10 days | Adaptive | 45.5 | 40.6 | 4.9 (1.7, 8.1) | 0.008 | 9/10 | +14 |
|  | 50 µg/L, 100 m, 25% of wells | 20 days | Door-to-door | 49.2 | 49.2 | 0.0 (0.0, 0.0) | - | 0/10 | +0 |
|  | 50 µg/L, 100 m, 25% of wells | 20 days | Static plan | 49.2 | 49.2 | 0.0 (0.0, 0.0) | - | 0/10 | +0 |
|  | 50 µg/L, 100 m, 25% of wells | 20 days | Adaptive | 49.2 | 49.2 | 0.0 (0.0, 0.0) | - | 0/10 | +0 |
|  | 50 µg/L, 100 m, 50% of wells | 20 days | Door-to-door | 59.5 | 56.6 | 2.9 (-0.5, 6.3) | 0.087 | 4/10 | +7 |
|  | 50 µg/L, 100 m, 50% of wells | 20 days | Static plan | 59.5 | 56.9 | 2.7 (-1.0, 6.3) | 0.138 | 4/10 | +8 |
|  | 50 µg/L, 100 m, 50% of wells | 20 days | Adaptive | 59.5 | 57.5 | 2.1 (-0.9, 5.0) | 0.144 | 4/10 | +6 |
| Census, 10 min per test | 50 µg/L, 100 m | primary | Door-to-door | 20.3 | 11.8 | 8.5 (3.3, 13.7) | 0.005 | 10/10 | +66 |
|  | 50 µg/L, 100 m | primary | Static plan | 20.3 | 14.8 | 5.5 (2.9, 8.0) | <0.001 | 9/10 | +43 |
|  | 50 µg/L, 100 m | primary | Adaptive | 20.3 | 17.6 | 2.7 (1.1, 4.3) | 0.004 | 9/10 | +17 |
|  | 50 µg/L, 100 m | 5 days | Door-to-door | 31.8 | 20.6 | 11.1 (4.2, 18.1) | 0.005 | 10/10 | +51 |
|  | 50 µg/L, 100 m | 5 days | Static plan | 31.8 | 23.7 | 8.1 (5.0, 11.2) | <0.001 | 10/10 | +39 |
|  | 50 µg/L, 100 m | 5 days | Adaptive | 31.8 | 28.2 | 3.6 (1.2, 5.9) | 0.007 | 9/10 | +14 |
|  | 50 µg/L, 100 m | 10 days | Door-to-door | 54.4 | 42.7 | 11.6 (4.9, 18.4) | 0.004 | 10/10 | +27 |
|  | 50 µg/L, 100 m | 10 days | Static plan | 54.4 | 45.1 | 9.3 (4.2, 14.4) | 0.003 | 9/10 | +26 |
|  | 50 µg/L, 100 m | 10 days | Adaptive | 54.4 | 48.8 | 5.6 (2.6, 8.6) | 0.002 | 10/10 | +13 |
|  | 50 µg/L, 100 m | 20 days | Door-to-door | 71.2 | 68.3 | 2.9 (-0.8, 6.6) | 0.109 | 4/10 | +7 |
|  | 50 µg/L, 100 m | 20 days | Static plan | 71.2 | 66.9 | 4.3 (-1.0, 9.5) | 0.099 | 4/10 | +11 |
|  | 50 µg/L, 100 m | 20 days | Adaptive | 71.2 | 68.8 | 2.4 (-0.9, 5.7) | 0.135 | 4/10 | +6 |
| Census, about 12 tests per day | 50 µg/L, 100 m | primary | Door-to-door | 7.2 | 3.8 | 3.4 (1.0, 5.8) | 0.010 | 9/10 | +79 |
|  | 50 µg/L, 100 m | primary | Static plan | 7.2 | 5.8 | 1.3 (-0.1, 2.8) | 0.070 | 8/10 | +28 |
|  | 50 µg/L, 100 m | primary | Adaptive | 7.2 | 6.4 | 0.8 (0.3, 1.3) | 0.005 | 9/10 | +14 |
|  | 50 µg/L, 100 m | 5 days | Door-to-door | 12.4 | 6.8 | 5.6 (1.8, 9.3) | 0.009 | 10/10 | +74 |
|  | 50 µg/L, 100 m | 5 days | Static plan | 12.4 | 9.6 | 2.8 (0.7, 4.9) | 0.016 | 9/10 | +32 |
|  | 50 µg/L, 100 m | 5 days | Adaptive | 12.4 | 10.9 | 1.4 (0.7, 2.2) | 0.002 | 10/10 | +14 |
|  | 50 µg/L, 100 m | 10 days | Door-to-door | 24.8 | 14.8 | 10.0 (4.0, 15.9) | 0.004 | 10/10 | +63 |
|  | 50 µg/L, 100 m | 10 days | Static plan | 24.8 | 18.2 | 6.6 (3.0, 10.2) | 0.003 | 10/10 | +39 |
|  | 50 µg/L, 100 m | 10 days | Adaptive | 24.8 | 21.6 | 3.1 (1.3, 5.0) | 0.004 | 9/10 | +17 |
|  | 50 µg/L, 100 m | 20 days | Door-to-door | 45.2 | 30.7 | 14.6 (7.4, 21.8) | 0.001 | 10/10 | +46 |
|  | 50 µg/L, 100 m | 20 days | Static plan | 45.2 | 34.3 | 10.9 (6.1, 15.7) | <0.001 | 10/10 | +35 |
|  | 50 µg/L, 100 m | 20 days | Adaptive | 45.2 | 40.3 | 4.9 (3.0, 6.8) | <0.001 | 9/10 | +15 |
| 2012-13, about 12 tests per day | 50 µg/L, 100 m | primary | Door-to-door | 11.2 | 4.9 | 6.3 (2.9, 9.7) | <0.001 | 38/52 | +46 |
|  | 50 µg/L, 100 m | primary | Static plan | 11.2 | 8.7 | 2.5 (1.1, 3.8) | <0.001 | 36/52 | +28 |
|  | 50 µg/L, 100 m | primary | Adaptive | 11.2 | 9.4 | 1.8 (0.6, 2.9) | 0.003 | 35/52 | +24 |
|  | 50 µg/L, 100 m | 5 days | Door-to-door | 17.0 | 8.2 | 8.7 (4.3, 13.2) | <0.001 | 41/52 | +39 |
|  | 50 µg/L, 100 m | 5 days | Static plan | 17.0 | 13.3 | 3.7 (1.8, 5.6) | <0.001 | 36/52 | +24 |
|  | 50 µg/L, 100 m | 5 days | Adaptive | 17.0 | 14.9 | 2.0 (0.7, 3.4) | 0.005 | 38/52 | +19 |
|  | 50 µg/L, 100 m | 10 days | Door-to-door | 29.8 | 15.5 | 14.3 (8.8, 19.9) | <0.001 | 48/52 | +46 |
|  | 50 µg/L, 100 m | 10 days | Static plan | 29.8 | 22.7 | 7.2 (4.7, 9.6) | <0.001 | 47/52 | +34 |
|  | 50 µg/L, 100 m | 10 days | Adaptive | 29.8 | 25.5 | 4.3 (2.6, 5.9) | <0.001 | 41/52 | +18 |
|  | 50 µg/L, 100 m | 20 days | Door-to-door | 44.6 | 30.3 | 14.3 (9.5, 19.1) | <0.001 | 46/52 | +44 |
|  | 50 µg/L, 100 m | 20 days | Static plan | 44.6 | 36.0 | 8.5 (6.4, 10.7) | <0.001 | 41/52 | +41 |
|  | 50 µg/L, 100 m | 20 days | Adaptive | 44.6 | 39.6 | 5.0 (3.1, 6.9) | <0.001 | 41/52 | +15 |
| 2012-13 with 2000-01 values (11 blocks) | 10 µg/L, 100 m | primary | Door-to-door | 11.3 | 7.2 | 4.1 (-1.6, 9.7) | 0.137 | 10/11 | +20 |
|  | 10 µg/L, 100 m | primary | Static plan | 11.3 | 9.2 | 2.1 (-0.1, 4.4) | 0.062 | 9/11 | +16 |
|  | 10 µg/L, 100 m | primary | Adaptive | 11.3 | 9.5 | 1.8 (-0.2, 3.7) | 0.074 | 10/11 | +17 |
|  | 50 µg/L, 100 m | primary | Door-to-door | 17.2 | 7.1 | 10.0 (-1.1, 21.2) | 0.073 | 11/11 | +56 |
|  | 50 µg/L, 100 m | primary | Static plan | 17.2 | 13.9 | 3.3 (1.3, 5.3) | 0.004 | 10/11 | +35 |
|  | 50 µg/L, 100 m | primary | Adaptive | 17.2 | 15.1 | 2.1 (0.7, 3.4) | 0.006 | 11/11 | +13 |
|  | 10 µg/L, 100 m | 5 days | Door-to-door | 11.7 | 6.9 | 4.8 (-3.0, 12.7) | 0.200 | 8/11 | +18 |
|  | 10 µg/L, 100 m | 5 days | Static plan | 11.7 | 10.3 | 1.4 (-0.8, 3.6) | 0.181 | 7/11 | +6 |
|  | 10 µg/L, 100 m | 5 days | Adaptive | 11.7 | 10.9 | 0.8 (0.1, 1.6) | 0.037 | 10/11 | +9 |
|  | 50 µg/L, 100 m | 5 days | Door-to-door | 17.0 | 6.8 | 10.2 (-3.2, 23.6) | 0.121 | 11/11 | +49 |
|  | 50 µg/L, 100 m | 5 days | Static plan | 17.0 | 14.5 | 2.5 (0.1, 4.8) | 0.041 | 9/11 | +19 |
|  | 50 µg/L, 100 m | 5 days | Adaptive | 17.0 | 16.1 | 0.9 (-0.2, 2.0) | 0.090 | 7/11 | +6 |
|  | 10 µg/L, 100 m | 10 days | Door-to-door | 23.1 | 13.2 | 9.8 (-2.1, 21.8) | 0.095 | 10/11 | +32 |
|  | 10 µg/L, 100 m | 10 days | Static plan | 23.1 | 16.7 | 6.3 (1.3, 11.3) | 0.018 | 10/11 | +31 |
|  | 10 µg/L, 100 m | 10 days | Adaptive | 23.1 | 18.6 | 4.5 (0.2, 8.8) | 0.041 | 10/11 | +20 |
|  | 50 µg/L, 100 m | 10 days | Door-to-door | 28.7 | 13.4 | 15.3 (-0.1, 30.7) | 0.052 | 11/11 | +56 |
|  | 50 µg/L, 100 m | 10 days | Static plan | 28.7 | 23.0 | 5.7 (2.3, 9.1) | 0.004 | 10/11 | +31 |
|  | 50 µg/L, 100 m | 10 days | Adaptive | 28.7 | 25.6 | 3.1 (0.1, 6.1) | 0.047 | 9/11 | +9 |
|  | 10 µg/L, 100 m | 20 days | Door-to-door | 39.9 | 26.5 | 13.4 (4.2, 22.6) | 0.009 | 11/11 | +35 |
|  | 10 µg/L, 100 m | 20 days | Static plan | 39.9 | 28.6 | 11.3 (6.2, 16.3) | <0.001 | 11/11 | +39 |
|  | 10 µg/L, 100 m | 20 days | Adaptive | 39.9 | 33.3 | 6.7 (1.4, 11.9) | 0.018 | 10/11 | +17 |
|  | 50 µg/L, 100 m | 20 days | Door-to-door | 48.0 | 27.7 | 20.3 (10.4, 30.3) | 0.001 | 11/11 | +56 |
|  | 50 µg/L, 100 m | 20 days | Static plan | 48.0 | 36.3 | 11.7 (8.4, 15.0) | <0.001 | 11/11 | +40 |
|  | 50 µg/L, 100 m | 20 days | Adaptive | 48.0 | 41.9 | 6.2 (2.2, 10.2) | 0.006 | 10/11 | +13 |
| Census, capacity 5 households | 50 µg/L, 100 m | 10 days | Door-to-door | 24.7 | 18.9 | 5.8 (1.7, 9.8) | 0.010 | 8/10 | +26 |
|  | 50 µg/L, 100 m | 10 days | Static plan | 24.7 | 19.7 | 5.0 (1.9, 8.2) | 0.006 | 10/10 | +25 |
|  | 50 µg/L, 100 m | 10 days | Adaptive | 24.7 | 21.5 | 3.2 (0.2, 6.2) | 0.037 | 8/10 | +12 |
|  | 10 µg/L, 100 m | 10 days | Door-to-door | 16.4 | 11.5 | 4.8 (1.9, 7.7) | 0.005 | 10/10 | +37 |
|  | 10 µg/L, 100 m | 10 days | Static plan | 16.4 | 13.3 | 3.1 (0.0, 6.2) | 0.050 | 7/10 | +19 |
|  | 10 µg/L, 100 m | 10 days | Adaptive | 16.4 | 15.0 | 1.4 (-0.5, 3.3) | 0.135 | 8/10 | +6 |
| Census, capacity 5, optimal assignment | 50 µg/L, 100 m | 10 days | Door-to-door | 26.7 | 19.9 | 6.8 (2.0, 11.6) | 0.010 | 8/10 | +29 |
|  | 50 µg/L, 100 m | 10 days | Static plan | 26.7 | 20.7 | 6.0 (2.5, 9.5) | 0.003 | 10/10 | +29 |
|  | 50 µg/L, 100 m | 10 days | Adaptive | 26.7 | 22.6 | 4.1 (0.5, 7.7) | 0.031 | 9/10 | +15 |
|  | 10 µg/L, 100 m | 10 days | Door-to-door | 17.9 | 12.2 | 5.6 (2.0, 9.2) | 0.006 | 10/10 | +40 |
|  | 10 µg/L, 100 m | 10 days | Static plan | 17.9 | 14.3 | 3.6 (0.2, 7.0) | 0.042 | 7/10 | +20 |
|  | 10 µg/L, 100 m | 10 days | Adaptive | 17.9 | 16.2 | 1.6 (-0.6, 3.9) | 0.128 | 8/10 | +7 |
| Census, expected switching | 50 µg/L, 100 m | 10 days | Door-to-door | 19.7 | 14.1 | 5.6 (2.5, 8.7) | 0.003 | 10/10 | +38 |
|  | 50 µg/L, 100 m | 10 days | Static plan | 19.7 | 14.5 | 5.2 (2.5, 7.9) | 0.002 | 10/10 | +37 |
|  | 50 µg/L, 100 m | 10 days | Adaptive | 19.7 | 16.6 | 3.1 (1.3, 4.8) | 0.003 | 10/10 | +19 |
|  | 10 µg/L, 100 m | 10 days | Door-to-door | 14.7 | 9.7 | 5.0 (2.9, 7.1) | <0.001 | 10/10 | +49 |
|  | 10 µg/L, 100 m | 10 days | Static plan | 14.7 | 11.9 | 2.8 (1.0, 4.7) | 0.007 | 9/10 | +22 |
|  | 10 µg/L, 100 m | 10 days | Adaptive | 14.7 | 13.8 | 0.9 (-0.1, 1.9) | 0.064 | 9/10 | +6 |
| 2012-13, capacity 5 households | 50 µg/L, 100 m | 10 days | Door-to-door | 30.3 | 19.5 | 10.8 (5.2, 16.3) | <0.001 | 34/52 | +12 |
|  | 50 µg/L, 100 m | 10 days | Static plan | 30.3 | 24.0 | 6.3 (3.8, 8.9) | <0.001 | 43/52 | +20 |
|  | 50 µg/L, 100 m | 10 days | Adaptive | 30.3 | 26.5 | 3.8 (2.0, 5.7) | <0.001 | 34/52 | +9 |
|  | 10 µg/L, 100 m | 10 days | Door-to-door | 21.7 | 16.1 | 5.6 (2.4, 8.9) | <0.001 | 30/52 | +6 |
|  | 10 µg/L, 100 m | 10 days | Static plan | 21.7 | 17.4 | 4.3 (2.6, 6.1) | <0.001 | 39/52 | +19 |
|  | 10 µg/L, 100 m | 10 days | Adaptive | 21.7 | 18.1 | 3.7 (1.9, 5.4) | <0.001 | 36/52 | +16 |
| 2012-13, capacity 5, optimal assignment | 50 µg/L, 100 m | 10 days | Door-to-door | 31.4 | 20.1 | 11.3 (5.6, 17.0) | <0.001 | 34/52 | +14 |
|  | 50 µg/L, 100 m | 10 days | Static plan | 31.4 | 24.9 | 6.5 (4.0, 9.0) | <0.001 | 44/52 | +22 |
|  | 50 µg/L, 100 m | 10 days | Adaptive | 31.4 | 27.3 | 4.1 (2.2, 5.9) | <0.001 | 38/52 | +11 |
|  | 10 µg/L, 100 m | 10 days | Door-to-door | 22.5 | 16.8 | 5.7 (2.4, 9.1) | 0.001 | 30/52 | +6 |
|  | 10 µg/L, 100 m | 10 days | Static plan | 22.5 | 18.1 | 4.5 (2.7, 6.2) | <0.001 | 38/52 | +19 |
|  | 10 µg/L, 100 m | 10 days | Adaptive | 22.5 | 18.8 | 3.8 (2.0, 5.5) | <0.001 | 37/52 | +17 |
| 2012-13, expected switching | 50 µg/L, 100 m | 10 days | Door-to-door | 23.9 | 15.6 | 8.3 (4.6, 12.0) | <0.001 | 39/52 | +22 |
|  | 50 µg/L, 100 m | 10 days | Static plan | 23.9 | 18.5 | 5.4 (3.6, 7.1) | <0.001 | 47/52 | +33 |
|  | 50 µg/L, 100 m | 10 days | Adaptive | 23.9 | 20.3 | 3.6 (2.3, 4.9) | <0.001 | 47/52 | +19 |
|  | 10 µg/L, 100 m | 10 days | Door-to-door | 18.1 | 13.2 | 4.9 (2.7, 7.0) | <0.001 | 41/52 | +19 |
|  | 10 µg/L, 100 m | 10 days | Static plan | 18.1 | 14.2 | 4.0 (2.6, 5.3) | <0.001 | 43/52 | +27 |
|  | 10 µg/L, 100 m | 10 days | Adaptive | 18.1 | 14.9 | 3.2 (2.0, 4.4) | <0.001 | 44/52 | +23 |

<!-- spec: L{1.6cm}L{2.4cm}R{1.6cm}R{1.6cm}L{2.4cm}R{1.0cm}R{1.0cm} -->
**Table A2.** The Gaussian-process learner compared with the other learners (2000-01 census, 100 m, kit error, primary effort; separate run with 2 random splits and 4 runs per block).

| Threshold (µg/L) | Learner compared | Gaussian process (%) | Other learner (%) | Difference (95% CI) | *p* | Blocks |
|---|---|---|---|---|---|---|
| 50 | Random forest | 10.5 | 10.2 | 0.3 (-0.7, 1.4) | 0.505 | 5/10 |
|  | LightGBM | 10.5 | 10.3 | 0.2 (-0.4, 0.9) | 0.437 | 6/10 |
|  | kNN | 10.5 | 10.2 | 0.3 (-0.1, 0.7) | 0.148 | 7/10 |
| 10 | Random forest | 8.4 | 7.9 | 0.5 (-0.5, 1.5) | 0.279 | 6/10 |
|  | LightGBM | 8.4 | 7.8 | 0.6 (-0.3, 1.6) | 0.177 | 6/10 |
|  | kNN | 8.4 | 8.4 | 0.0 (-0.2, 0.3) | 0.689 | 5/10 |

<!-- spec: L{2.6cm}R{1.2cm}L{1.6cm}L{1.8cm}R{1.2cm}R{1.4cm}L{2.4cm}R{1.0cm}R{1.0cm} -->
**Table A3.** Exposed households protected after ten team-days, by tertile (within each block) of the local share of safe wells within 200 m; learning compared with door-to-door testing and the static plan.

| Survey | Threshold (µg/L) | Tertile | Comparator | Learning (%) | Comparator (%) | Difference (95% CI) | *p* | Blocks |
|---|---|---|---|---|---|---|---|---|
| 2000-01 census | 50 | fewest safe | Door-to-door | 32.9 | 16.3 | 16.5 (8.4, 24.7) | 0.001 | 10/10 |
|  | 50 | fewest safe | Static plan | 32.9 | 16.0 | 16.9 (9.4, 24.3) | <0.001 | 10/10 |
|  | 50 | middle | Door-to-door | 33.7 | 27.5 | 6.2 (-3.2, 15.6) | 0.172 | 5/10 |
|  | 50 | middle | Static plan | 33.7 | 30.6 | 3.1 (-4.8, 11.0) | 0.396 | 5/10 |
|  | 50 | most safe | Door-to-door | 18.6 | 27.0 | -8.4 (-19.9, 3.1) | 0.133 | 3/10 |
|  | 50 | most safe | Static plan | 18.6 | 28.0 | -9.3 (-17.9, -0.8) | 0.036 | 2/10 |
|  | 10 | fewest safe | Door-to-door | 7.6 | 5.8 | 1.8 (-2.3, 5.9) | 0.342 | 6/10 |
|  | 10 | fewest safe | Static plan | 7.6 | 4.4 | 3.1 (0.3, 6.0) | 0.034 | 7/10 |
|  | 10 | middle | Door-to-door | 32.0 | 20.8 | 11.1 (2.1, 20.2) | 0.021 | 10/10 |
|  | 10 | middle | Static plan | 32.0 | 22.6 | 9.4 (1.4, 17.3) | 0.026 | 8/10 |
|  | 10 | most safe | Door-to-door | 38.2 | 25.2 | 13.0 (2.0, 24.0) | 0.026 | 8/10 |
|  | 10 | most safe | Static plan | 38.2 | 38.1 | 0.1 (-10.5, 10.6) | 0.989 | 5/10 |
| 2012-13 upazila survey | 50 | fewest safe | Door-to-door | 40.0 | 18.3 | 21.8 (16.0, 27.5) | <0.001 | 48/52 |
|  | 50 | fewest safe | Static plan | 40.0 | 27.3 | 12.7 (9.2, 16.2) | <0.001 | 46/52 |
|  | 50 | middle | Door-to-door | 30.9 | 24.5 | 6.3 (1.2, 11.5) | 0.017 | 31/52 |
|  | 50 | middle | Static plan | 30.9 | 29.2 | 1.7 (-2.2, 5.5) | 0.386 | 23/52 |
|  | 50 | most safe | Door-to-door | 13.7 | 21.5 | -7.9 (-11.1, -4.6) | <0.001 | 5/52 |
|  | 50 | most safe | Static plan | 13.7 | 19.5 | -5.8 (-10.6, -1.1) | 0.017 | 11/52 |
|  | 10 | fewest safe | Door-to-door | 27.5 | 14.3 | 13.3 (8.4, 18.1) | <0.001 | 45/52 |
|  | 10 | fewest safe | Static plan | 27.5 | 17.7 | 9.9 (6.9, 12.8) | <0.001 | 47/52 |
|  | 10 | middle | Door-to-door | 28.3 | 23.5 | 4.8 (2.0, 7.7) | 0.001 | 33/52 |
|  | 10 | middle | Static plan | 28.3 | 25.4 | 2.9 (-0.3, 6.1) | 0.075 | 29/52 |
|  | 10 | most safe | Door-to-door | 21.0 | 21.0 | -0.0 (-3.6, 3.6) | 0.998 | 20/52 |
|  | 10 | most safe | Static plan | 21.0 | 21.3 | -0.3 (-3.8, 3.1) | 0.840 | 22/52 |

<!-- spec: L{1.6cm}L{8.0cm}L{3.8cm}L{3.6cm} -->
**Table A4.** Design and analysis history. Earlier versions were evaluated on held-out blocks before the present design was fixed; the outcome, score and comparators changed between versions as shown. The primary endpoint of v6 was declared before its final runs. Note: the Scopus search cited in the Introduction used TITLE-ABS-KEY(arsenic AND (well∗ OR tubewell∗) AND (test∗ OR screening OR sampling)), 1995-2026, retrieved 24 November 2026 (5,348 records, 5,212 with reference lists).

| Version | Design | Evaluated on | Status |
|---|---|---|---|
| v1 (pilot) | Outcome 'access' (household with a tested-safe well within 100 m); test counts, no travel; random forest learner | Development blocks only | Replaced: the outcome counted households whose own well was already safe |
| v2 | Access outcome; 10 min per test, then amended (before viewing test results) to 20 min and a 10-day horizon; random-forest headline; uncertainty sampling | Five test blocks (first test-block evaluation) | Seeding error found by internal review; re-run as v3 |
| v3 | As v2 with shared random numbers; oracle, risk-map plan, capacity and distance-decay analyses | Five test blocks (second look) | Replaced: access outcome judged inappropriate for public health |
| v4 | Cross-fitting run of v3 started on the other half | Stopped before completion | Superseded by the v5 outcome change |
| v5 | Protection outcome; look-ahead score (weight 0.1 for all policies); Gaussian-process headline (chosen on development blocks); uncertainty sampling and risk-map plan dropped; 10 runs, 20 days | 10 census blocks (cross-fitted); 2012-13 upazila survey (first look, kit readings as truth) | Replaced after a second internal review |
| v6 (this paper) | Per-policy look-ahead weight; uncertainty sampling restored; repeated random splits; primary endpoint (5 team-days per 1,000 wells, Holm) declared before final runs; 2012-13 truth imputed from kit-laboratory pairs; 100 m link check; field realism, capacity and switching outcomes; non-overlap set defined as ≥20 wells within 100 m of a census well | 10 census blocks (5 splits); 52 upazila blocks (2 splits; second look) | Reported |

<!-- spec: L{1.8cm}L{1.6cm}R{0.8cm}R{1.1cm}L{2.4cm}L{1.5cm}L{2.4cm}L{2.4cm}L{1.4cm} -->
**Table A5.** Further tests of the six primary comparisons (learning against each comparator; 5 team-days per 1,000 wells, 50 µg/L, 100 m), in percentage points. Permutation: paired sign-flip test, exact for the 10 census blocks, 200,000 random patterns otherwise; Holm adjustment over the six comparisons for each test. Super-blocks: 17 upazila cells of 4 km × 4 km, permutation *p*. Confirmatory: the five census blocks never used for a design choice, exact permutation *p* (smallest attainable 0.0625). MC SE: Monte Carlo standard error of a block's mean difference over its runs, against the standard deviation of the block means.

| Survey | Comparator | Blocks | Median difference | Holm *p*: *t*; permutation; Wilcoxon | Leave one block out | Super-blocks (95% CI); *p* | Confirmatory (95% CI); *p* | MC SE / SD between blocks |
|---|---|---|---|---|---|---|---|---|
| 2000-01 census | Door-to-door | 10 | 3.3 | 0.011; 0.006; 0.006 | 3.3 to 4.4 | - | 4.5 (-0.1, 9.0); 0.0625 | 0.77 / 3.51 |
|  | Static plan | 10 | 2.2 | 0.011; 0.008; 0.010 | 2.3 to 3.3 | - | 4.2 (1.0, 7.5); 0.0625 | 0.60 / 2.67 |
|  | Adaptive | 10 | 1.4 | <0.001; 0.006; 0.006 | 1.2 to 1.4 | - | 1.1 (0.3, 1.9); 0.0625 | 0.41 / 0.67 |
| 2012-13 upazila survey | Door-to-door | 52 | 3.0 | <0.001; <0.001; <0.001 | 7.4 to 8.6 | 7.8 (2.0, 13.6); 0.002 | - | 1.86 / 14.38 |
|  | Static plan | 52 | 1.9 | 0.001; <0.001; <0.001 | 2.9 to 3.7 | 3.5 (0.9, 6.1); <0.001 | - | 1.27 / 6.37 |
|  | Adaptive | 52 | 1.0 | 0.008; <0.001; <0.001 | 1.6 to 2.4 | 2.2 (0.6, 3.9); <0.001 | - | 1.11 / 4.98 |

<!-- spec: L{3.2cm}L{1.9cm}R{1.5cm}R{1.6cm}R{1.6cm}R{1.4cm}R{2.0cm}R{2.0cm} -->
**Table A6.** Estimated field cost per person protected in the simulation at the primary effort (5 team-days per 1,000 wells, 50 µg/L, 100 m; pooled over blocks), under the three costings of Appendix A with 8 persons per well, and under the time-based costing with 5 or 10 persons per well. This is a field cost per simulated person protected, not a programme cost: transport to the area, training, smartphones, software, household communication, programme management and monitoring are excluded.

| Survey | Policy | Tests per team-day | Time-based (US$) | Time plus kit (US$) | Per test (US$) | Time-based, 5 persons per well (US$) | Time-based, 10 persons per well (US$) |
|---|---|---|---|---|---|---|---|
| 2000-01 census | Door-to-door | 17.2 | 0.56 | 0.57 | 0.80 | 0.89 | 0.45 |
|  | Static plan | 16.6 | 0.49 | 0.50 | 0.68 | 0.78 | 0.39 |
|  | Adaptive | 16.0 | 0.41 | 0.41 | 0.54 | 0.65 | 0.32 |
|  | Learning | 16.2 | 0.35 | 0.35 | 0.47 | 0.55 | 0.28 |
| 2012-13 upazila survey | Door-to-door | 17.4 | 0.62 | 0.65 | 0.91 | 1.00 | 0.50 |
|  | Static plan | 16.6 | 0.55 | 0.56 | 0.75 | 0.87 | 0.44 |
|  | Adaptive | 15.9 | 0.49 | 0.50 | 0.65 | 0.79 | 0.39 |
|  | Learning | 15.7 | 0.41 | 0.42 | 0.53 | 0.65 | 0.33 |
| Census, field realism | Door-to-door | 14.1 | 0.66 | 0.67 | 0.78 | 1.06 | 0.53 |
|  | Static plan | 13.1 | 0.58 | 0.58 | 0.63 | 0.93 | 0.46 |
|  | Adaptive | 12.7 | 0.53 | 0.53 | 0.56 | 0.85 | 0.42 |
|  | Learning | 12.5 | 0.48 | 0.49 | 0.50 | 0.77 | 0.39 |
| Census, about 12 tests per day | Door-to-door | 12.0 | 0.89 | 0.89 | 0.89 | 1.42 | 0.71 |
|  | Static plan | 11.6 | 0.64 | 0.63 | 0.61 | 1.02 | 0.51 |
|  | Adaptive | 11.1 | 0.57 | 0.56 | 0.52 | 0.91 | 0.45 |
|  | Learning | 11.3 | 0.50 | 0.49 | 0.47 | 0.79 | 0.40 |
| 2012-13, about 12 tests per day | Door-to-door | 12.1 | 0.92 | 0.92 | 0.93 | 1.47 | 0.74 |
|  | Static plan | 11.5 | 0.81 | 0.81 | 0.78 | 1.30 | 0.65 |
|  | Adaptive | 11.0 | 0.78 | 0.78 | 0.72 | 1.25 | 0.63 |
|  | Learning | 10.9 | 0.63 | 0.63 | 0.57 | 1.01 | 0.50 |

<!-- spec: L{2.0cm}L{2.4cm}R{1.2cm}L{1.6cm}R{1.4cm}R{1.2cm}R{1.3cm}L{2.2cm}R{1.0cm}R{0.9cm} -->
**Table A7.** Sensitivity of the primary comparisons (5 team-days per 1,000 wells, 100 m) to measurement assumptions of the Gaussian process and to the spatial separation of fitting and evaluated wells: kit readings of 0 set to 5 µg/L (half the lowest non-zero category) instead of 1 µg/L, in both surveys; and the kit-error variance of the 944 kit-laboratory pairs (0.113 in squared $\log_{10}$ units) added to the census nugget (Appendix A); and a 600 m exclusion buffer, removing fitting wells within 600 m of any evaluated well (both surveys, 50 µg/L). Same design as the main analysis; paired *t*-test over blocks, unadjusted.

| Survey | Analysis | Threshold (µg/L) | Comparator | Learning, main analysis (%) | Learning (%) | Comparator (%) | Difference (95% CI) | *p* | Blocks |
|---|---|---|---|---|---|---|---|---|---|
| 2000-01 census | kit reading 0 set to 5 µg/L | 50 | Door-to-door | 10.3 | 10.3 | 6.3 | 4.0 (1.5, 6.6) | 0.006 | 10/10 |
|  |  | 50 | Static plan | 10.3 | 10.3 | 7.4 | 2.9 (1.0, 4.7) | 0.007 | 9/10 |
|  |  | 50 | Adaptive | 10.3 | 10.3 | 9.0 | 1.4 (0.8, 1.9) | <0.001 | 10/10 |
|  |  | 10 | Door-to-door | 8.2 | 8.1 | 4.4 | 3.7 (1.9, 5.4) | <0.001 | 10/10 |
|  |  | 10 | Static plan | 8.2 | 8.1 | 5.9 | 2.2 (0.7, 3.6) | 0.008 | 9/10 |
|  |  | 10 | Adaptive | 8.2 | 8.1 | 7.6 | 0.4 (-0.1, 1.0) | 0.102 | 8/10 |
| 2012-13 upazila survey | kit reading 0 set to 5 µg/L | 50 | Door-to-door | 15.6 | 15.7 | 7.2 | 8.5 (4.5, 12.4) | <0.001 | 44/52 |
|  |  | 50 | Static plan | 15.6 | 15.7 | 12.0 | 3.7 (2.2, 5.1) | <0.001 | 41/52 |
|  |  | 50 | Adaptive | 15.6 | 15.7 | 13.2 | 2.5 (1.3, 3.7) | <0.001 | 42/52 |
|  |  | 10 | Door-to-door | 10.0 | 10.1 | 5.9 | 4.3 (2.7, 5.8) | <0.001 | 48/52 |
|  |  | 10 | Static plan | 10.0 | 10.1 | 7.0 | 3.1 (1.9, 4.3) | <0.001 | 45/52 |
|  |  | 10 | Adaptive | 10.0 | 10.1 | 8.4 | 1.8 (0.9, 2.6) | <0.001 | 42/52 |
| 2000-01 census | kit-error variance added to nugget | 50 | Door-to-door | 10.3 | 10.2 | 6.3 | 4.0 (1.4, 6.5) | 0.006 | 10/10 |
|  |  | 50 | Static plan | 10.3 | 10.2 | 7.4 | 2.8 (0.9, 4.7) | 0.009 | 9/10 |
|  |  | 50 | Adaptive | 10.3 | 10.2 | 9.0 | 1.3 (0.9, 1.7) | <0.001 | 10/10 |
|  |  | 10 | Door-to-door | 8.2 | 8.2 | 4.4 | 3.8 (2.1, 5.5) | <0.001 | 10/10 |
|  |  | 10 | Static plan | 8.2 | 8.2 | 5.9 | 2.3 (0.8, 3.8) | 0.006 | 9/10 |
|  |  | 10 | Adaptive | 8.2 | 8.2 | 7.6 | 0.6 (-0.0, 1.2) | 0.053 | 8/10 |
|  | 600 m exclusion buffer around evaluated wells | 50 | Door-to-door | 10.3 | 10.4 | 6.3 | 4.1 (1.7, 6.5) | 0.004 | 10/10 |
|  |  | 50 | Static plan | 10.3 | 10.4 | 7.4 | 3.0 (1.3, 4.7) | 0.003 | 9/10 |
|  |  | 50 | Adaptive | 10.3 | 10.4 | 8.7 | 1.7 (0.8, 2.6) | 0.002 | 10/10 |
| 2012-13 upazila survey | 600 m exclusion buffer around evaluated wells | 50 | Door-to-door | 15.6 | 15.5 | 7.2 | 8.3 (4.3, 12.2) | <0.001 | 44/52 |
|  |  | 50 | Static plan | 15.6 | 15.5 | 11.8 | 3.6 (2.0, 5.2) | <0.001 | 39/52 |
|  |  | 50 | Adaptive | 15.6 | 15.5 | 13.4 | 2.0 (0.7, 3.4) | 0.003 | 41/52 |

<!-- spec: L{5.0cm}R{1.6cm}R{1.4cm}R{2.6cm}R{2.0cm} -->
**Table A8.** Uncertainty of the imputed laboratory truth in the 2012-13 upazila survey. The 944 kit-laboratory pairs were resampled with replacement; the share of the 46,976 wells truly safe is from 200 bootstrap samples, and the policy comparisons (5 team-days per 1,000 wells, 50 µg/L, 100 m; mean over 52 blocks) are from 20 bootstrap replicates of the campaigns, each with its own resampled pairs shared by all blocks and policies.

| Quantity | Replicates | Mean | 2.5-97.5 percentile | Replicates positive |
|---|---|---|---|---|
| truly safe at 50 µg/L (%) | 200 | 60.4 | 59.1 to 61.8 | 200 |
| truly safe at 10 µg/L (%) | 200 | 42.0 | 40.3 to 43.3 | 200 |
| learning protected (%) | 20 | 15.6 | 14.7 to 16.8 | 20 |
| gain vs door-to-door (pp) | 20 | 8.5 | 7.1 to 9.7 | 20 |
| relative gain vs door-to-door (%) | 20 | 53.6 | 36.0 to 70.8 | 20 |
| gain vs static plan (pp) | 20 | 3.9 | 2.9 to 5.0 | 20 |
| relative gain vs static plan (%) | 20 | 34.0 | 26.8 to 45.4 | 20 |
| gain vs adaptive (pp) | 20 | 2.4 | 1.4 to 3.6 | 20 |
| relative gain vs adaptive (%) | 20 | 19.2 | 15.2 to 23.9 | 20 |

<!-- spec: L{2.2cm}R{1.2cm}R{1.2cm}L{1.2cm}R{1.4cm}R{1.2cm}R{1.2cm}R{1.2cm}R{1.3cm}R{1.3cm} -->
**Table A9.** Held-out calibration experiment for the predicted probability that a well is safe (split 0, both directions of cross-fitting, 3 repetitions). The tested wells were chosen at random, not by an adaptive policy: a random 5% or 20% of the wells of each block were treated as tested, and the Gaussian process (GP) fitted on the other half, or the depth prior alone, predicted the remaining wells; lower Brier score and log loss are better. Reliability diagrams are in Figure A2.

| Survey | Threshold (µg/L) | Wells tested (%) | Layer | Predictions | Truly safe (%) | Brier, GP | Brier, prior | Log loss, GP | Log loss, prior |
|---|---|---|---|---|---|---|---|---|---|
| 2000-01 census | 10 | 5 | all | 18810 | 26.4 | 0.152 | 0.164 | 0.513 | 0.515 |
|  |  |  | shallow | 15880 | 18.4 | 0.139 | 0.148 | 0.489 | 0.480 |
|  |  |  | deep | 2930 | 69.8 | 0.225 | 0.254 | 0.641 | 0.701 |
|  |  | 20 | all | 15837 | 26.4 | 0.135 | 0.165 | 0.462 | 0.517 |
|  |  |  | shallow | 13374 | 18.4 | 0.123 | 0.148 | 0.438 | 0.483 |
|  |  |  | deep | 2463 | 70.0 | 0.204 | 0.255 | 0.594 | 0.703 |
|  | 50 | 5 | all | 18810 | 47.5 | 0.187 | 0.218 | 0.559 | 0.626 |
|  |  |  | shallow | 15877 | 41.1 | 0.197 | 0.230 | 0.585 | 0.652 |
|  |  |  | deep | 2933 | 81.9 | 0.132 | 0.150 | 0.418 | 0.483 |
|  |  | 20 | all | 15837 | 47.9 | 0.167 | 0.217 | 0.505 | 0.625 |
|  |  |  | shallow | 13336 | 41.5 | 0.176 | 0.230 | 0.528 | 0.652 |
|  |  |  | deep | 2501 | 81.7 | 0.118 | 0.152 | 0.382 | 0.486 |
| 2012-13 upazila survey | 10 | 5 | all | 133959 | 42.2 | 0.148 | 0.190 | 0.463 | 0.571 |
|  |  |  | shallow | 98993 | 30.3 | 0.151 | 0.194 | 0.470 | 0.581 |
|  |  |  | deep | 34966 | 76.0 | 0.140 | 0.180 | 0.444 | 0.544 |
|  |  | 20 | all | 112806 | 42.2 | 0.131 | 0.190 | 0.419 | 0.570 |
|  |  |  | shallow | 83357 | 30.3 | 0.133 | 0.193 | 0.423 | 0.580 |
|  |  |  | deep | 29449 | 76.0 | 0.126 | 0.180 | 0.409 | 0.543 |
|  | 50 | 5 | all | 133959 | 60.5 | 0.166 | 0.199 | 0.501 | 0.583 |
|  |  |  | shallow | 98967 | 50.7 | 0.194 | 0.234 | 0.573 | 0.661 |
|  |  |  | deep | 34992 | 88.2 | 0.085 | 0.102 | 0.295 | 0.362 |
|  |  | 20 | all | 112806 | 60.5 | 0.149 | 0.199 | 0.457 | 0.582 |
|  |  |  | shallow | 83316 | 50.7 | 0.175 | 0.233 | 0.525 | 0.660 |
|  |  |  | deep | 29490 | 88.0 | 0.077 | 0.103 | 0.266 | 0.364 |

![**Figure A2.** Reliability of the Gaussian-process probability that a well is safe on held-out blocks: observed share of truly safe wells against the mean predicted probability, by decile of prediction (deciles with at least 30 wells), for 5% and 20% of wells tested and both thresholds.](figures/figA2_calibration.png)

<!-- spec: R{2.2cm}R{1.4cm}R{2.8cm}R{2.6cm}R{2.6cm} -->
**Table A10.** Laboratory arsenic by field-kit category in the 944 kit-laboratory pairs used for the kit-error model (census) and for imputing the laboratory truth (upazila survey). Readings of 999 and 1000 µg/L are the top category, coded both ways in the source data.

| Kit reading (µg/L) | Pairs | Median laboratory value (µg/L) | Laboratory ≤10 µg/L (%) | Laboratory ≤50 µg/L (%) |
|---|---|---|---|---|
| 0 | 144 | 0.5 | 95.8 | 100.0 |
| 10 | 62 | 6.7 | 75.8 | 100.0 |
| 25 | 108 | 17.6 | 14.8 | 97.2 |
| 50 | 135 | 33.5 | 3.0 | 80.7 |
| 100 | 196 | 59.6 | 0.5 | 32.7 |
| 200 | 118 | 108.3 | 0.8 | 5.1 |
| 300 | 74 | 172.4 | 0.0 | 1.4 |
| 500 | 71 | 299.6 | 0.0 | 0.0 |
| 999 | 30 | 375.9 | 0.0 | 10.0 |
| 1000 | 6 | 541.8 | 0.0 | 0.0 |
