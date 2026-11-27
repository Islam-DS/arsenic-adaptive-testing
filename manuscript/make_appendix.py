"""Write manuscript/appendix.md (Appendix A: methods details; Appendix B: Figure A1 and Tables A1-A7) from the results.

    python -X utf8 manuscript/make_appendix.py

The appendix is part of the main manuscript (no separate supplementary file). Table rows come from results/*.csv.
A line '<!-- spec: ... -->' before a table gives the LaTeX column specification; tables longer than 40 rows are set
as longtable by build_latex.py.
"""
import pathlib
import pandas as pd

M = pathlib.Path(__file__).resolve().parent
R = M.parent / "results"

SHORT = {"Door-to-door sweep": "Door-to-door", "Static depth plan": "Static plan", "Re-planning only": "Re-planning",
         "Uncertainty sampling": "Uncertainty", "Adaptive (depth + results)": "Adaptive",
         "Oracle (perfect knowledge)": "Oracle", "Learning: random forest": "Random forest",
         "Learning: LightGBM": "LightGBM", "Learning: kNN": "kNN"}
ANALYSIS = {"2000-01 census": "2000-01 census", "2012-13 upazila survey": "2012-13 upazila survey",
            "2012-13, blocks outside the census area": "2012-13, 38 blocks outside census area",
            "field realism (daily return, 15% absent)": "Census, field realism",
            "advise if kit ≤ ½ threshold": "Census, safe if kit ≤ half threshold", "radius": "Census, other radius",
            "no kit error": "Census, no kit error", "thinned": "Census, thinned wells", "10 min per test": "Census, 10 min per test",
            "2012-13 with 2000-01 survey": "2012-13 with 2000-01 values (11 blocks)",
            "2000-01 census; outcome prot_cap_share": "Census, capacity 5 households",
            "2000-01 census; outcome switch_share": "Census, expected switching",
            "2012-13 upazila survey; outcome prot_cap_share": "2012-13, capacity 5 households",
            "2012-13 upazila survey; outcome switch_share": "2012-13, expected switching"}


def p(v):
    if pd.isna(v):
        return "-"                               # no variation between blocks (all wells tested)
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def ci(d, lo, hi):
    return f"{d:.1f} ({lo:.1f}, {hi:.1f})"


def table(caption, spec, df, blank_repeats=()):
    df = df.copy()
    df = df.astype(str)
    orig = df.copy()
    for k, c in enumerate(blank_repeats):        # show a grouping value only when it (or an outer group) changes
        cols = list(blank_repeats[:k + 1])
        prev = None
        for i in df.index:
            cur = tuple(orig.loc[i, cols])
            if cur == prev:
                df.at[i, c] = ""
            prev = cur
    lines = [f"<!-- spec: {spec} -->", caption, "", "| " + " | ".join(df.columns) + " |",
             "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


TEXT_A = r"""# Appendix A. Details of the Score, the Gaussian Process, the Costings and the Inference

## Score

The score $g_j$ of Equation (1) is computed for every untested well *j*. $N(j)$ is the set of wells within the radius *r* of *j*, including *j*. For each household *i* (one per well), $\mathrm{inf}_i = 1$ if its well has been tested and read unsafe, $\mathrm{unt}_i = 1$ if its well is untested, and $c_i$ is the probability that a tested well within *r* of *i* has read safe. For the adaptive, re-planning and learning policies, $c_i = 1$ as soon as such a well exists and 0 otherwise. The static plan never looks at results: it treats a tested well *k* as informing its household with probability $1 - p_k$ and sets $c_i = 1 - \prod_k (1 - p_k)$ over the tested wells *k* within *r* of *i*. The terms of Equation (1) are $E_j = \sum_{i \in N(j)} \mathrm{inf}_i (1 - c_i)$, $S_j = c_j$, $U_j = \sum_{i \in N(j), i \neq j} \mathrm{unt}_i (1 - c_i)(1 - p_i)$ and $Q_j = 1 - \prod_{i \in N(j), i \neq j, \mathrm{unt}_i = 1} (1 - p_i)$. The expected gain is divided by the test time (20 min) plus the walking time from the team's position (straight-line distance × 1.3 at 3 km/h). Probabilities are clipped to $[10^{-9}, 1 - 10^{-9}]$ and ties are broken at random. Door-to-door testing goes to the nearest untested well. Uncertainty sampling maximises $p_j (1 - p_j)$ times the number of untested wells within *r*, per minute. The oracle uses $p_j = 1$ for truly safe wells and 0 otherwise. Outcomes are evaluated per household, so a household with several safe wells nearby is counted once.

## Depth Prior, Gaussian Process and Kit Readings

Depth classes are <30, 30-50, 50-70, 70-100, 100-150 and ≥150 ft, and unknown depth. For each class the prior mean $m$ and standard deviation $\sigma$ of $y = \log_{10} \max(\mathrm{As}, 1)$ are estimated on the fitting half. The residual $z = y - m$ is modelled as a zero-mean Gaussian process with covariance $C(h) = \sigma_s^2 \exp(-h/\rho)$ and nugget $\tau^2$, on planar coordinates in metres (local projection, not standardised). Hyperparameters are fitted by maximum likelihood (scikit-learn GaussianProcessRegressor with a constant kernel times a Matérn kernel with $\nu = 0.5$, plus a white-noise kernel; bounds 0.001-5, 10-5,000 m and 0.0001-2) on 1,500 shallow wells drawn at random from the fitting half, separately for each fitting half and run. Deep (>100 ft) and shallow wells form separate layers with the same hyperparameters, and an untested well is conditioned only on tested wells of its own layer (wells of unknown depth are shallow). The posterior is computed by Cholesky decomposition.

Kit readings are ordered categories (0, 10, 25, 50, 100, 200, 300, 500 and about 1000 µg/L) and enter the model as $y = \log_{10} \max(\mathrm{reading}, 1)$, so a reading of 0 gives $y = 0$. In the upazila survey the model is fitted to kit readings, so its nugget includes kit error. In the census it is fitted to laboratory values, which the simulator treats as the truth, so its nugget contains no kit error. In the sensitivity analysis of Table A7, the kit-error variance $\kappa^2 = 0.113$ (variance of $\log_{10}$ kit minus $\log_{10}$ laboratory value over the 944 pairs) is added once to the census nugget. Because $\kappa^2$ is measured against the laboratory value, which is the truth by construction, it contains no laboratory error and nothing is counted twice. The covariance of the observed readings becomes $\mathbf{K} + (\tau^2 + \kappa^2)\mathbf{I}$, and the predictive variance then refers to the kit reading the team will see. In the census, the kit reading for a well is drawn at random from the 25 of the 944 kit-laboratory pairs whose laboratory values are closest (in $\log_{10}$) to the well's laboratory value. In the upazila survey, the recorded kit reading is observed and the true laboratory value of each well is drawn for each run from the pairs with the same kit reading.

## Costings

Persons protected are exposed households protected × 8 persons per well. The time-based costing charges US$28.80 per team-day, the all-in cost of the 12 tests one tester completed per day in Araihazar in 2012-2013 (US$2.40 per test, including labour, supervision, supplies and a placard). The time-plus-kit costing charges 12 × (US$2.40 - US$0.17) per team-day plus US$0.17 for each test performed, separating kit consumables from the daily cost of the team. The per-test costing charges US$2.40 for every test performed. Costs are pooled over blocks at the primary effort (Table A6).

## Inference

For a comparison, $d_b$ is the mean over the runs of block *b* of the paired difference (learning minus comparator). The *t*-interval and *t*-test use the $d_b$. The sign-flip permutation test compares $|\bar{d}|$ with its distribution when the signs of the $d_b$ are flipped at random (exact enumeration of all $2^{10}$ patterns in the census). The Wilcoxon signed-rank test is two-sided. Leave-one-block-out gives the mean of the $d_b$ with each block removed in turn. For the super-blocks, the 2 km grid indices of the upazila blocks are halved, giving 17 cells of 4 km × 4 km within which the $d_b$ are averaged. The Monte Carlo standard error is the standard deviation of the run-level differences within a block divided by the square root of the number of runs, averaged over blocks (Table A5, Figure A1).

# Appendix B. Further Results

![**Figure A1.** Paired difference in exposed households protected (learning minus comparator, percentage points) after 5 team-days per 1,000 wells (50 µg/L, 100 m) in every block. Open circles: the five confirmatory census blocks; diamonds: mean over blocks with 95% *t*-interval.](figures/figS1_blocks.png)
"""

HISTORY = [
    ("v1 (pilot)", "Outcome 'access' (household with a tested-safe well within 100 m); test counts, no travel; random forest learner", "Development blocks only", "Replaced: the outcome counted households whose own well was already safe"),
    ("v2", "Access outcome; 10 min per test, then amended (before viewing test results) to 20 min and a 10-day horizon; random-forest headline; uncertainty sampling", "Five test blocks (first test-block evaluation)", "Seeding error found by internal review; re-run as v3"),
    ("v3", "As v2 with shared random numbers; oracle, risk-map plan, capacity and distance-decay analyses", "Five test blocks (second look)", "Replaced: access outcome judged inappropriate for public health"),
    ("v4", "Cross-fitting run of v3 started on the other half", "Stopped before completion", "Superseded by the v5 outcome change"),
    ("v5", "Protection outcome; look-ahead score (weight 0.1 for all policies); Gaussian-process headline (chosen on development blocks); uncertainty sampling and risk-map plan dropped; 10 runs, 20 days", "10 census blocks (cross-fitted); 2012-13 upazila survey (first look, kit readings as truth)", "Replaced after a second internal review"),
    ("v6 (this paper)", "Per-policy look-ahead weight; uncertainty sampling restored; repeated random splits; primary endpoint (5 team-days per 1,000 wells, Holm) declared before final runs; 2012-13 truth imputed from kit-laboratory pairs; 100 m link check; field realism, capacity and switching outcomes; non-overlap set defined as ≥20 wells within 100 m of a census well", "10 census blocks (5 splits); 52 upazila blocks (2 splits; second look)", "Reported"),
]


def main():
    out = [TEXT_A]
    # A1: all comparisons
    S1 = pd.read_csv(R / "f1_v6_table_S1.csv")
    S1["Analysis"] = S1.setting.map(ANALYSIS).fillna(S1.setting)
    S1["Setting"] = [f"{int(t)} µg/L, {int(r)} m" + ("" if th >= 1 else f", {int(100 * th)}% of wells")
                     for t, r, th in zip(S1.thr, S1.radius, S1.thin)]
    S1["Effort"] = S1.horizon.replace({"5 d/1000 wells": "primary", "5 days": "5 days", "10 days": "10 days", "20 days": "20 days"})
    A1 = pd.DataFrame({"Analysis": S1.Analysis, "Setting": S1.Setting, "Effort": S1.Effort,
                       "Comparator": S1.comparator.map(SHORT).fillna(S1.comparator),
                       "Learning (%)": S1.a_pct.map("{:.1f}".format), "Comparator (%)": S1.b_pct.map("{:.1f}".format),
                       "Difference (95% CI)": [ci(*x) for x in zip(S1["diff"], S1.lo, S1.hi)], "*p*": S1.p.map(p),
                       "Blocks": S1.units_positive, "Relative gain (%)": S1.rel_pooled.map("{:+.0f}".format)})
    out.append(table("**Table A1.** All comparisons of the learning policy (Gaussian process) with other policies. Effort 'primary' is 5 team-days per 1,000 wells; the capacity and expected-switching outcomes are after ten team-days. Difference in percentage points of exposed households protected (mean over blocks, 95% *t*-interval); *p*, paired *t*-test over blocks (unadjusted); blocks favouring learning; relative gain in the well-weighted share.",
                     r"L{3.0cm}L{2.0cm}L{1.0cm}L{1.6cm}R{1.0cm}R{1.4cm}L{2.4cm}R{0.9cm}R{0.8cm}R{0.9cm}", A1, blank_repeats=("Analysis",)))
    # A2: learners
    S2 = pd.read_csv(R / "f1_v6_table_S2.csv")
    A2 = pd.DataFrame({"Threshold (µg/L)": S2.thr.astype(int), "Learner compared": S2.comparator.map(SHORT).fillna(S2.comparator),
                       "Gaussian process (%)": S2.a_pct.map("{:.1f}".format), "Other learner (%)": S2.b_pct.map("{:.1f}".format),
                       "Difference (95% CI)": [ci(*x) for x in zip(S2["diff"], S2.lo, S2.hi)], "*p*": S2.p.map(p),
                       "Blocks": S2.units_positive})
    out.append(table("**Table A2.** The Gaussian-process learner compared with the other learners (2000-01 census, 100 m, kit error, primary effort; separate run with 2 random splits and 4 runs per block).",
                     r"L{1.6cm}L{2.4cm}R{1.6cm}R{1.6cm}L{2.4cm}R{1.0cm}R{1.0cm}", A2, blank_repeats=("Threshold (µg/L)",)))
    # A3: equity
    S3 = pd.read_csv(R / "f1_v6_table_S3.csv")
    A3 = pd.DataFrame({"Survey": S3.survey, "Threshold (µg/L)": S3.thr.astype(int), "Tertile": S3.tertile,
                       "Comparator": S3.comparator.map(SHORT).fillna(S3.comparator),
                       "Learning (%)": S3.a_pct.map("{:.1f}".format), "Comparator (%)": S3.b_pct.map("{:.1f}".format),
                       "Difference (95% CI)": [ci(*x) for x in zip(S3["diff"], S3.lo, S3.hi)], "*p*": S3.p.map(p),
                       "Blocks": S3.units_positive})
    out.append(table("**Table A3.** Exposed households protected after ten team-days, by tertile (within each block) of the local share of safe wells within 200 m; learning compared with door-to-door testing and the static plan.",
                     r"L{2.6cm}R{1.2cm}L{1.6cm}L{1.8cm}R{1.2cm}R{1.4cm}L{2.4cm}R{1.0cm}R{1.0cm}", A3, blank_repeats=("Survey",)))
    # A4: history
    A4 = pd.DataFrame(HISTORY, columns=["Version", "Design", "Evaluated on", "Status"])
    out.append(table("**Table A4.** Design and analysis history. Earlier versions were evaluated on held-out blocks before the present design was fixed; the outcome, score and comparators changed between versions as shown. The primary endpoint of v6 was declared before its final runs. Note: the Scopus search cited in the Introduction used TITLE-ABS-KEY(arsenic AND (well∗ OR tubewell∗) AND (test∗ OR screening OR sampling)), 1995-2026, retrieved 24 November 2026 (5,348 records, 5,212 with reference lists).",
                     r"L{1.6cm}L{8.0cm}L{3.8cm}L{3.6cm}", A4))
    # A5: further tests
    T5 = pd.read_csv(R / "f1_v6_inference.csv")
    A5 = pd.DataFrame({"Survey": T5.survey, "Comparator": T5.comparator.map(SHORT).fillna(T5.comparator), "Blocks": T5.blocks,
                       "Median difference": T5["median"].map("{:.1f}".format),
                       "Holm *p*: *t*; permutation; Wilcoxon": [f"{p(a)}; {p(b)}; {p(c)}" for a, b, c in zip(T5.p_t_holm, T5.p_perm_holm, T5.p_wilcoxon_holm)],
                       "Leave one block out": [f"{a:.1f} to {b:.1f}" for a, b in zip(T5.loo_min, T5.loo_max)],
                       "Super-blocks (95% CI); *p*": [f"{ci(d, lo, hi)}; {p(q)}" if pd.notna(d) else "-" for d, lo, hi, q in zip(T5.super_diff, T5.super_lo, T5.super_hi, T5.super_p_perm)],
                       "Confirmatory (95% CI); *p*": [f"{ci(d, lo, hi)}; {q:.4f}" if pd.notna(d) else "-" for d, lo, hi, q in zip(T5.conf_diff, T5.conf_lo, T5.conf_hi, T5.conf_p_perm)],
                       "MC SE / SD between blocks": [f"{a:.2f} / {b:.2f}" for a, b in zip(T5.mc_se, T5.sd_between)]})
    out.append(table("**Table A5.** Further tests of the six primary comparisons (learning against each comparator; 5 team-days per 1,000 wells, 50 µg/L, 100 m), in percentage points. Permutation: paired sign-flip test, exact for the 10 census blocks, 200,000 random patterns otherwise; Holm adjustment over the six comparisons for each test. Super-blocks: 17 upazila cells of 4 km × 4 km, permutation *p*. Confirmatory: the five census blocks never used for a design choice, exact permutation *p* (smallest attainable 0.0625). MC SE: Monte Carlo standard error of a block's mean difference over its runs, against the standard deviation of the block means.",
                     r"L{1.8cm}L{1.6cm}R{0.8cm}R{1.1cm}L{2.4cm}L{1.5cm}L{2.4cm}L{2.4cm}L{1.4cm}", A5, blank_repeats=("Survey",)))
    # A6: costs
    rows = []
    for tag, lab in (("main", "2000-01 census"), ("bd2012", "2012-13 upazila survey"), ("field", "Census, field realism")):
        c = pd.read_csv(R / f"f1_v6_cost_{tag}.csv", index_col=0)
        for arm, name in (("sweep:-", "Door-to-door"), ("static:-", "Static plan"), ("adaptive:-", "Adaptive"), ("learning:gp", "Learning")):
            r = c.loc[arm]
            rows.append({"Survey": lab, "Policy": name, "Tests per team-day": f"{r.tests_per_day:.1f}",
                         "Time-based (US$)": f"{r.time_based:.2f}", "Time plus kit (US$)": f"{r.mixed:.2f}", "Per test (US$)": f"{r.per_test:.2f}"})
    out.append(table("**Table A6.** Estimated field cost per person protected in the simulation at the primary effort (5 team-days per 1,000 wells, 50 µg/L, 100 m; 8 persons per well; pooled over blocks), under the three costings of Appendix A. Transport to the area, training, smartphones and programme management are excluded.",
                     r"L{3.4cm}L{2.2cm}R{1.8cm}R{1.8cm}R{1.8cm}R{1.6cm}", pd.DataFrame(rows), blank_repeats=("Survey",)))
    # A7: sensitivity
    T7 = pd.read_csv(R / "f1_v6_table_S7.csv")
    A7 = pd.DataFrame({"Survey": T7.survey, "Analysis": T7.analysis.str.replace("ug/L", "µg/L"), "Threshold (µg/L)": T7.thr.astype(int),
                       "Comparator": T7.comparator.map(SHORT).fillna(T7.comparator),
                       "Learning, main analysis (%)": T7.learning_base.map("{:.1f}".format), "Learning (%)": T7.a_pct.map("{:.1f}".format),
                       "Comparator (%)": T7.b_pct.map("{:.1f}".format),
                       "Difference (95% CI)": [ci(*x) for x in zip(T7["diff"], T7.lo, T7.hi)], "*p*": T7.p.map(p), "Blocks": T7.units_positive})
    out.append(table("**Table A7.** Sensitivity of the primary comparisons (5 team-days per 1,000 wells, 100 m) to measurement assumptions of the Gaussian process: kit readings of 0 set to 5 µg/L (half the lowest non-zero category) instead of 1 µg/L, in both surveys; and the kit-error variance of the 944 kit-laboratory pairs (0.113 in squared $\\log_{10}$ units) added to the census nugget (Appendix A). Same design as the main analysis; paired *t*-test over blocks, unadjusted.",
                     r"L{2.0cm}L{2.4cm}R{1.2cm}L{1.6cm}R{1.4cm}R{1.2cm}R{1.3cm}L{2.2cm}R{1.0cm}R{0.9cm}", A7, blank_repeats=("Survey", "Analysis")))
    (M / "appendix.md").write_text("\n".join(out), encoding="utf-8")
    print("appendix.md written:", sum(len(x.splitlines()) for x in out), "lines")


if __name__ == "__main__":
    main()
