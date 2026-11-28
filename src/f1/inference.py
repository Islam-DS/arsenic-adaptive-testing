"""Further inference for the primary endpoint and the cost analysis.   python -X utf8 src/f1/inference.py

* paired t-test (as in figures6) plus a sign-flip permutation test (exact for the 10 census blocks, 200,000 random
  sign flips for the 52 upazila blocks) and a Wilcoxon signed-rank test, each Holm-adjusted over the primary family;
* confirmatory set: the five census blocks never used for any design choice;
* influence of single blocks (leave-one-block-out range of the mean difference);
* spatial dependence: upazila blocks aggregated into 4 km x 4 km super-blocks;
* Monte Carlo error of block means (runs and splits) against the spread between blocks;
* cost per person protected under three costings;
* Figure A1: paired difference in every block.
"""
import itertools, pathlib, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from figures6 import load, effort, unit_means, holm, HEAD, ARMS, EFFORT, RES, FIG

CONFIRM = {"araihazar/0_0", "araihazar/0_1", "araihazar/1_0", "araihazar/1_1", "araihazar/3_2"}  # data.load_wells split == "test"
COMPARATORS = ("sweep:-", "static:-", "adaptive:-")
PERSONS_PER_WELL = 8
USD_PER_TEST_ALLIN = 2.40      # labour, supervision, supplies and placard, Araihazar 2012-13
TESTS_PER_DAY_FIELD = 12       # reported throughput of one tester
USD_KIT = 0.17                 # kit consumables per test


def signflip_p(x, n_mc=200_000, seed=0):
    """Two-sided paired sign-flip test of mean(x) = 0. Exact enumeration up to 16 units (the observed pattern is one of
    the 2^n patterns); otherwise n_mc random patterns with the (b + 1) / (B + 1) estimate, which is never zero."""
    x = np.asarray(x, float)
    obs = abs(x.mean())
    if len(x) <= 16:
        signs = np.array(list(itertools.product((1, -1), repeat=len(x))))
        return float((np.abs((signs * x).mean(1)) >= obs - 1e-12).mean())
    signs = np.random.default_rng(seed).choice((1, -1), size=(n_mc, len(x)))
    b = int((np.abs((signs * x).mean(1)) >= obs - 1e-12).sum())
    return (b + 1) / (n_mc + 1)


def tci(x):
    x = np.asarray(x, float); n = len(x)
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return x.mean(), x.mean() - h, x.mean() + h


def block_diffs(e, b):
    u = unit_means(e[e.thr == 50], "protected_share").pivot(index="unit", columns="arm", values="protected_share")
    return 100 * (u[HEAD] - u[b]).dropna()


def superblock(unit):
    ix, iy = unit.split("/")[1].split("_")
    return f"{int(ix) // 2}_{int(iy) // 2}"


def mc_error(e, b):
    """Monte Carlo SE of each block's mean difference against the SD of block means (percentage points)."""
    s = e[e.thr == 50]
    w = s.pivot_table(index=["unit", "split", "seed"], columns="arm", values="protected_share")
    d = 100 * (w[HEAD] - w[b]).dropna().reset_index(name="d")
    g = d.groupby("unit").d
    se = (g.std(ddof=1) / np.sqrt(g.size())).mean()
    return se, g.mean().std(ddof=1), int(g.size().median())


def costs(d):
    """USD per person protected at the primary effort, pooled over blocks (50 ug/L)."""
    p = effort(d)[lambda x: x.thr == 50]
    t = effort(d, "tests")[lambda x: x.thr == 50]
    keys = [c for c in p.columns if c not in ("protected_share", "exposed", "n")]
    m = p.merge(t[keys + ["tests"]], on=keys)
    out = {}
    for arm, g in m.groupby("arm"):
        persons = (g.protected_share * g.exposed).sum() * PERSONS_PER_WELL
        days = (EFFORT * g.n / 1000).sum()
        tests = g.tests.sum()
        out[arm] = dict(
            time_based=days * TESTS_PER_DAY_FIELD * USD_PER_TEST_ALLIN / persons,
            mixed=(days * TESTS_PER_DAY_FIELD * (USD_PER_TEST_ALLIN - USD_KIT) + tests * USD_KIT) / persons,
            per_test=tests * USD_PER_TEST_ALLIN / persons,
            tests_per_day=tests / days)
    return pd.DataFrame(out).T


def fig_blocks(diffs):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
    for ax, survey in zip(axes, ("2000-01 census", "2012-13 upazila survey")):
        for k, b in enumerate(COMPARATORS):
            x = diffs[(survey, b)]
            jit = np.random.default_rng(k).uniform(-0.18, 0.18, len(x))
            conf = x.index.isin(CONFIRM)
            ax.scatter(k + jit[~conf], x[~conf], s=14, color=ARMS[b][1], alpha=0.8, lw=0)
            if conf.any():
                ax.scatter(k + jit[conf], x[conf], s=22, facecolor="white", edgecolor="#1d6a3a", lw=1.2)
            m, lo, hi = tci(x)
            ax.errorbar(k + 0.32, m, yerr=[[m - lo], [hi - m]], fmt="D", ms=5, color="black", capsize=3)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(range(3), ["vs door-\nto-door", "vs static\nplan", "vs\nadaptive"], fontsize=9)
        ax.set_title(f"{survey} ({len(x)} blocks)", fontsize=10)
        ax.set_ylabel("Learning minus comparator (pp)", fontsize=9)
    axes[0].scatter([], [], s=22, facecolor="white", edgecolor="#1d6a3a", lw=1.2, label="block never used for design")
    axes[0].errorbar([], [], fmt="D", color="black", label="mean, 95% t-interval")
    fig.legend(frameon=False, fontsize=8.5, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(FIG / "figS1_blocks.png"); plt.close(fig)


def bootstrap_truth(n_boot=200, seed=0):
    """Table A8 and A10: uncertainty of the imputed 2012-13 laboratory truth.
    (a) Kit categories of the 944 pairs: n, median laboratory value, share safe at 10 and 50 ug/L.
    (b) Share of the 46,976 wells truly safe, imputing from n_boot bootstrap samples of the 944 pairs.
    (c) Primary comparisons in each of 20 bootstrap replicates of the policy runs (v6_bd2012_boot; one bootstrap
        sample per split x seed, shared by all blocks and policies)."""
    from data import load_kit
    from data_ext import load_bd2012
    from experiment4 import impute_lab
    from figures6 import compare
    k = load_kit()
    cat = k.groupby("kit").lab.agg(n="size", median="median", safe10=lambda s: (s <= 10).mean(),
                                   safe50=lambda s: (s <= 50).mean()).reset_index()
    cat.to_csv(RES / "f1_v6_table_A10.csv", index=False)
    kits = load_bd2012().as_ppb.to_numpy()
    rng = np.random.default_rng(seed)
    share = []
    for _ in range(n_boot):
        kb = k.sample(frac=1.0, replace=True, random_state=int(rng.integers(1 << 31)))
        lab = impute_lab(kb, kits, rng)
        share.append(((lab <= 50).mean(), (lab <= 10).mean()))
    share = np.array(share)
    pd.DataFrame(100 * share, columns=["truly_safe_50_pct", "truly_safe_10_pct"]).to_csv(RES / "f1_v6_boot_truth.csv", index=False)
    try:
        b = effort(load("v6_bd2012_boot"))
    except FileNotFoundError:
        return
    b = b[b.thr == 50]
    rows = []
    for (split, seed_), g in b.groupby(["split", "seed"]):
        r = {"split": split, "seed": seed_,
             "learning": 100 * unit_means(g, "protected_share").query("arm == @HEAD").protected_share.mean()}
        for c in COMPARATORS:
            x = compare(g, HEAD, c)
            r[c] = x["diff"]; r[c + "_rel"] = x["rel_pooled"]
        rows.append(r)
    R = pd.DataFrame(rows)
    summ = {"truly safe at 50 ug/L (%)": 100 * share[:, 0], "truly safe at 10 ug/L (%)": 100 * share[:, 1],
            "learning protected (%)": R.learning.to_numpy()}
    for c, lab in zip(COMPARATORS, ("door-to-door", "static plan", "adaptive")):
        summ[f"gain vs {lab} (pp)"] = R[c].to_numpy()
        summ[f"relative gain vs {lab} (%)"] = R[c + "_rel"].to_numpy()
    A8 = pd.DataFrame([dict(quantity=q, replicates=len(v), mean=np.mean(v), lo=np.percentile(v, 2.5),
                            hi=np.percentile(v, 97.5), min=np.min(v), positive=int((v > 0).sum()))
                       for q, v in summ.items()])
    A8.to_csv(RES / "f1_v6_table_A8.csv", index=False)
    R.to_csv(RES / "f1_v6_boot_replicates.csv", index=False)
    print(A8.round(3).to_string())


def sensitivity():
    """Table A7: measurement assumptions of the Gaussian process (kit readings of 0; kit error in the nugget)."""
    from figures6 import compare
    rows = []
    for lab, base, tag in (("2000-01 census", "v6_main", "v6_zero5"), ("2012-13 upazila survey", "v6_bd2012", "v6_bd2012_zero5"),
                           ("2000-01 census", "v6_main", "v6_kitnug"), ("2000-01 census", "v6_main", "v6_buffer"),
                           ("2012-13 upazila survey", "v6_bd2012", "v6_bd2012_buffer")):
        try:
            s = effort(load(tag))
        except FileNotFoundError:
            continue
        b = effort(load(base))
        name = ("kit reading 0 set to 5 ug/L" if "zero5" in tag else "kit-error variance added to nugget" if "kitnug" in tag
                else "600 m exclusion buffer around evaluated wells")
        for thr in (50, 10):
            ss, bb = s[s.thr == thr], b[b.thr == thr]
            if ss.empty:
                continue
            base_learn = 100 * unit_means(bb, "protected_share").query("arm == @HEAD").protected_share.mean()
            for c in COMPARATORS:
                r = compare(ss, HEAD, c)
                rows.append(dict(survey=lab, analysis=name, thr=thr, comparator=ARMS[c][0], learning_base=base_learn, **r))
    S7 = pd.DataFrame(rows); S7.to_csv(RES / "f1_v6_table_A7.csv", index=False)
    print(S7[["survey", "analysis", "thr", "comparator", "learning_base", "a_pct", "b_pct", "diff", "lo", "hi", "p",
              "units_positive", "rel_pooled"]].round(3).to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    e = {"2000-01 census": effort(load("v6_main")), "2012-13 upazila survey": effort(load("v6_bd2012"))}
    rows, diffs = [], {}
    for survey, ee in e.items():
        for b in COMPARATORS:
            x = block_diffs(ee, b); diffs[(survey, b)] = x
            m, lo, hi = tci(x)
            loo = [x.drop(k).mean() for k in x.index]
            r = dict(survey=survey, comparator=ARMS[b][0], blocks=len(x), diff=m, lo=lo, hi=hi,
                     p_t=stats.ttest_1samp(x, 0).pvalue, p_perm=signflip_p(x), p_wilcoxon=stats.wilcoxon(x).pvalue,
                     loo_min=min(loo), loo_max=max(loo), median=x.median(), positive=int((x > 0).sum()))
            if survey.startswith("2000"):
                c = x[x.index.isin(CONFIRM)]
                cm, clo, chi = tci(c)
                r.update(conf_blocks=len(c), conf_diff=cm, conf_lo=clo, conf_hi=chi, conf_p_t=stats.ttest_1samp(c, 0).pvalue,
                         conf_p_perm=signflip_p(c), conf_positive=int((c > 0).sum()))
            else:
                sb = x.groupby(x.index.map(superblock)).mean()
                sm, slo, shi = tci(sb)
                r.update(super_blocks=len(sb), super_diff=sm, super_lo=slo, super_hi=shi,
                         super_p_t=stats.ttest_1samp(sb, 0).pvalue, super_p_perm=signflip_p(sb))
            se, sd_between, runs = mc_error(ee, b)
            r.update(mc_se=se, sd_between=sd_between, runs_per_block=runs)
            rows.append(r)
    T = pd.DataFrame(rows)
    for c in ("p_t", "p_perm", "p_wilcoxon"):
        T[c + "_holm"] = holm(T[c])
    T.to_csv(RES / "f1_v6_table_A5.csv", index=False)
    print(T.round(4).T.to_string())
    fig_blocks(diffs)
    for survey, tag in (("2000-01 census", "v6_main"), ("2012-13 upazila survey", "v6_bd2012")):
        C = costs(load(tag)); C.to_csv(RES / f"f1_v6_cost_{tag.split('_')[1]}.csv")
        print(survey, "\n", C.loc[["sweep:-", "static:-", "adaptive:-", HEAD]].round(3).to_string())
    for tag, out in (("v6_field", "field"), ("v6_t29", "t29"), ("v6_bd2012_t29", "bd2012_t29")):
        C = costs(load(tag)); C.to_csv(RES / f"f1_v6_cost_{out}.csv")
        print(tag, "\n", C.loc[["sweep:-", "static:-", "adaptive:-", HEAD]].round(3).to_string())
    sensitivity()
    bootstrap_truth()
