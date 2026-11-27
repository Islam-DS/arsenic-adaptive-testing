"""Independent recomputation of the reported numbers from the campaign-level results.   python -X utf8 src/f1/audit.py

Written separately from figures6.py and inference.py (own interpolation, scipy.stats.ttest_rel / t.interval,
statsmodels Holm, scipy.stats.permutation_test) and compared with the manuscript text, Table 2 and Tables S5-S7.
Writes results/f1_v6_audit.csv and prints every check with its status.
"""
import pathlib, re
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "results"
MD = (ROOT / "manuscript" / "manuscript.md").read_text(encoding="utf-8")
CAMPAIGN = ["split", "block", "policy", "learner", "seed", "thr", "radius", "thin", "advise", "known", "lam",
            "daily_return", "absent", "cap", "impute"]
LEARN, COMPS = "learning|gp", ["sweep|-", "static|-", "adaptive|-"]
CONFIRM = {"0_0", "0_1", "1_0", "1_1", "3_2"}
checks = []


def check(name, reported, recomputed, tol):
    ok = abs(float(reported) - float(recomputed)) <= tol
    checks.append(dict(check=name, reported=reported, recomputed=round(float(recomputed), 4), tolerance=tol,
                       status="OK" if ok else "MISMATCH"))


def at_effort(df, col="protected_share", per1000=5.0):
    """Value of `col` at 5 team-days per 1,000 wells for every campaign (linear between recorded days, from 0 at day 0)."""
    out = []
    for key, g in df.groupby([c for c in CAMPAIGN if c in df.columns], dropna=False):
        g = g.sort_values("days")
        target = min(per1000 * g["n"].iloc[0] / 1000.0, g["days"].max())
        val = np.interp(target, np.concatenate([[0.0], g["days"].to_numpy()]), np.concatenate([[0.0], g[col].to_numpy()]))
        out.append(dict(block=g["block"].iloc[0], arm=f"{g['policy'].iloc[0]}|{g['learner'].iloc[0]}", thr=g["thr"].iloc[0],
                        value=val, exposed=g["exposed"].iloc[0], n=g["n"].iloc[0], days=target, tests_col=col))
    return pd.DataFrame(out)


def paired(e, a, b, blocks=None):
    m = e.groupby(["block", "arm"]).agg(value=("value", "mean"), exposed=("exposed", "mean")).reset_index()
    if blocks is not None:
        m = m[m.block.isin(blocks)]
    va = m[m.arm == a].set_index("block"); vb = m[m.arm == b].set_index("block").reindex(va.index)
    d = 100 * (va.value - vb.value)
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    p = stats.ttest_rel(100 * va.value, 100 * vb.value).pvalue
    pooled = lambda v: (v.value * va.exposed).sum() / va.exposed.sum()
    return dict(a=100 * va.value.mean(), b=100 * vb.value.mean(), diff=d.mean(), lo=lo, hi=hi, p=p,
                pos=int((d > 0).sum()), n=len(d), rel=100 * (pooled(va) / pooled(vb) - 1), d=d)


def load(tag):
    return pd.read_csv(RES / f"f1_{tag}.csv")


# ---------------------------------------------------------------- Table 2 and the primary family
table2 = [r for r in re.findall(r"^\|(.+)\|$", MD, re.M) if re.search(r"\d+/\d+", r)]
rows = [[c.strip() for c in r.split("|")] for r in table2]
surveys = {"census": load("v6_main"), "upazila": load("v6_bd2012")}
eff = {k: at_effort(v[v.thr == 50]) for k, v in surveys.items()}
prim, i = [], 0
for key, label in (("census", "census"), ("confirm", "census, 5 confirmatory"), ("upazila", "upazila")):
    for c in COMPS:
        r = paired(eff["upazila" if key == "upazila" else "census"], LEARN, c, CONFIRM if key == "confirm" else None)
        row = rows[i]; i += 1
        lo, hi = re.findall(r"\(([-\d.]+), ([-\d.]+)\)", row[4])[0]
        check(f"Table 2 {label} vs {c}: comparator %", row[2], r["b"], 0.05)
        check(f"Table 2 {label} vs {c}: learning %", row[3], r["a"], 0.05)
        check(f"Table 2 {label} vs {c}: difference", row[4].split()[0], r["diff"], 0.05)
        check(f"Table 2 {label} vs {c}: CI low", lo, r["lo"], 0.05)
        check(f"Table 2 {label} vs {c}: CI high", hi, r["hi"], 0.05)
        check(f"Table 2 {label} vs {c}: blocks positive", row[6].split("/")[0], r["pos"], 0)
        check(f"Table 2 {label} vs {c}: relative gain %", row[7].strip("+%"), r["rel"], 0.5)
        if key != "confirm":
            prim.append((label, c, r["p"], row[5]))
        else:
            check(f"Table 2 {label} vs {c}: unadjusted p", row[5], r["p"], 0.0005)
holm = multipletests([x[2] for x in prim], method="holm")[1]
for (label, c, _, rep), h in zip(prim, holm):
    check(f"Table 2 {label} vs {c}: Holm p", rep, h, 0.0005)
check("Results/abstract: max Holm p <= 0.011", 0.011, max(holm), 0.0005)

# ---------------------------------------------------------------- permutation and Wilcoxon (Table S5)
S5 = pd.read_csv(RES / "f1_v6_inference.csv")
perm, wil = [], []
for key in ("census", "upazila"):
    for c in COMPS:
        d = paired(eff[key], LEARN, c)["d"].to_numpy()
        n_res = np.inf if len(d) <= 12 else 200_000
        perm.append(stats.permutation_test((d,), np.mean, permutation_type="samples", n_resamples=n_res,
                                           alternative="two-sided", random_state=1).pvalue)
        wil.append(stats.wilcoxon(d).pvalue)
perm_h, wil_h = multipletests(perm, method="holm")[1], multipletests(wil, method="holm")[1]
for j in range(6):
    check(f"Table S5 row {j + 1}: Holm p, permutation", S5.p_perm_holm[j], perm_h[j], 0.002)
    check(f"Table S5 row {j + 1}: Holm p, Wilcoxon", S5.p_wilcoxon_holm[j], wil_h[j], 0.0005)
check("Results: permutation and Wilcoxon Holm p <= 0.010", 0.010, max(perm_h.max(), wil_h.max()), 0.0005)

# ---------------------------------------------------------------- abstract numbers
rel = {(k, c): paired(eff[k], LEARN, c)["rel"] for k in eff for c in COMPS}
check("Abstract: +61% vs door-to-door (census)", 61, rel[("census", "sweep|-")], 0.5)
check("Abstract: +53% vs door-to-door (upazila)", 53, rel[("upazila", "sweep|-")], 0.5)
check("Abstract: 33% (lower end, vs static)", 33, min(rel[("census", "static|-")], rel[("upazila", "static|-")]), 0.5)
check("Abstract: 41% (upper end, vs static)", 41, max(rel[("census", "static|-")], rel[("upazila", "static|-")]), 0.5)

# ---------------------------------------------------------------- costs (Table S6, abstract, results)
def cost(df):
    e = at_effort(df[df.thr == 50]); t = at_effort(df[df.thr == 50], "tests")
    e["tests"] = t["value"].to_numpy()
    out = {}
    for arm, g in e.groupby("arm"):
        persons = (g.value * g.exposed).sum() * 8
        out[arm] = dict(time=g.days.sum() * 28.80 / persons, per_test=g.tests.sum() * 2.40 / persons,
                        mixed=(g.days.sum() * 12 * 2.23 + g.tests.sum() * 0.17) / persons, tpd=g.tests.sum() / g.days.sum())
    return out
cc, cu = cost(surveys["census"]), cost(surveys["upazila"])
for (lab, c), rep in {(("census", "learning|gp"), "time"): 0.35, (("census", "sweep|-"), "time"): 0.56,
                      (("upazila", "learning|gp"), "time"): 0.41, (("upazila", "sweep|-"), "time"): 0.62,
                      (("census", "adaptive|-"), "time"): 0.41, (("census", "static|-"), "time"): 0.49,
                      (("upazila", "adaptive|-"), "time"): 0.49, (("upazila", "static|-"), "time"): 0.55,
                      (("census", "learning|gp"), "per_test"): 0.47, (("census", "sweep|-"), "per_test"): 0.80,
                      (("census", "learning|gp"), "tpd"): 16.2, (("census", "sweep|-"), "tpd"): 17.2}.items():
    src = cc if lab[0] == "census" else cu
    check(f"Cost {lab[0]} {lab[1]} {c}", rep, src[lab[1]][c], 0.005 if c != "tpd" else 0.05)
S6 = pd.read_csv(RES / "f1_v6_cost_main.csv", index_col=0)
check("Table S6 census learning, time plus kit", round(S6.loc["learning:gp", "mixed"], 2), cc["learning|gp"]["mixed"], 0.005)

# ---------------------------------------------------------------- Table S7 and its text claims
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import load_kit
k = load_kit()
kv = np.var(np.log10(np.clip(k["kit"], 1, None)) - np.log10(k["lab"].clip(lower=1)), ddof=1)
check("Methods/Table S7: kit-error variance of the 944 pairs", 0.113, kv, 0.0005)
base = {"census": eff["census"], "upazila": eff["upazila"]}
level_change, gain_change = [], []
for tag, key in (("v6_zero5", "census"), ("v6_bd2012_zero5", "upazila"), ("v6_kitnug", "census")):
    s = at_effort(load(tag).query("thr == 50"))
    lb = base[key].query("arm == @LEARN").groupby("block").value.mean().mean()
    ls = s.query("arm == @LEARN").groupby("block").value.mean().mean()
    level_change.append(100 * abs(ls - lb))
    for c in COMPS:
        rs, rb = paired(s, LEARN, c), paired(base[key], LEARN, c)
        gain_change.append(abs(rs["diff"] - rb["diff"]))
        check(f"Table S7 {tag} vs {c}: significant at 50 ug/L (1 = p < 0.05)", 1, int(rs["p"] < 0.05), 0)
check("Results: learning protection changes by at most 0.1 points", 0.1, max(level_change), 0.05)
check("Results: gains change by at most 0.3 points", 0.3, max(gain_change), 0.05)

# ---------------------------------------------------------------- wording
indep = re.findall(r"[^.]*independent[^.]*\.", MD)
bad = [s for s in indep if "independent observations" not in s]      # statistical independence of runs is fine
checks.append(dict(check="Wording: 'independent' used for validation", reported=len(bad), recomputed=None, tolerance=None,
                   status="REVIEW" if bad else "OK"))

A = pd.DataFrame(checks)
A.to_csv(RES / "f1_v6_audit.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_colwidth", 70)
print(A.to_string())
print("\n", A.status.value_counts().to_string())
for s in indep:
    print("  ->", s.strip()[:200])
