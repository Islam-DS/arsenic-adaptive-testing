"""Independent recomputation of the published numbers from the campaign-level results.   python -X utf8 src/f1/audit.py

Written separately from figures6.py and inference.py (own interpolation to the primary effort, scipy.stats.ttest_rel and
t.interval, Holm correction from statsmodels, scipy.stats.permutation_test). It recomputes every cell of Table 2 and of
Appendix Tables A1-A8 and A10, and the numbers in the abstract and in selected sentences of the text, and compares them
with what the manuscript prints (manuscript/manuscript.md and manuscript/appendix.md).

Tolerance: a printed value passes if it equals the recomputed value up to rounding (half a unit of its last printed
digit); Monte Carlo permutation p-values may also differ by three Monte Carlo standard errors. Table A9 (probability
calibration) is not recomputed here. Checks that need the raw data (not redistributed) are reported as SKIPPED when
the data are absent. The script exits with status 1 if any check fails. Results: results/f1_v6_audit.csv.
"""
import itertools, math, pathlib, re, sys
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "results"
DATA_OK = (ROOT / "data" / "raw" / "araihazar" / "Dataset1.csv").exists() and \
          (ROOT / "data" / "raw" / "bd_araihazar2012_kit" / "Well_Water_Arsenic_Concentration.xlsx").exists()
CAMPAIGN = ["split", "block", "policy", "learner", "seed", "thr", "radius", "thin", "advise", "known", "lam",
            "daily_return", "absent", "cap", "impute"]
LEARN = "learning|gp"
CONFIRM = {"0_0", "0_1", "1_0", "1_1", "3_2"}
SHORT = {"sweep|-": "Door-to-door", "static|-": "Static plan", "geometry|-": "Re-planning", "uncertainty|gp": "Uncertainty",
         "adaptive|-": "Adaptive", "oracle|-": "Oracle", "learning|rf": "Random forest", "learning|lgbm": "LightGBM",
         "learning|knn": "kNN"}
checks = []


# ---------------------------------------------------------------- comparison helpers
def decimals(s):
    s = s.strip().lstrip("+")
    return len(s.split(".")[1]) if "." in s else 0


def check(name, printed, value, mc_se=0.0):
    """printed: the string in the manuscript; value: the recomputed number (or None if it cannot be recomputed)."""
    printed = str(printed).strip()
    if value is None:
        status = "SKIPPED"
    elif printed in ("-", "nan"):
        status = "OK" if (value is None or (isinstance(value, float) and math.isnan(value))) else "MISMATCH"
    elif printed.startswith("<"):
        status = "OK" if value < float(printed[1:]) else "MISMATCH"
    else:
        tol = 0.5 * 10 ** -decimals(printed) + 1e-9 + 3 * mc_se
        status = "OK" if abs(float(printed.replace("+", "")) - float(value)) <= tol else "MISMATCH"
    checks.append(dict(check=name, printed=printed, recomputed=None if value is None else round(float(value), 6), status=status))
    return status


def ci_parts(cell):
    m = re.match(r"\s*([-+\d.]+)\s*\(\s*([-+\d.]+)\s*(?:,|to)\s*([-+\d.]+)\s*\)", cell)
    return m.groups() if m else (None, None, None)


def md_table(text, caption_start):
    """Rows of the markdown table whose caption line starts with caption_start; blank cells repeat the cell above."""
    lines = text.split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith(caption_start))
    while not lines[i].startswith("|"):
        i += 1
    rows = []
    while i < len(lines) and lines[i].startswith("|"):
        if not re.match(r"^\|[\s\-|:]+\|$", lines[i]):
            rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
        i += 1
    head, body = rows[0], rows[1:]
    for r_prev, r in zip(body, body[1:]):
        for j, c in enumerate(r):
            if c == "" and j < 3:
                r[j] = r_prev[j]
    return head, body


# ---------------------------------------------------------------- independent recomputation
def load(tag):
    d = pd.read_csv(RES / f"f1_{tag}.csv")
    d["arm"] = d.policy + "|" + d.learner
    return d


def at_effort(df, col="protected_share", per1000=5.0):
    keys = [c for c in CAMPAIGN if c in df.columns]
    out = []
    for _, g in df.groupby(keys, dropna=False):
        g = g.sort_values("days")
        target = min(per1000 * g["n"].iloc[0] / 1000.0, g["days"].max())
        val = np.interp(target, np.r_[0.0, g["days"].to_numpy()], np.r_[0.0, g[col].to_numpy()])
        out.append(dict(block=g["block"].iloc[0], arm=g["arm"].iloc[0], thr=g["thr"].iloc[0], radius=g["radius"].iloc[0],
                        thin=g["thin"].iloc[0] if "thin" in g else 1.0, split=g["split"].iloc[0], seed=g["seed"].iloc[0],
                        value=val, exposed=g["exposed"].iloc[0], n=g["n"].iloc[0], days=target))
    return pd.DataFrame(out)


def at_day(df, day, col="protected_share"):
    g = df[df.days == day]
    return pd.DataFrame(dict(block=g.block, arm=g.arm, thr=g.thr, radius=g.radius, thin=g.get("thin", 1.0), split=g.split,
                             seed=g.seed, value=g[col], exposed=g.exposed, n=g.n))


def paired(e, a, b, blocks=None):
    m = e.groupby(["block", "arm"]).agg(value=("value", "mean"), exposed=("exposed", "mean")).reset_index()
    if blocks is not None:
        m = m[m.block.isin(blocks)]
    va = m[m.arm == a].set_index("block")
    vb = m[m.arm == b].set_index("block").reindex(va.index)
    if vb.value.isna().all():
        return None
    d = 100 * (va.value - vb.value)
    if d.std(ddof=1) == 0:
        lo = hi = d.mean(); p = float("nan")
    else:
        lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
        p = stats.ttest_rel(va.value, vb.value).pvalue
    pooled = lambda v: (v.value * va.exposed).sum() / va.exposed.sum()
    return dict(a=100 * va.value.mean(), b=100 * vb.value.mean(), diff=d.mean(), lo=lo, hi=hi, p=p,
                pos=int((d > 0).sum()), n=len(d), rel=100 * (pooled(va) / pooled(vb) - 1), d=d)


def compare_row(name, cells, r, cols):
    """cols: mapping from column name to the recomputed key ('a', 'b', 'ci', 'p', 'blocks', 'rel')."""
    for col, key in cols.items():
        cell = cells[col]
        if key == "ci":
            dd, lo, hi = ci_parts(cell)
            check(f"{name}: difference", dd, r["diff"]); check(f"{name}: CI low", lo, r["lo"]); check(f"{name}: CI high", hi, r["hi"])
        elif key == "blocks":
            pos, n = cell.split("/")
            check(f"{name}: blocks positive", pos, r["pos"]); check(f"{name}: blocks", n, r["n"])
        else:
            check(f"{name}: {col}", cell, r[key])


def table2(md, eff):
    head, body = md_table(md, "**Table 2.**")
    c = {h: i for i, h in enumerate(head)}
    groups = [("census", None), ("census", CONFIRM), ("upazila", None)]
    prim = []
    for k, row in enumerate(body):
        survey, blocks = groups[k // 3]
        comp = {"Door-to-door sweep": "sweep|-", "Static depth plan": "static|-", "Adaptive": "adaptive|-"}[row[c["Comparator"]]]
        r = paired(eff[survey], LEARN, comp, blocks)
        name = f"Table 2 {survey}{' confirmatory' if blocks else ''} vs {comp}"
        cells = dict(zip(head, row))
        compare_row(name, cells, r, {"Comparator (%)": "b", "Learning (%)": "a", "Difference (95% CI)": "ci", "Blocks": "blocks"})
        check(f"{name}: relative gain", cells["Relative gain"].replace("%", ""), r["rel"])
        if blocks:
            check(f"{name}: unadjusted p", cells["*p*"], r["p"])
        else:
            prim.append((name, cells["*p*"], r["p"]))
    adj = multipletests([x[2] for x in prim], method="holm")[1]
    for (name, printed, _), h in zip(prim, adj):
        check(f"{name}: Holm p", printed, h)
    return adj


# ---------------------------------------------------------------- Table A1 (all comparisons)
def label_maps():
    sys.path.insert(0, str(ROOT / "manuscript"))
    from make_appendix import ANALYSIS
    return ANALYSIS


def overlap_blocks():
    if not DATA_OK:
        return None
    sys.path.insert(0, str(HERE))
    from scipy.spatial import cKDTree
    from data_ext import load_bd2012
    u = load_bd2012()
    c = pd.read_csv(ROOT / "data" / "raw" / "araihazar" / "Dataset2.csv").dropna(subset=["Latitude", "Longitude"])
    lat0, lon0 = u.lat.mean(), u.lon.mean()
    proj = lambda la, lo: np.c_[(lo - lon0) * 111_320 * np.cos(np.radians(lat0)), (la - lat0) * 110_574]
    dd, _ = cKDTree(proj(c.Latitude.values, c.Longitude.values)).query(proj(u.lat.values, u.lon.values))
    u["near"] = dd <= 100
    cnt = u.groupby("block").near.sum()
    return set(cnt[cnt >= 20].index)


def table_a1(app):
    ANALYSIS = label_maps()
    head, body = md_table(app, "**Table A1.**")
    ov = overlap_blocks()
    bd = load("v6_bd2012")
    sets = {"2000-01 census": load("v6_main"), "2012-13 upazila survey": bd,
            "2012-13, blocks outside the census area": None if ov is None else bd[~bd.block.isin(ov)]}
    for tag, lab in (("v6_field", "field realism (daily return, 15% absent)"), ("v6_advise", "advise if kit ≤ ½ threshold"),
                     ("v6_radius", "radius"), ("v6_nokit", "no kit error"), ("v6_thin", "thinned"), ("v6_t10", "10 min per test"),
                     ("v6_t29", "about 12 tests per day (29 min per test)"), ("v6_bd2012_t29", "2012-13, about 12 tests per day"),
                     ("v6_bd2012_old", "2012-13 with 2000-01 survey")):
        if (RES / f"f1_{tag}.csv").exists():
            sets[lab] = load(tag)
    inv = {v: k for k, v in ANALYSIS.items()}
    short_inv = {v: k for k, v in SHORT.items()}
    cache = {}
    for row in body:
        cells = dict(zip(head, row))
        setting = inv.get(cells["Analysis"], cells["Analysis"])
        outcome = None
        for o in ("prot_cap_opt_share", "prot_cap_share", "switch_share"):
            if setting.endswith("; outcome " + o):
                setting, outcome = setting.replace("; outcome " + o, ""), o
        d = sets.get(setting)
        name = f"Table A1 {cells['Analysis']} | {cells['Setting']} | {cells['Effort']} | {cells['Comparator']}"
        if d is None:
            check(name, cells["Learning (%)"], None); continue
        m = re.match(r"(\d+) µg/L, (\d+) m(?:, (\d+)% of wells)?", cells["Setting"])
        thr, radius, thin = float(m.group(1)), float(m.group(2)), (float(m.group(3)) / 100 if m.group(3) else 1.0)
        sub = d[(d.thr == thr) & (d.radius == radius) & ((d.thin == thin) if "thin" in d else True)]
        key = (setting, thr, radius, thin, cells["Effort"], outcome)
        if key not in cache:
            col = outcome or "protected_share"
            cache[key] = at_effort(sub, col) if cells["Effort"] == "primary" else at_day(sub, float(cells["Effort"].split()[0]), col)
        r = paired(cache[key], LEARN, short_inv[cells["Comparator"]])
        compare_row(name, cells, r, {"Learning (%)": "a", "Comparator (%)": "b", "Difference (95% CI)": "ci", "*p*": "p",
                                     "Blocks": "blocks", "Relative gain (%)": "rel"})


def table_a2(app):
    head, body = md_table(app, "**Table A2.**")
    d = load("v6_learners")
    short_inv = {v: k for k, v in SHORT.items()}
    for row in body:
        cells = dict(zip(head, row))
        e = at_effort(d[d.thr == float(cells["Threshold (µg/L)"])])
        r = paired(e, LEARN, short_inv[cells["Learner compared"]])
        compare_row(f"Table A2 {cells['Threshold (µg/L)']} vs {cells['Learner compared']}", cells, r,
                    {"Gaussian process (%)": "a", "Other learner (%)": "b", "Difference (95% CI)": "ci", "*p*": "p", "Blocks": "blocks"})


def table_a3(app):
    head, body = md_table(app, "**Table A3.**")
    data = {"2000-01 census": load("v6_main"), "2012-13 upazila survey": load("v6_bd2012")}
    tert = {"fewest safe": 0, "middle": 1, "most safe": 2}
    short_inv = {v: k for k, v in SHORT.items()}
    for row in body:
        cells = dict(zip(head, row))
        d = data[cells["Survey"]]
        e = at_day(d[d.thr == float(cells["Threshold (µg/L)"])], 10, f"prot_t{tert[cells['Tertile']]}_share")
        r = paired(e, LEARN, short_inv[cells["Comparator"]])
        compare_row(f"Table A3 {cells['Survey']} {cells['Threshold (µg/L)']} {cells['Tertile']} vs {cells['Comparator']}",
                    cells, r, {"Learning (%)": "a", "Comparator (%)": "b", "Difference (95% CI)": "ci", "*p*": "p", "Blocks": "blocks"})


def sign_flip_p(d, n_mc=200_000):
    d = np.asarray(d, float)
    if len(d) <= 16:
        res = stats.permutation_test((d,), np.mean, permutation_type="samples", n_resamples=np.inf, alternative="two-sided")
        return res.pvalue, 0.0
    res = stats.permutation_test((d,), np.mean, permutation_type="samples", n_resamples=n_mc, alternative="two-sided",
                                 random_state=12345)
    return res.pvalue, math.sqrt(max(res.pvalue * (1 - res.pvalue), 1e-12) / n_mc)


def table_a5(app, eff):
    head, body = md_table(app, "**Table A5.**")
    comps = {"Door-to-door": "sweep|-", "Static plan": "static|-", "Adaptive": "adaptive|-"}
    surveys = {"2000-01 census": "census", "2012-13 upazila survey": "upazila"}
    t_p, perm, perm_se, wil, rows = [], [], [], [], []
    for row in body:
        cells = dict(zip(head, row))
        e = eff[surveys[cells["Survey"]]]
        r = paired(e, LEARN, comps[cells["Comparator"]])
        p, se = sign_flip_p(r["d"])
        t_p.append(r["p"]); perm.append(p); perm_se.append(se); wil.append(stats.wilcoxon(r["d"]).pvalue)
        rows.append((cells, r, e))
    holm = [multipletests(x, method="holm")[1] for x in (t_p, perm, wil)]
    for k, (cells, r, e) in enumerate(rows):
        name = f"Table A5 {cells['Survey']} vs {cells['Comparator']}"
        check(f"{name}: blocks", cells["Blocks"], r["n"])
        check(f"{name}: median difference", cells["Median difference"], float(np.median(r["d"])))
        pt, pp, pw = [s.strip() for s in cells["Holm *p*: *t*; permutation; Wilcoxon"].split(";")]
        check(f"{name}: Holm p t-test", pt, holm[0][k])
        check(f"{name}: Holm p permutation", pp, holm[1][k], mc_se=perm_se[k] * 6)
        check(f"{name}: Holm p Wilcoxon", pw, holm[2][k])
        lo, hi = cells["Leave one block out"].split(" to ")
        loo = [r["d"].drop(b).mean() for b in r["d"].index]
        check(f"{name}: leave-one-out min", lo, min(loo)); check(f"{name}: leave-one-out max", hi, max(loo))
        sb = cells["Super-blocks (95% CI); *p*"]
        if sb != "-":
            d_sb = r["d"].groupby(r["d"].index.map(lambda u: f"{int(u.split('_')[0]) // 2}_{int(u.split('_')[1]) // 2}")).mean()
            dd, l2, h2 = ci_parts(sb)
            lo2, hi2 = stats.t.interval(0.95, len(d_sb) - 1, loc=d_sb.mean(), scale=stats.sem(d_sb))
            check(f"{name}: super-blocks difference", dd, d_sb.mean()); check(f"{name}: super-blocks CI low", l2, lo2)
            check(f"{name}: super-blocks CI high", h2, hi2)
            p_sb, se_sb = sign_flip_p(d_sb.to_numpy())
            check(f"{name}: super-blocks permutation p", sb.split(";")[-1], p_sb, mc_se=se_sb)
        cf = cells["Confirmatory (95% CI); *p*"]
        if cf != "-":
            rc = paired(e, LEARN, comps[cells["Comparator"]], CONFIRM)
            dd, l2, h2 = ci_parts(cf)
            check(f"{name}: confirmatory difference", dd, rc["diff"]); check(f"{name}: confirmatory CI low", l2, rc["lo"])
            check(f"{name}: confirmatory CI high", h2, rc["hi"])
            check(f"{name}: confirmatory permutation p", cf.split(";")[-1], sign_flip_p(rc["d"].to_numpy())[0])
        # Monte Carlo SE of block means and SD between blocks
        runs = e.pivot_table(index=["block", "split", "seed"], columns="arm", values="value")
        rd = 100 * (runs[LEARN] - runs[comps[cells["Comparator"]]]).dropna()
        g = rd.groupby(level="block")
        mc, sd = (g.std(ddof=1) / np.sqrt(g.size())).mean(), g.mean().std(ddof=1)
        a, b = cells["MC SE / SD between blocks"].split(" / ")
        check(f"{name}: Monte Carlo SE", a, mc); check(f"{name}: SD between blocks", b, sd)
    return holm


def cost_table(tag):
    d = load(tag)
    d = d[d.thr == 50]
    e = at_effort(d); t = at_effort(d, "tests")
    e["tests"] = t["value"].to_numpy()
    out = {}
    for arm, g in e.groupby("arm"):
        persons = (g.value * g.exposed).sum() * 8
        days = g.days.sum()
        out[arm] = dict(tpd=g.tests.sum() / days, time=days * 28.80 / persons,
                        mixed=(days * 12 * (2.40 - 0.17) + g.tests.sum() * 0.17) / persons, per_test=g.tests.sum() * 2.40 / persons)
    return out


def table_a6(app):
    head, body = md_table(app, "**Table A6.**")
    tags = {"2000-01 census": "v6_main", "2012-13 upazila survey": "v6_bd2012", "Census, field realism": "v6_field",
            "Census, about 12 tests per day": "v6_t29", "2012-13, about 12 tests per day": "v6_bd2012_t29"}
    pol = {"Door-to-door": "sweep|-", "Static plan": "static|-", "Adaptive": "adaptive|-", "Learning": "learning|gp"}
    cache = {}
    for row in body:
        cells = dict(zip(head, row))
        tag = tags[cells["Survey"]]
        cache.setdefault(tag, cost_table(tag))
        c = cache[tag][pol[cells["Policy"]]]
        name = f"Table A6 {cells['Survey']} {cells['Policy']}"
        check(f"{name}: tests per team-day", cells["Tests per team-day"], c["tpd"])
        check(f"{name}: time-based", cells["Time-based (US$)"], c["time"])
        check(f"{name}: time plus kit", cells["Time plus kit (US$)"], c["mixed"])
        check(f"{name}: per test", cells["Per test (US$)"], c["per_test"])
        for k in (5, 10):
            col = f"Time-based, {k} persons per well (US$)"
            if col in cells:
                check(f"{name}: time-based {k} persons", cells[col], c["time"] * 8 / k)
    return cache


def table_a7(app):
    head, body = md_table(app, "**Table A7.**")
    tags = {("2000-01 census", "kit reading 0 set to 5 µg/L"): ("v6_main", "v6_zero5"),
            ("2012-13 upazila survey", "kit reading 0 set to 5 µg/L"): ("v6_bd2012", "v6_bd2012_zero5"),
            ("2000-01 census", "kit-error variance added to nugget"): ("v6_main", "v6_kitnug"),
            ("2000-01 census", "600 m exclusion buffer around evaluated wells"): ("v6_main", "v6_buffer"),
            ("2012-13 upazila survey", "600 m exclusion buffer around evaluated wells"): ("v6_bd2012", "v6_bd2012_buffer")}
    short_inv = {v: k for k, v in SHORT.items()}
    cache = {}
    changes = {"level": [], "gain": []}
    for row in body:
        cells = dict(zip(head, row))
        base, tag = tags[(cells["Survey"], cells["Analysis"])]
        thr = float(cells["Threshold (µg/L)"])
        for t in (base, tag):
            if (t, thr) not in cache:
                d = load(t)
                cache[(t, thr)] = at_effort(d[d.thr == thr])
        rs = paired(cache[(tag, thr)], LEARN, short_inv[cells["Comparator"]])
        rb = paired(cache[(base, thr)], LEARN, short_inv[cells["Comparator"]])
        name = f"Table A7 {cells['Survey']} | {cells['Analysis']} | {cells['Threshold (µg/L)']} | {cells['Comparator']}"
        check(f"{name}: learning, main analysis", cells["Learning, main analysis (%)"], rb["a"])
        compare_row(name, cells, rs, {"Learning (%)": "a", "Comparator (%)": "b", "Difference (95% CI)": "ci", "*p*": "p",
                                      "Blocks": "blocks"})
        if thr == 50:
            changes["level"].append(abs(rs["a"] - rb["a"])); changes["gain"].append(abs(rs["diff"] - rb["diff"]))
    return changes


def table_a8(app):
    head, body = md_table(app, "**Table A8.**")
    d = load("v6_bd2012_boot")
    d = d[d.thr == 50]
    reps = {}
    for (split, seed), g in d.groupby(["split", "seed"]):
        e = at_effort(g)
        m = e.groupby(["block", "arm"]).value.mean().unstack()
        r = {"learning protected (%)": 100 * m[LEARN].mean()}
        for c, lab in (("sweep|-", "door-to-door"), ("static|-", "static plan"), ("adaptive|-", "adaptive")):
            pr = paired(e, LEARN, c)
            r[f"gain vs {lab} (pp)"] = pr["diff"]; r[f"relative gain vs {lab} (%)"] = pr["rel"]
        reps[(split, seed)] = r
    R = pd.DataFrame(reps.values())
    for row in body:
        cells = dict(zip(head, row))
        q = cells["Quantity"]
        if q not in R:
            check(f"Table A8 {q}", cells["Mean"], None); continue
        v = R[q].to_numpy()
        check(f"Table A8 {q}: replicates", cells["Replicates"], len(v))
        check(f"Table A8 {q}: mean", cells["Mean"], v.mean())
        lo, hi = cells["2.5-97.5 percentile"].split(" to ")
        check(f"Table A8 {q}: 2.5 percentile", lo, np.percentile(v, 2.5)); check(f"Table A8 {q}: 97.5 percentile", hi, np.percentile(v, 97.5))
        check(f"Table A8 {q}: replicates positive", cells["Replicates positive"], int((v > 0).sum()))


def table_a10(app):
    head, body = md_table(app, "**Table A10.**")
    if not DATA_OK:
        for row in body:
            check(f"Table A10 kit {row[0]}", row[1], None)
        return
    sys.path.insert(0, str(HERE))
    from data import load_kit
    k = load_kit()
    for row in body:
        cells = dict(zip(head, row))
        g = k[k.kit == float(cells["Kit reading (µg/L)"])].lab
        name = f"Table A10 kit {cells['Kit reading (µg/L)']}"
        check(f"{name}: pairs", cells["Pairs"], len(g)); check(f"{name}: median", cells["Median laboratory value (µg/L)"], g.median())
        check(f"{name}: safe 10", cells["Laboratory ≤10 µg/L (%)"], 100 * (g <= 10).mean())
        check(f"{name}: safe 50", cells["Laboratory ≤50 µg/L (%)"], 100 * (g <= 50).mean())


def text_claims(md, eff, holm_primary, holm_a5, a7):
    abstract = md[md.index("# Abstract"): md.index("**Keywords")]
    rel = {(k, c): paired(eff[k], LEARN, c)["rel"] for k in eff for c in ("sweep|-", "static|-")}
    check("Abstract: 61% vs door-to-door (census)", re.search(r"protected (\d+)% \(census\)", abstract).group(1), rel[("census", "sweep|-")])
    check("Abstract: 53% vs door-to-door (upazila)", re.search(r"and (\d+)% \(upazila survey\)", abstract).group(1), rel[("upazila", "sweep|-")])
    lo, hi = re.search(r"and (\d+)-(\d+)% more than a depth-based", abstract).groups()
    check("Abstract: lower relative gain vs static", lo, min(rel[("census", "static|-")], rel[("upazila", "static|-")]))
    check("Abstract: upper relative gain vs static", hi, max(rel[("census", "static|-")], rel[("upazila", "static|-")]))
    check("Abstract/results: largest Holm p (printed bound)", re.search(r"Holm-adjusted p ≤ ([\d.]+)", abstract).group(1), max(holm_primary))
    cm, cu = cost_table("v6_main"), cost_table("v6_bd2012")
    m = re.search(r"US\$(\d+\.\d+)-(\d+\.\d+) to US\$(\d+\.\d+)-(\d+\.\d+)", abstract)
    for printed, v in zip(m.groups(), (cm["sweep|-"]["time"], cu["sweep|-"]["time"], cm["learning|gp"]["time"], cu["learning|gp"]["time"])):
        check("Abstract: cost per person protected", printed, v)
    allp = max(np.max(holm_a5[1]), np.max(holm_a5[2]))
    check("Results: permutation and Wilcoxon Holm p bound", re.search(r"Permutation and Wilcoxon tests agreed \(Holm-adjusted \*p\* ≤ ([\d.]+)\)", md).group(1), allp)
    if a7["level"]:
        check("Results: largest change in learning protection (Table A7)", re.search(r"protection at the primary effort by at most ([\d.]+) points", md).group(1), max(a7["level"]))
        check("Results: largest change in gains (Table A7)", re.search(r"gains over the three comparators by at most ([\d.]+) points", md).group(1), max(a7["gain"]))


def main():
    md = (ROOT / "manuscript" / "manuscript.md").read_text(encoding="utf-8")
    app = (ROOT / "manuscript" / "appendix.md").read_text(encoding="utf-8")
    eff = {"census": at_effort(load("v6_main").query("thr == 50")), "upazila": at_effort(load("v6_bd2012").query("thr == 50"))}
    holm_primary = table2(md, eff)
    table_a1(app); table_a2(app); table_a3(app)
    holm_a5 = table_a5(app, eff)
    table_a6(app)
    a7 = table_a7(app)
    table_a8(app); table_a10(app)
    text_claims(md, eff, holm_primary, holm_a5, a7)
    A = pd.DataFrame(checks)
    A.to_csv(RES / "f1_v6_audit.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
    bad = A[A.status == "MISMATCH"]
    print(A.status.value_counts().to_string())
    if len(bad):
        print("\nMISMATCHES:\n" + bad.to_string())
        sys.exit(1)
    print("\nAll recomputed values match the manuscript.")


if __name__ == "__main__":
    main()
