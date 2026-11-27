"""Tables and figures for the v6 analysis.   python -X utf8 src/f1/figures6.py

Primary endpoint (declared in docs/f1_final_config.md before the runs): share of exposed households protected after
5 team-days per 1,000 wells (interpolated per campaign), 50 µg/L, 100 m; learning (GP) vs door-to-door sweep, static
plan and adaptive, in both surveys; Holm-adjusted p over these 6 comparisons. Units are blocks; each block's value is
the mean over runs and random splits.
"""
import pathlib, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import cKDTree
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import ROOT

RES, FIG = ROOT / "results", ROOT / "manuscript" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})
HEAD = "learning:gp"
ARMS = {"sweep:-": ("Door-to-door sweep", "#9aa3ad"), "static:-": ("Static depth plan", "#7a9cc0"),
        "geometry:-": ("Re-planning only", "#b8a9d9"), "uncertainty:gp": ("Uncertainty sampling", "#d9a3b8"),
        "adaptive:-": ("Adaptive (depth + results)", "#4a7fb0"), "learning:knn": ("Learning: kNN", "#e0a458"),
        "learning:rf": ("Learning: random forest", "#5a9a6e"), "learning:lgbm": ("Learning: LightGBM", "#8fbf8f"),
        "learning:gp": ("Learning: Gaussian process", "#1d6a3a"), "oracle:-": ("Oracle (perfect knowledge)", "#222222")}
KEYS = ["area", "split", "block", "policy", "learner", "seed", "radius", "thr", "advise", "known", "lam",
        "daily_return", "absent", "cap", "thin", "impute"]
EFFORT = 5.0     # team-days per 1,000 wells


def load(tag):
    d = pd.read_csv(RES / f"f1_{tag}.csv")
    d["arm"] = d.policy + ":" + d.learner
    d["unit"] = d.area + "/" + d.block
    return d


def effort(d, metric="protected_share", per1000=EFFORT):
    """Interpolate `metric` at days = per1000 * n / 1000 for every campaign."""
    rows = []
    for k, g in d.groupby(KEYS + ["arm", "unit"], dropna=False):
        g = g.sort_values("days")
        x = np.r_[0, g.days.values]
        y = np.r_[0, g[metric].values]
        target = min(per1000 * g.n.iat[0] / 1000, x.max())
        r = dict(zip(KEYS + ["arm", "unit"], k))
        r[metric] = float(np.interp(target, x, y))
        r["exposed"] = g.exposed.iat[0]
        r["n"] = g.n.iat[0]
        rows.append(r)
    return pd.DataFrame(rows)


def unit_means(d, metric):
    return d.groupby(["unit", "arm"])[[metric, "exposed"]].mean().reset_index()


def compare(d, a, b, metric="protected_share"):
    u = unit_means(d, metric).pivot(index="unit", columns="arm", values=metric)
    if a not in u or b not in u:
        return None
    diff = (u[a] - u[b]).dropna()
    m, s, n = diff.mean(), diff.std(ddof=1), len(diff)
    half = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    p = stats.ttest_1samp(diff, 0).pvalue if n > 1 else np.nan
    w = unit_means(d, metric).pivot(index="unit", columns="arm", values="exposed")[a].reindex(diff.index)
    pooled_a = (u[a] * w).sum() / w.sum()
    pooled_b = (u[b] * w).sum() / w.sum()
    return dict(a_pct=100 * u[a].mean(), b_pct=100 * u[b].mean(), diff=100 * m, lo=100 * (m - half),
                hi=100 * (m + half), p=p, units_positive=f"{int((diff > 0).sum())}/{n}",
                pooled_a=100 * pooled_a, pooled_b=100 * pooled_b, rel_pooled=100 * (pooled_a / pooled_b - 1))


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def overlap_units():
    """2012-13 blocks overlapping the 2000-01 census: at least 20 wells within 100 m of a census well."""
    from data_ext import load_bd2012
    u = load_bd2012()
    c = pd.read_csv(ROOT / "data" / "raw" / "araihazar" / "Dataset2.csv").dropna(subset=["Latitude", "Longitude"])
    lat0, lon0 = u.lat.mean(), u.lon.mean()
    proj = lambda la, lo: np.c_[(lo - lon0) * 111_320 * np.cos(np.radians(lat0)), (la - lat0) * 110_574]
    tree = cKDTree(proj(c.Latitude.values, c.Longitude.values))
    dd, _ = tree.query(proj(u.lat.values, u.lon.values))
    u["near"] = dd <= 100
    cnt = u.groupby("block").near.sum()          # declared definition: >= 20 wells within 100 m of a census well
    return set("bd2012/" + cnt[cnt >= 20].index)


def curves(d, fname, title, unit="block"):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for ax, thr in zip(axes, (50, 10)):
        s = d[(d.thr == thr) & (d.days <= 20)]
        for arm in ("sweep:-", "static:-", "adaptive:-", HEAD, "oracle:-"):
            g = s[s.arm == arm]
            if g.empty:
                continue
            u = g.groupby(["days", "unit"]).protected_share.mean().reset_index()
            m = u.groupby("days").protected_share.agg(["mean", "std", "count"])
            half = stats.t.ppf(0.975, m["count"] - 1) * m["std"] / np.sqrt(m["count"])
            lab, col = ARMS[arm]
            ax.plot(m.index, 100 * m["mean"], color=col, lw=2.2 if arm == HEAD else 1.1,
                    ls="--" if arm == "oracle:-" else "-", label=lab)
            if arm in (HEAD, "sweep:-", "static:-"):
                ax.fill_between(m.index, 100 * (m["mean"] - half), 100 * (m["mean"] + half), color=col, alpha=0.15, lw=0)
        ax.set_title(f"{title}{thr} µg/L, 100 m", fontsize=8.5)
        ax.set_xlabel(f"Field-team days per {unit}")
        ax.set_ylabel("Exposed households protected (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(FIG / fname); plt.close(fig)


def decomposition(e_main, e_bd):
    order = ["sweep:-", "static:-", "geometry:-", "uncertainty:gp", "adaptive:-", HEAD, "oracle:-"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, (e, title) in zip(axes, ((e_main, "2000–01 census"), (e_bd, "2012–13 upazila survey"))):
        s = unit_means(e[e.thr == 50], "protected_share").groupby("arm").protected_share.mean().reindex(order) * 100
        ax.barh(range(len(order)), s.values, color=[ARMS[a][1] for a in order])
        ax.set_yticks(range(len(order)), [ARMS[a][0] for a in order], fontsize=7)
        for i, v in enumerate(s.values):
            ax.text(v + 0.4, i, f"{v:.1f}", va="center", fontsize=7)
        ax.invert_yaxis(); ax.set_title(f"{title}, 50 µg/L", fontsize=8.5)
        ax.set_xlabel("Protected (%), 5 team-days/1,000 wells", fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIG / "fig3_decomposition.png"); plt.close(fig)


def harm(main, adv):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9))
    for ax, thr in zip(axes, (50, 10)):
        for d, mk, rule in ((main, "o", "kit ≤ threshold"), (adv, "s", "kit ≤ ½ threshold")):
            s = d[(d.thr == thr) & (d.days == 10)]
            for arm in ("sweep:-", "static:-", "adaptive:-", HEAD):
                g = s[s.arm == arm]
                if len(g):
                    mis = 100 * (g.misled_own_share + g.misled_src_share).mean()
                    ax.scatter(mis, 100 * g.protected_share.mean(), color=ARMS[arm][1], marker=mk, s=38,
                               label=f"{ARMS[arm][0]}; {rule}")
        ax.set_xlabel("Exposed households misled (%)"); ax.set_ylabel("Exposed households protected (%)")
        ax.set_title(f"2000–01 census, {thr} µg/L, 10 days", fontsize=8.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.3, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.24, 1, 1)); fig.savefig(FIG / "fig6_harm.png"); plt.close(fig)


def forest(rows):
    r = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.0, 0.28 * len(r) + 1.0))
    for i, x in enumerate(r.itertuples()):
        ax.errorbar(x.diff, i, xerr=[[x.diff - x.lo], [x.hi - x.diff]], fmt="o", ms=3.5, capsize=2,
                    color="#1d6a3a" if "sweep" in x.comparator else "#4a7fb0")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(range(len(r)), [f"{x.setting} — vs {x.comparator}" for x in r.itertuples()], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Learning minus comparator (percentage points; 95% CI)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig5_robustness.png"); plt.close(fig)


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    main, bd = load("v6_main"), load("v6_bd2012")
    e_main, e_bd = effort(main), effort(bd)
    # ---- primary family
    prim = []
    for name, e in (("2000–01 census", e_main), ("2012–13 upazila survey", e_bd)):
        for b in ("sweep:-", "static:-", "adaptive:-"):
            prim.append(dict(survey=name, comparator=ARMS[b][0], **compare(e[e.thr == 50], HEAD, b)))
    P = pd.DataFrame(prim); P["p_holm"] = holm(P.p)
    P.to_csv(RES / "f1_v6_primary.csv", index=False)
    print("PRIMARY (5 team-days per 1,000 wells, 50 µg/L)\n", P.round(4).to_string())
    # ---- secondary table S1
    rows = []
    sets = [("2000–01 census", main), ("2012–13 upazila survey", bd)]
    ov = overlap_units()
    sets.append(("2012–13, blocks outside the census area", bd[~bd.unit.isin(ov)]))
    for tag, lab in (("v6_field", "field realism (daily return, 15% absent)"), ("v6_advise", "advise if kit ≤ ½ threshold"),
                     ("v6_radius", "radius"), ("v6_nokit", "no kit error"), ("v6_thin", "thinned"),
                     ("v6_t10", "10 min per test"), ("v6_bd2012_old", "2012–13 with 2000–01 survey")):
        try:
            sets.append((lab, load(tag)))
        except FileNotFoundError:
            pass
    for lab, d in sets:
        e = effort(d)
        for keys, g in e.groupby(["thr", "radius", "thin"]):
            for b in ("sweep:-", "static:-", "geometry:-", "uncertainty:gp", "adaptive:-", "oracle:-"):
                r = compare(g, HEAD, b)
                if r:
                    rows.append(dict(setting=lab, thr=keys[0], radius=keys[1], thin=keys[2], horizon="5 d/1000 wells",
                                     comparator=b, **r))
        for days in (5, 10, 20):
            for keys, g in d[d.days == days].groupby(["thr", "radius", "thin"]):
                for b in ("sweep:-", "static:-", "adaptive:-"):
                    r = compare(g, HEAD, b)
                    if r:
                        rows.append(dict(setting=lab, thr=keys[0], radius=keys[1], thin=keys[2], horizon=f"{days} days",
                                         comparator=b, **r))
    # other outcomes (main + bd, 10 days)
    for lab, d in (("2000–01 census", main), ("2012–13 upazila survey", bd)):
        for metric in ("prot_cap_share", "switch_share"):
            for thr in (50, 10):
                for b in ("sweep:-", "static:-", "adaptive:-"):
                    r = compare(d[(d.thr == thr) & (d.days == 10)], HEAD, b, metric)
                    if r:
                        rows.append(dict(setting=f"{lab}; outcome {metric}", thr=thr, radius=100, thin=1,
                                         horizon="10 days", comparator=b, **r))
    S1 = pd.DataFrame(rows)
    S1_out = S1.assign(comparator=S1.comparator.map(lambda c: ARMS.get(c, (c,))[0]))
    S1_out.to_csv(RES / "f1_v6_table_S1.csv", index=False)
    print("overlap blocks", len(ov))
    # ---- learners
    try:
        L = load("v6_learners"); eL = effort(L)
        S2 = pd.DataFrame([dict(thr=t, comparator=b, **compare(eL[eL.thr == t], HEAD, b))
                           for t in (50, 10) for b in ("learning:rf", "learning:lgbm", "learning:knn")])
        S2.assign(comparator=S2.comparator.map(lambda c: ARMS[c][0])).to_csv(RES / "f1_v6_table_S2.csv", index=False); print("LEARNERS\n", S2.round(3).to_string())
    except FileNotFoundError:
        pass
    # ---- equity
    eq = []
    for lab, d in (("2000–01 census", main), ("2012–13 upazila survey", bd)):
        for thr in (50, 10):
            s = d[(d.thr == thr) & (d.days == 10)]
            for t in range(3):
                for b in ("sweep:-", "static:-"):
                    r = compare(s, HEAD, b, f"prot_t{t}_share")
                    if r:
                        eq.append(dict(survey=lab, thr=thr, tertile=["fewest safe", "middle", "most safe"][t], comparator=b, **r))
    S3 = pd.DataFrame(eq); S3.assign(comparator=S3.comparator.map(lambda c: ARMS[c][0])).to_csv(RES / "f1_v6_table_S3.csv", index=False); print("EQUITY\n", S3.round(2).to_string())
    # ---- harm, misled split, false alarms, tests/day, cost
    for lab, d in (("census", main), ("upazila", bd)):
        s = d[(d.thr == 50) & (d.days == 10)]
        print(lab, "harm @10d 50:", s.groupby("arm")[["protected_share", "misled_own_share", "misled_src_share",
                                                      "false_alarm_share"]].mean().mul(100).round(2).to_string())
        print(lab, "tests/day:", (d[d.days == 10].groupby("arm").tests.mean() / 10).round(1).to_dict())
        e = effort(d)[lambda x: x.thr == 50]
        cost = e.groupby("arm").apply(lambda g: (EFFORT * g.n / 1000 * 28.80).sum() / ((g.protected_share * g.exposed).sum() * 8))
        print(lab, "USD per person protected (primary effort):", cost.round(2).to_dict())
    try:
        adv = load("v6_advise")
        s = adv[(adv.thr == 50) & (adv.days == 10)]
        print("advise:", s.groupby("arm")[["protected_share", "misled_own_share", "misled_src_share"]].mean().mul(100).round(2).to_string())
    except FileNotFoundError:
        adv = None
    # ---- figures
    curves(main, "fig2_protected.png", "2000–01 census, ")
    curves(bd, "fig4_upazila.png", "2012–13 upazila survey, ")
    decomposition(e_main, e_bd)
    if adv is not None:
        harm(main, adv)
    fr = []
    for lab in ["2000–01 census", "2012–13 upazila survey", "2012–13, blocks outside the census area",
                "field realism (daily return, 15% absent)", "advise if kit ≤ ½ threshold", "no kit error",
                "10 min per test", "2012–13 with 2000–01 survey"]:
        for b in ("sweep:-", "static:-"):
            x = S1[(S1.setting == lab) & (S1.thr == 50) & (S1.radius == 100) & (S1.thin == 1) &
                   (S1.horizon == "5 d/1000 wells") & (S1.comparator == b)]
            if len(x):
                fr.append(dict(setting=lab, comparator=ARMS[b][0].lower(), **x.iloc[0][["diff", "lo", "hi"]].to_dict()))
    for lab in ("2000–01 census; outcome prot_cap_share", "2000–01 census; outcome switch_share"):
        for b in ("sweep:-", "static:-"):
            x = S1[(S1.setting == lab) & (S1.thr == 50) & (S1.comparator == b)]
            if len(x):
                fr.append(dict(setting=lab.replace("2000–01 census; outcome prot_cap_share", "capacity 5 households (10 days)")
                               .replace("2000–01 census; outcome switch_share", "distance-decay switching (10 days)"),
                               comparator=ARMS[b][0].lower(), **x.iloc[0][["diff", "lo", "hi"]].to_dict()))
    forest(fr)
    print("done")
