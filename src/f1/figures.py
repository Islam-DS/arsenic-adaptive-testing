"""Figures and tables for the manuscript from the final (test-block) results.

    python -X utf8 src/f1/figures.py
"""
import pathlib, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import load_wells, ROOT

RES, FIG = ROOT / "results", ROOT / "manuscript" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300})
ARMS = {"sweep:-": ("Door-to-door sweep", "#9aa3ad"), "static:-": ("Static depth plan", "#7a9cc0"),
        "adaptive:-": ("Adaptive (no learning)", "#4a7fb0"), "learning:knn": ("Learning: kNN", "#e0a458"),
        "learning:gp": ("Learning: GP", "#c97b3a"), "learning:lgbm": ("Learning: LightGBM", "#5a9a6e"),
        "learning:rf": ("Learning: random forest", "#1d6a3a"),
        "uncertainty:rf": ("Uncertainty-driven (RF)", "#b05a7a"), "oracle:-": ("Oracle (knows all wells)", "#222222")}


def load(tag):
    d = pd.read_csv(RES / f"f1_{tag}.csv")
    d["arm"] = d.policy + ":" + d.learner
    return d


def boot_diff(d, a, b, metric="true_sec", n=4000, seed=0):
    """Mean difference a-b over blocks, CI by resampling blocks then seeds (paired on block, seed)."""
    w = d.pivot_table(index=["block", "seed"], columns="arm", values=metric)
    diff = (w[a] - w[b]).reset_index()
    blocks = diff.block.unique()
    rng = np.random.default_rng(seed)
    by = {k: g[0].to_numpy() for k, g in diff.set_index("block").groupby(level=0)}
    est = np.mean([by[k].mean() for k in blocks])
    bs = []
    for _ in range(n):
        bb = rng.choice(blocks, len(blocks))
        bs.append(np.mean([rng.choice(by[k], len(by[k])).mean() for k in bb]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    rel = est / w[b].groupby(level=0).mean().mean()
    blocks_better = sum(by[k].mean() > 0 for k in blocks)
    from scipy import stats
    bm = np.array([by[k].mean() for k in blocks])          # block means: t-interval with n_blocks - 1 df
    half = stats.t.ppf(0.975, len(bm) - 1) * bm.std(ddof=1) / np.sqrt(len(bm))
    return est, lo, hi, rel, blocks_better, len(blocks), est - half, est + half


def fig_map():
    from scipy.spatial import ConvexHull
    from data_ext import load_bd2012
    d = load_wells()
    u = load_bd2012()
    cols = {"≤10 µg/L": "#2c7fb8", "11-50 µg/L": "#fdae61", ">50 µg/L": "#d7191c"}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.3), gridspec_kw=dict(width_ratios=[1, 1.25]))
    for ax, df, title in ((axes[0], d, "(a) 2000-01 laboratory census (6,595 wells)"),
                          (axes[1], u, "(b) 2012-13 upazila field-kit survey (46,976 wells)")):
        cat = pd.cut(df.as_ppb, [-1, 10, 50, 1e9], labels=list(cols))
        for c, g in df.groupby(cat, observed=True):
            ax.scatter(g.x / 1000, g.y / 1000, s=0.6 if len(df) > 10000 else 1.2, c=cols[c], linewidths=0,
                       label=f"{c} ({100 * len(g) / len(df):.0f}%)")
        for bname, g in df.groupby("block"):
            P = g[["x", "y"]].to_numpy() / 1000
            if len(P) < 3:
                continue
            h = P[ConvexHull(P).vertices]; h = np.vstack([h, h[:1]])
            test = g.split.iat[0] == "test"
            ax.plot(h[:, 0], h[:, 1], ls="-" if test else "--", lw=0.5, color="black" if test else "#555555")
        if df is u:                                   # outline of the 2000-01 census area on the upazila map
            c = d.dropna(subset=["Latitude", "Longitude"])
            lat0, lon0 = u["lat"].mean(), u["lon"].mean()
            C = np.c_[(c.Longitude - lon0) * 111_320 * np.cos(np.radians(lat0)), (c.Latitude - lat0) * 110_574] / 1000
            hc = C[ConvexHull(C).vertices]; hc = np.vstack([hc, hc[:1]])
            ax.plot(hc[:, 0], hc[:, 1], color="#6a3d9a", lw=1.2, label="2000-01 census area")
        ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
        ax.set_xlabel("Easting (km)", fontsize=10); ax.set_ylabel("Northing (km)", fontsize=10)
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        bx, by = x0 + 0.05 * (x1 - x0), y0 + 0.05 * (y1 - y0)
        ax.plot([bx, bx + 1], [by, by], color="black", lw=2)
        ax.text(bx + 0.5, by + 0.03 * (y1 - y0), "1 km", ha="center", fontsize=9)
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, l, markerscale=9, frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_map.png")
    plt.close(fig)


def fig_curves(d):
    sub = d[(d.radius == 100) & d.kit & (d.thin == 1)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=False)
    for ax, thr in zip(axes, (50, 10)):
        s = sub[sub.thr == thr]
        for arm, (lab, col) in ARMS.items():
            g = s[s.arm == arm].groupby("days").true_sec
            if g.ngroups == 0:
                continue
            m = g.mean()
            ax.plot(m.index, 100 * m.values, marker="o", ms=3, color=col, label=lab,
                    lw=2.2 if arm == "learning:rf" else 1.2)
        ax.set_title(f"Safe threshold {thr} µg/L, access radius 100 m", fontsize=9)
        ax.set_xlabel("Field-team days")
        ax.set_ylabel("Households with a tested-safe well (%)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_curves.png")
    plt.close(fig)


def fig_learners(d):
    s = d[(d.days == 3) & d.kit & (d.thin == 1)]
    piv = s.pivot_table(index=["thr", "radius"], columns="arm", values="true_sec") * 100
    piv = piv[[a for a in ARMS if a in piv.columns]]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    im = ax.imshow(piv.values, cmap="Greens", aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [ARMS[a][0].replace("Learning: ", "") for a in piv.columns], rotation=25,
                  ha="right", fontsize=7)
    ax.set_yticks(range(piv.shape[0]), [f"{int(t)} µg/L, {int(r)} m" for t, r in piv.index], fontsize=7)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 60 else "black")
    ax.set_title("Households secured after 3 team-days (%), test blocks, with kit error", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_learners.png")
    plt.close(fig)


def fig_harm(main, adv):
    rows = []
    for name, d in (("Advise if kit ≤ threshold", main), ("Advise if kit ≤ ½ threshold", adv)):
        s = d[(d.days == 3) & d.kit & (d.radius == 100) & (d.thin == 1)]
        for (thr, arm), g in s.groupby(["thr", "arm"]):
            if arm in ("static:-", "learning:rf", "adaptive:-"):
                rows.append(dict(rule=name, thr=thr, arm=arm, sec=100 * g.true_sec.mean(), mis=100 * g.misled.mean()))
    r = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, thr in zip(axes, (50, 10)):
        for (rule, arm), g in r[r.thr == thr].groupby(["rule", "arm"]):
            ax.scatter(g.mis, g.sec, color=ARMS[arm][1], marker="o" if "½" not in rule else "s", s=40,
                       label=f"{ARMS[arm][0]}; {rule}")
        ax.set_xlabel("Households misled by a false-safe kit reading (%)")
        ax.set_ylabel("Households truly secured (%)")
        ax.set_title(f"{thr} µg/L, 100 m, 3 team-days", fontsize=9)
    axes[0].legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_harm.png")
    plt.close(fig)


def table_main(sets):
    """sets: list of (label, dataframe) already filtered to one setting. Learning RF vs comparators, days 1/3/10."""
    out = []
    for label, s, comps in sets:
        for b in comps:
            for days in (1, 3, 10):
                x = s[s.days == days]
                if b not in set(x.arm) or "learning:rf" not in set(x.arm):
                    continue
                est, lo, hi, rel, nb, nbt, tlo, thi = boot_diff(x, "learning:rf", b)
                out.append(dict(setting=label, days=days, comparator=b, comparator_pct=100 * x[x.arm == b].true_sec.mean(),
                                learning_pct=100 * x[x.arm == "learning:rf"].true_sec.mean(), diff=100 * est,
                                boot_lo=100 * lo, boot_hi=100 * hi, t_lo=100 * tlo, t_hi=100 * thi,
                                rel_gain=100 * rel, blocks_positive=f"{nb}/{nbt}"))
    t = pd.DataFrame(out).round(1)
    t.to_csv(RES / "f1_table_S1.csv", index=False)
    return t


def days_to_reach(d, target):
    """Team-days (interpolated on the day marks) for the mean curve to reach `target` share secured."""
    out = {}
    for arm, g in d.groupby("arm"):
        m = g.groupby("days").true_sec.mean()
        x = np.r_[0, m.index.values]
        y = np.r_[0, m.values]
        out[arm] = float(np.interp(target, y, x)) if y.max() >= target else np.nan
    return out


if __name__ == "__main__":
    main = load("v2_main"); adv = load("v2_advise"); known = load("v3_known"); noisy = load("v3_known_noisy"); cap = load("v2_cap")
    thin = load("v2_thin"); t10 = load("v2_t10"); t5 = load("v2_t5")
    fig_map(); fig_curves(main); fig_learners(main); fig_harm(main, adv)
    K = lambda d, thr=50, r=100, kit=True: d[(d.thr == thr) & (d.radius == r) & (d.kit == kit)]
    sets = []
    for thr in (50, 10):
        for r in (50, 100, 200):
            for kit in (True, False):
                comps = ["static:-", "adaptive:-", "sweep:-", "uncertainty:rf", "oracle:-"] if (r == 100 and kit) else ["static:-", "adaptive:-"]
                sets.append((f"{thr} µg/L, {r} m, kit {'on' if kit else 'off'}", K(main, thr, r, kit), comps))
    for thr in (50, 10):
        sets.append((f"{thr} µg/L, 100 m, advise ≤ ½ threshold", adv[adv.thr == thr], ["static:-", "adaptive:-"]))
        mp = ["static_map:rf", "static_map:lgbm", "static_map:gp", "static_map:knn", "static:-", "adaptive:-"]
        sets.append((f"{thr} µg/L, 100 m, earlier 5% survey", known[known.thr == thr], mp))
        sets.append((f"{thr} µg/L, 100 m, earlier 5% survey with change", noisy[noisy.thr == thr], mp))
    sets.append(("50 µg/L, 100 m, capacity 5", cap, ["static:-", "adaptive:-", "sweep:-", "oracle:-"]))
    for th in (0.5, 0.25):
        sets.append((f"50 µg/L, 100 m, {int(th*100)}% of wells", thin[thin.thin == th], ["static:-", "adaptive:-", "sweep:-"]))
    sets.append(("50 µg/L, 100 m, 10 min/test", t10, ["static:-", "adaptive:-", "sweep:-"]))
    sets.append(("50 µg/L, 100 m, 5 min/test", t5, ["static:-", "adaptive:-", "sweep:-"]))
    t = table_main(sets)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(t[t.days == 3].to_string())
    print(t[(t.days != 3) & t.setting.str.startswith("50 µg/L, 100 m, kit on")].to_string())
    for thr, tgs in ((50, (0.3, 0.5, 0.7)), (10, (0.2, 0.3, 0.4))):
        for tg in tgs:
            print(thr, "team-days to", tg, {k: round(v, 2) for k, v in days_to_reach(K(main, thr), tg).items()})
    s = K(main)
    print("switch metric 3d:", s[s.days == 3].groupby("arm").switch.mean().mul(100).round(1).to_dict())
    e = boot_diff(s[s.days == 3], "learning:rf", "static:-", metric="switch"); print("switch diff rf-static", [round(100*x, 1) if isinstance(x, float) else x for x in e])
    print("misled 3d:", s[s.days == 3].groupby("arm")[["true_sec", "misled", "believed"]].mean().mul(100).round(1).to_string())
    a3 = adv[(adv.days == 3)]; print("advise:", a3.groupby(["thr", "arm"])[["true_sec", "misled"]].mean().mul(100).round(1).to_string())
    print("tests/day:", s[s.days == 1].groupby("arm").tests.mean().round(1).to_dict())
    pb = s[s.days == 3].pivot_table(index="block", columns="arm", values="true_sec").mul(100).round(1)
    pb[[c for c in ARMS if c in pb.columns]].to_csv(RES / "f1_table_S3_blocks.csv")
    print("per-block 3d:", s[s.days == 3].pivot_table(index="block", columns="arm", values="true_sec").mul(100).round(1).to_string())
