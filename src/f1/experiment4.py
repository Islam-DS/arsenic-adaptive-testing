"""v6 experiments (version used in the article). Uses sim4.run4.

Key changes against experiment3:
  * repeated random splits (--splits): every split re-assigns blocks to the two halves at random and cross-fits both ways;
  * per-policy look-ahead weights (--lams policy=value ...), chosen on development blocks;
  * 2012-13 survey: --impute draws a laboratory 'truth' for every well from the 944 kit-lab pairs with the same kit
    reading, while the recorded kit reading is what the team observes (so kit error and misled households exist);
  * earlier survey (--known -1): links checked spatially (<= 100 m) and old status converted to calibrated
    P(safe now | old status) estimated on the fitting half;
  * field realism (--daily_return, --absent q) and extra outcomes (capacity, distance decay, misled split).

    python -X utf8 src/f1/experiment4.py --tag dev6_lam --dev --lams static=0 static=0.1 ...   (see README)
"""
import argparse, itertools, json, os, pathlib, sys, time, zlib
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import load_wells, load_kit, ROOT
from sim2 import Campaign, KitModel, Field, GPLearner
from sim4 import run4
from experiment import build_models
from experiment3 import tertiles

OUT = ROOT / "results"
PERSONS_PER_WELL = 8
COST_PER_DAY = 28.80


def resplit(d, split_seed):
    if split_seed == 0:
        return d
    blocks = sorted(d["block"].unique())
    rng = np.random.default_rng(1000 + split_seed)
    half = set(rng.choice(blocks, size=len(blocks) // 2, replace=False))
    d = d.copy()
    d["split"] = np.where(d["block"].isin(half), "test", "dev")
    return d


def impute_lab(kit_df, kit_vals, rng):
    """Draw a laboratory value for each recorded kit category from the paired wells with that category."""
    lab = kit_df["lab"].to_numpy()
    cat = kit_df["kit"].to_numpy()
    out = np.empty(len(kit_vals))
    for v in np.unique(kit_vals):
        idx = np.flatnonzero(kit_vals == v)
        pool = lab[cat == v]
        if len(pool) == 0:                       # categories absent from the pairs: nearest category
            near = np.unique(cat)[np.argmin(np.abs(np.unique(cat) - v))]
            pool = lab[cat == near]
        out[idx] = rng.choice(pool, size=len(idx))
    return out


def known_probs(fit_df, thr):
    """P(safe now | old safe), P(safe now | old unsafe) from linked wells of the fitting half."""
    x = fit_df[fit_df["old_as"].notna()]
    if len(x) < 20:
        return None
    now = x["as_ppb"] <= thr
    old = x["old_as"] <= thr
    p1 = (now[old].sum() + 1) / (old.sum() + 2)
    p0 = (now[~old].sum() + 1) / ((~old).sum() + 2)
    return (float(p1), float(p0))


def one(area, split, blk_name, blk, prior, models, policy, learner, seed, radius, thr, kit, kit_df, fieldp, days, lam,
        advise, known, known_p, impute, cap, daily_return, absent, thin=1.0, boot_pairs=False):
    key = zlib.crc32(f"{area}/{blk_name}".encode())
    rng = np.random.default_rng([seed, key, int(radius), int(thr), split, int(1000 * thin)])
    b = blk if thin >= 1 else blk.sample(frac=thin, random_state=seed).reset_index(drop=True)
    if impute:
        b = b.copy()
        b["kit_obs"] = b["as_ppb"].to_numpy()
        kd = kit_df
        if boot_pairs:      # one bootstrap sample of the 944 kit-laboratory pairs per replicate (split, seed), shared by all
            kd = kit_df.sample(frac=1.0, replace=True, random_state=10_000 * (split + 1) + seed)   # blocks and policies
        b["as_ppb"] = impute_lab(kd, b["kit_obs"].to_numpy(), rng)
        b["log_as"] = np.log10(b["as_ppb"].clip(lower=1))
    camp = Campaign(b, prior, radius, thr, rng, kit=kit, advise_frac=advise, known_frac=known, known_p=known_p)
    rng = np.random.default_rng([seed, key, 7, split])
    tert = tertiles(b, thr)
    t0 = time.time()
    out, n_exp = run4(camp, policy, models.get(learner), rng, fieldp, days=days, lam=lam, tertile=tert,
                      marks_days=(0.5, 1, 2, 3, 5, 7, 10, 15, 20), cap=cap, daily_return=daily_return, absent=absent)
    n_t = [int(((tert == k) & ~camp.truth_safe).sum()) for k in range(3)]
    rows = []
    for r in out:
        row = dict(area=area, split=split, block=blk_name, policy=policy,
                   learner=learner if policy in ("learning", "uncertainty") else "-", seed=seed, radius=radius,
                   thr=thr, kit=(kit is not None) or impute, impute=impute, advise=advise, known=known, lam=lam,
                   daily_return=daily_return, absent=absent, cap=cap, thin=thin, n=camp.n, exposed=n_exp, secs=time.time() - t0)
        for k, v in r.items():
            row[k] = v
        for k in ("protected", "misled_own", "misled_src", "informed", "switch", "prot_cap", "prot_cap_opt"):
            row[k + "_share"] = r[k] / max(n_exp, 1)
        row["false_alarm_share"] = r["false_alarm"] / max(camp.n - n_exp, 1)
        for t in range(3):
            row[f"prot_t{t}_share"] = r[f"prot_t{t}"] / max(n_t[t], 1)
            row[f"exp_t{t}"] = n_t[t]
        rows.append(row)
    return rows


def add_kit_var(prior, models, kit_var):
    """Sensitivity analysis: GP nugget plus the kit-error variance (the GP is fitted to laboratory values in the census)."""
    if kit_var > 0 and "gp" in models:
        g = models["gp"]
        models = dict(models, gp=GPLearner(g.sill, g.range_m, g.nugget + kit_var))
    return prior, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dev", action="store_true", help="leave-one-block-out on the development half of split 0")
    ap.add_argument("--area", default="araihazar")
    ap.add_argument("--splits", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--radius", type=float, nargs="+", default=[100])
    ap.add_argument("--thr", type=float, nargs="+", default=[50])
    ap.add_argument("--kit", choices=["on", "off"], default="on")
    ap.add_argument("--impute", action="store_true")
    ap.add_argument("--days", type=float, default=20)
    ap.add_argument("--lams", nargs="+", default=["static=0.1", "geometry=0.1", "adaptive=0.1", "learning=0.1",
                                                    "oracle=0.1", "sweep=0", "uncertainty=0"])
    ap.add_argument("--policies", nargs="+", default=["sweep", "static", "geometry", "adaptive", "uncertainty",
                                                        "learning", "oracle"])
    ap.add_argument("--learners", nargs="+", default=["gp"])
    ap.add_argument("--test_min", type=float, default=20)
    ap.add_argument("--advise", type=float, default=1.0)
    ap.add_argument("--known", type=float, default=0.0)
    ap.add_argument("--min_known", type=int, default=0)
    ap.add_argument("--cap", type=int, default=5)
    ap.add_argument("--daily_return", action="store_true")
    ap.add_argument("--absent", type=float, default=0.0)
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--thin", type=float, nargs="+", default=[1.0])
    ap.add_argument("--zero_as", type=float, default=1.0, help="value (ug/L) given to kit readings of 0 before log10")
    ap.add_argument("--boot_pairs", action="store_true", help="bootstrap the 944 kit-laboratory pairs for the imputed truth")
    ap.add_argument("--buffer_m", type=float, default=0.0, help="exclude fitting wells within this distance of evaluated wells")
    ap.add_argument("--kit_nugget", action="store_true", help="add the kit-error variance (944 pairs) to the GP nugget")
    a = ap.parse_args()

    lams = {}
    for s in a.lams:
        k, v = s.split("=")
        lams.setdefault(k, []).append(float(v))
    if a.area == "araihazar":
        d0 = load_wells()
    else:
        from data_ext import load_bd2012
        d0 = load_bd2012()
    kit_df = load_kit()
    os.environ["F1_ZERO_AS"] = str(a.zero_as)          # read by sim2.Campaign in every worker
    if a.area != "araihazar":                          # fitting data are kit readings: same transform
        d0["log_as"] = np.log10(np.clip(np.where(d0["as_ppb"] == 0, a.zero_as, d0["as_ppb"]), 1, None))
    kit_var = 0.0
    if a.kit_nugget:
        kl = np.log10(np.clip(np.where(kit_df["kit"] == 0, a.zero_as, kit_df["kit"]), 1, None))
        kit_var = float(np.var(kl - np.log10(kit_df["lab"].clip(lower=1)), ddof=1))
        print("kit-error variance added to the nugget:", round(kit_var, 4), flush=True)
    kitm = KitModel(kit_df) if (a.kit == "on" and not a.impute) else None
    fieldp = Field(test_min=a.test_min)
    jobs = []
    for split, thr in itertools.product(range(a.splits), a.thr):
        d = resplit(d0, split)
        halves = {s: {b: g.reset_index(drop=True) for b, g in d[d.split == s].groupby("block")} for s in ("dev", "test")}
        targets = []
        if a.dev:
            dev = halves["dev"]
            for b in dev:
                fit = [g for k, g in dev.items() if k != b]
                prior, models = add_kit_var(*build_models(fit, thr, 0, a.learners), kit_var)
                targets.append((b, dev[b], prior, models, known_probs(pd.concat(fit), thr) if a.known < 0 else None))
        else:
            for fit, ev in (("dev", "test"), ("test", "dev")):
                fitb = list(halves[fit].values())
                if a.buffer_m > 0:   # drop fitting wells within buffer_m of any evaluated well (spatial exclusion buffer)
                    tree = cKDTree(np.vstack([g[["x", "y"]].to_numpy() for g in halves[ev].values()]))
                    fitb = [g[tree.query(g[["x", "y"]].to_numpy())[0] > a.buffer_m].reset_index(drop=True) for g in fitb]
                    fitb = [g for g in fitb if len(g)]
                prior, models = add_kit_var(*build_models(fitb, thr, 0, a.learners), kit_var)
                kp = known_probs(pd.concat(fitb), thr) if a.known < 0 else None
                targets += [(b, g, prior, models, kp) for b, g in halves[ev].items()]
        if a.min_known:
            targets = [x for x in targets if "old_as" in x[1] and x[1]["old_as"].notna().sum() >= a.min_known]
        for (b, blk, prior, models, kp), radius, seed, thin in itertools.product(targets, a.radius, range(a.seeds), a.thin):
            for pol in a.policies:
                for lam in lams.get(pol, [0.1]):
                    for lr in (a.learners if pol in ("learning", "uncertainty") else ["prior"]):
                        jobs.append((a.area, split, b, blk, prior, models, pol, lr, seed, radius, thr, kitm, kit_df,
                                     fieldp, a.days, lam, a.advise, a.known, kp, a.impute, a.cap, a.daily_return,
                                     a.absent, thin, a.boot_pairs))
    print(len(jobs), "campaigns", flush=True)
    t = time.time()
    rows = Parallel(n_jobs=a.jobs, verbose=2)(delayed(one)(*j) for j in jobs)
    df = pd.DataFrame([r for rr in rows for r in rr])
    df.to_csv(OUT / f"f1_{a.tag}.csv", index=False)
    (OUT / f"f1_{a.tag}.json").write_text(json.dumps(vars(a), indent=1))
    print(f"done in {time.time()-t:.0f}s")
    df["arm"] = df.policy + ":" + df.learner
    print(df[df.days.isin([5, 10])].groupby(["thr", "days", "arm", "lam"])[["protected_share", "misled_own_share",
          "misled_src_share", "prot_cap_share", "switch_share"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
