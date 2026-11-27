"""Experiments with the v5 public-health outcome (sim3.run3).

Development mode: leave-one-block-out on the 5 development blocks.
Final mode (--final): cross-fitting. Models are fitted on one half (dev or test blocks) and evaluated on the other,
in both directions, so all 10 blocks are held-out once.  --area bd2012 runs the 2012-13 upazila survey.

    python -X utf8 src/f1/experiment3.py --tag dev5 --lam 0 0.5 1
    python -X utf8 src/f1/experiment3.py --final --tag v5_main
"""
import argparse, itertools, json, pathlib, sys, time, zlib
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import load_wells, load_kit, ROOT
from sim2 import Campaign, KitModel, Field
from sim3 import run3
from experiment import build_models

OUT = ROOT / "results"
DAY_MARKS = (0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30)
PERSONS_PER_WELL = 8          # Araihazar 2012-13 (Jamil et al. 2019)
COST_PER_DAY = 12 * 2.40      # ~12 tests/day at US$2.40 all-in per test (van Geen et al. 2014)


def tertiles(blk, thr, r=200):
    """Tertile (0 = fewest safe wells nearby) of the local share of safe wells within r, for evaluation only."""
    from scipy.spatial import cKDTree
    X = blk[["x", "y"]].to_numpy()
    safe = (blk["as_ppb"].to_numpy() <= thr).astype(float)
    nb = cKDTree(X).query_ball_point(X, r)
    share = np.array([safe[v].mean() for v in nb])
    return np.digitize(share, np.quantile(share, [1 / 3, 2 / 3]))


def one(area, blk_name, blk, prior, models, policy, learner, seed, radius, thr, kit, fieldp, days, lam,
        advise=1.0, known=0.0, known_noise=0.0, thin=1.0):
    key = zlib.crc32(f"{area}/{blk_name}".encode())
    b = blk if thin >= 1 else blk.sample(frac=thin, random_state=seed).reset_index(drop=True)
    rng = np.random.default_rng([seed, key, int(radius), int(thr), int(1000 * thin)])
    camp = Campaign(b, prior, radius, thr, rng, kit=kit, advise_frac=advise, known_frac=known, known_noise=known_noise)
    rng = np.random.default_rng([seed, key, 7])
    tert = tertiles(b, thr)
    t0 = time.time()
    out, n_exp = run3(camp, policy, models.get(learner), rng, fieldp, days=days, lam=lam, tertile=tert)
    n_t = [(tert == k)[~camp.truth_safe].sum() for k in range(3)]
    rows = []
    for dm in DAY_MARKS:
        k = np.searchsorted(out["minutes"], dm * fieldp.day_min, side="right") - 1
        g = (lambda name: out[name][k] if k >= 0 else 0)
        row = dict(area=area, block=blk_name, policy=policy, learner=learner if policy == "learning" else "-",
                   seed=seed, radius=radius, thr=thr, kit=kit is not None, advise=advise, known=known,
                   known_noise=known_noise, thin=thin, lam=lam, days=dm, n=camp.n, exposed=n_exp, tests=k + 1,
                   protected=g("protected") / max(n_exp, 1), misled=g("misled") / max(n_exp, 1),
                   informed=g("informed") / max(n_exp, 1), access=g("access") / camp.n,
                   persons_protected=g("protected") * PERSONS_PER_WELL,
                   usd_per_person=(dm * COST_PER_DAY) / max(g("protected") * PERSONS_PER_WELL, 1),
                   secs=time.time() - t0)
        for t in range(3):
            row[f"prot_t{t}"] = g(f"prot_t{t}") / max(n_t[t], 1)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--radius", type=float, nargs="+", default=[100])
    ap.add_argument("--thr", type=float, nargs="+", default=[50])
    ap.add_argument("--kit", choices=["on", "off", "both"], default="on")
    ap.add_argument("--days", type=float, default=10)
    ap.add_argument("--lam", type=float, nargs="+", default=[0.5])
    ap.add_argument("--policies", nargs="+", default=["sweep", "static", "geometry", "adaptive", "learning", "oracle"])
    ap.add_argument("--learners", nargs="+", default=["gp", "knn", "lgbm", "rf"])
    ap.add_argument("--test_min", type=float, default=20)
    ap.add_argument("--advise", type=float, default=1.0)
    ap.add_argument("--known", type=float, default=0.0)
    ap.add_argument("--known_noise", type=float, default=0.0)
    ap.add_argument("--thin", type=float, nargs="+", default=[1.0])
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--area", default="araihazar")
    ap.add_argument("--min_known", type=int, default=0, help="only blocks with at least this many wells from the earlier survey")
    a = ap.parse_args()

    if a.area == "araihazar":
        d = load_wells()
    elif a.area == "bd2012":
        from data_ext import load_bd2012
        d = load_bd2012()
    else:
        raise SystemExit(f"unknown area: {a.area}")
    kitm = KitModel(load_kit())
    fieldp = Field(test_min=a.test_min)
    kits = {"on": [kitm], "off": [None], "both": [None, kitm]}[a.kit]
    halves = {s: {b: g.reset_index(drop=True) for b, g in d[d.split == s].groupby("block")} for s in ("dev", "test")}
    jobs = []
    for thr in a.thr:
        targets = []                                   # (block, frame, prior, models)
        if a.final:
            for fit, ev in (("dev", "test"), ("test", "dev")):
                prior, models = build_models(list(halves[fit].values()), thr, 0, a.learners)
                targets += [(b, g, prior, models) for b, g in halves[ev].items()]
        else:
            dev = halves["dev"]
            for b in dev:
                prior, models = build_models([g for k, g in dev.items() if k != b], thr, 0, a.learners)
                targets.append((b, dev[b], prior, models))
        if a.min_known:
            targets = [x for x in targets if "old_as" in x[1] and x[1]["old_as"].notna().sum() >= a.min_known]
        for (b, blk, prior, models), radius, kit, seed, lam, thin in itertools.product(
                targets, a.radius, kits, range(a.seeds), a.lam, a.thin):
            for pol in a.policies:
                for lr in (a.learners if pol == "learning" else ["prior"]):
                    jobs.append((a.area, b, blk, prior, models, pol, lr, seed, radius, thr, kit, fieldp, a.days,
                                 lam, a.advise, a.known, a.known_noise, thin))
    print(len(jobs), "campaigns", flush=True)
    t = time.time()
    rows = Parallel(n_jobs=a.jobs, verbose=2)(delayed(one)(*j) for j in jobs)
    df = pd.DataFrame([r for rr in rows for r in rr])
    df.to_csv(OUT / f"f1_{a.tag}.csv", index=False)
    (OUT / f"f1_{a.tag}.json").write_text(json.dumps(vars(a), indent=1))
    print(f"done in {time.time()-t:.0f}s")
    df["arm"] = df.policy + ":" + df.learner
    g = df[df.days.isin([1, 3, 10])].groupby(["thr", "radius", "kit", "lam", "days", "arm"])
    print(g[["protected", "misled", "informed", "access", "tests"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
