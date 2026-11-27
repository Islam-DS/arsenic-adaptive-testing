"""Run campaigns over blocks x policies x learners x seeds x settings.

Development mode (default) uses dev blocks with leave-one-block-out fitting: every model used on a dev block
is fitted on the other dev blocks. Final mode (--final) fits on all dev blocks and runs on the test blocks.

    python -X utf8 src/f1/experiment.py --tag dev_main
    python -X utf8 src/f1/experiment.py --final --tag final_main
"""
import argparse, itertools, json, pathlib, sys, time, zlib
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import load_wells, load_kit, ROOT
from sim import fit_prior, fit_gp
from sim2 import Campaign, KitModel, Field, PriorLearner, GPLearner, KNNLearner, GBMLearner, train_gbm, run

OUT = ROOT / "results"
DAY_MARKS = (0.5, 1, 2, 3, 5, 7, 10)


def build_models(fit_blocks, thr, seed, learners, kit_train=None):
    fit = pd.concat(fit_blocks)
    prior = fit_prior(fit)
    g = fit_gp(fit, prior, seed=seed)
    models = {"prior": PriorLearner()}
    if "gp" in learners:
        models["gp"] = GPLearner(g.sill, g.range_m, g.nugget)
    if "knn" in learners:
        models["knn"] = KNNLearner()
    rng = np.random.default_rng(seed)
    blocks = [b.reset_index(drop=True) for b in fit_blocks]
    if "lgbm" in learners:
        models["lgbm"] = GBMLearner(train_gbm(blocks, prior, thr, rng, "lgbm", kit=kit_train), name="lgbm")
    if "rf" in learners:
        models["rf"] = GBMLearner(train_gbm(blocks, prior, thr, rng, "rf", kit=kit_train), name="rf")
    return prior, models


LEARNER_POLICIES = ("learning", "uncertainty", "static_map")


def one(blk_name, blk, prior, models, policy, learner, seed, radius, thr, kit, fieldp, days, thin, advise=1.0,
        cap=None, known=0.0, known_noise=0.0):
    # stable seed: same block/seed/setting -> same start well, kit draws and earlier-survey wells for every policy
    rng = np.random.default_rng([seed, zlib.crc32(blk_name.encode()), int(radius), int(thr), int(1000 * thin)])
    b = blk
    if thin < 1:
        b = blk.sample(frac=thin, random_state=seed).reset_index(drop=True)
    camp = Campaign(b, prior, radius, thr, rng, kit=kit, advise_frac=advise, known_frac=known, known_noise=known_noise)
    rng = np.random.default_rng([seed, zlib.crc32(blk_name.encode()), 7])   # policy randomness (start, ties)
    lr = models.get(learner)
    t = time.time()
    r = run(camp, policy, lr, rng, fieldp, days=days, cap=cap)
    rows = []
    for dm in DAY_MARKS:
        k = np.searchsorted(r.minutes, dm * fieldp.day_min, side="right") - 1
        rows.append(dict(block=blk_name, policy=policy, learner=learner if policy in LEARNER_POLICIES else "-",
                         seed=seed, radius=radius, thr=thr, kit=kit is not None, thin=thin, advise=advise, cap=cap, known=known, known_noise=known_noise, days=dm,
                         n=r.n, can_secure=r.can_secure, tests=k + 1,
                         true_sec=(r.true_sec[k] if k >= 0 else 0) / r.n,
                         believed=(r.believed[k] if k >= 0 else 0) / r.n,
                         misled=(r.misled[k] if k >= 0 else 0) / r.n,
                         switch=(r.extra['switch'][k] if k >= 0 else 0) / r.n,
                         nozone_auc=r.nozone_auc, nozone_prec=r.nozone_prec, nozone_rec=r.nozone_rec,
                         secs=time.time() - t))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--radius", type=float, nargs="+", default=[100])
    ap.add_argument("--thr", type=float, nargs="+", default=[50])
    ap.add_argument("--kit", choices=["on", "off", "both"], default="on")
    ap.add_argument("--thin", type=float, nargs="+", default=[1.0])
    ap.add_argument("--days", type=float, default=5)
    ap.add_argument("--policies", nargs="+", default=["sweep", "static", "adaptive", "learning"])
    ap.add_argument("--learners", nargs="+", default=["gp", "knn", "lgbm", "rf"])
    ap.add_argument("--test_min", type=float, default=10)
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--advise", type=float, default=1.0)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--known", type=float, default=0.0)
    ap.add_argument("--known_noise", type=float, default=0.0)
    ap.add_argument("--kit_train", action="store_true", help="train tree learners on simulated kit readings")
    ap.add_argument("--swap", action="store_true", help="cross-fitting: fit on test blocks, evaluate on dev blocks")
    a = ap.parse_args()

    d = load_wells()
    kitm = KitModel(load_kit())
    fieldp = Field(test_min=a.test_min)
    dev = {b: g.reset_index(drop=True) for b, g in d[d.split == "dev"].groupby("block")}
    kits = {"on": [kitm], "off": [None], "both": [None, kitm]}[a.kit]

    jobs = []
    for thr in a.thr:
        kt = kitm if a.kit_train else None
        test = {b: g.reset_index(drop=True) for b, g in d[d.split == "test"].groupby("block")}
        fit_set, eval_set = (test, dev) if a.swap else (dev, test)
        if a.final:
            prior, models = build_models(list(fit_set.values()), thr, 0, a.learners, kt)
            targets = {b: (g, prior, models) for b, g in eval_set.items()}
        else:
            targets = {}
            for b in dev:
                prior, models = build_models([g for k, g in dev.items() if k != b], thr, 0, a.learners, kt)
                targets[b] = (dev[b], prior, models)
        for (b, (blk, prior, models)), radius, kit, thin, seed in itertools.product(
                targets.items(), a.radius, kits, a.thin, range(a.seeds)):
            for pol in a.policies:
                for lr in (a.learners if pol in LEARNER_POLICIES else ["prior"]):
                    jobs.append((b, blk, prior, models, pol, lr, seed, radius, thr, kit, fieldp, a.days, thin, a.advise,
                                 a.cap, a.known, a.known_noise))
    print(len(jobs), "campaigns", flush=True)
    t = time.time()
    rows = Parallel(n_jobs=a.jobs, verbose=5)(delayed(one)(*j) for j in jobs)
    df = pd.DataFrame([r for rr in rows for r in rr])
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / f"f1_{a.tag}.csv", index=False)
    (OUT / f"f1_{a.tag}.json").write_text(json.dumps(vars(a), indent=1))
    print(f"done in {time.time()-t:.0f}s")
    g = df[df.days.isin([1, 3])].groupby(["thr", "radius", "kit", "thin", "days", "policy", "learner"])
    print(g[["true_sec", "misled", "tests", "nozone_auc"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
