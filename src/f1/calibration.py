"""Calibration of the predicted probability that a well is safe (Gaussian process and depth prior) on held-out blocks.

    python -X utf8 src/f1/calibration.py

For every block of the evaluation half (split 0, both directions of cross-fitting), a random share of its wells is
treated as tested (5% or 20%; their kit readings are observed), and P(safe) is predicted for the remaining wells with
the Gaussian process fitted on the other half and with the depth prior alone. Predictions are scored against the true
status (laboratory value; imputed from the kit-laboratory pairs in the 2012-13 survey): Brier score, log loss and
reliability by decile of predicted probability, by layer (shallow/deep) and threshold.
Writes results/f1_v6_calibration.csv, results/f1_v6_calibration_bins.csv, every prediction in
results/f1_v6_calibration_predictions.csv.gz and manuscript/figures/figA2_calibration.png.
"""
import pathlib, sys, zlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import load_wells, load_kit, ROOT
from sim2 import Campaign, KitModel
from experiment import build_models
from experiment4 import impute_lab

RES, FIG = ROOT / "results", ROOT / "manuscript" / "figures"
EPS = 1e-6


def predictions(area, frac, thr, reps=3):
    if area == "census":
        d, kitm, impute = load_wells(), KitModel(load_kit()), False
    else:
        from data_ext import load_bd2012
        d, kitm, impute = load_bd2012(), None, True
    kit_df = load_kit()
    halves = {s: {b: g.reset_index(drop=True) for b, g in d[d.split == s].groupby("block")} for s in ("dev", "test")}
    rows = []
    for fit, ev in (("dev", "test"), ("test", "dev")):
        prior, models = build_models(list(halves[fit].values()), thr, 0, ["gp"])
        gp = models["gp"]
        for name, blk in halves[ev].items():
            for rep in range(reps):
                rng = np.random.default_rng([rep, zlib.crc32(f"{area}/{name}".encode()), int(thr), int(100 * frac)])
                b = blk
                if impute:
                    b = blk.copy()
                    b["kit_obs"] = b["as_ppb"].to_numpy()
                    b["as_ppb"] = impute_lab(kit_df, b["kit_obs"].to_numpy(), rng)
                camp = Campaign(b, prior, 100, thr, rng, kit=kitm)
                n = camp.n
                tested = np.sort(rng.choice(n, max(1, int(frac * n)), replace=False))
                untested = np.setdiff1d(np.arange(n), tested)
                p_gp = gp.predict(camp, tested, untested)
                p_prior = camp.p_prior[untested]
                rows.append(pd.DataFrame(dict(area=area, thr=thr, frac=frac, block=name, rep=rep,
                                              layer=np.where(camp.layer[untested] == 1, "deep", "shallow"),
                                              y=camp.truth_safe[untested].astype(int), p_gp=p_gp, p_prior=p_prior)))
    return pd.concat(rows, ignore_index=True)


def scores(g):
    out = {}
    for m in ("gp", "prior"):
        p = np.clip(g[f"p_{m}"].to_numpy(), EPS, 1 - EPS)
        y = g.y.to_numpy()
        out[f"brier_{m}"] = np.mean((p - y) ** 2)
        out[f"logloss_{m}"] = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    out["safe_share"] = g.y.mean()
    out["n"] = len(g)
    return pd.Series(out)


def main():
    P = pd.concat([predictions(a, f, t) for a in ("census", "upazila") for t in (50, 10) for f in (0.05, 0.2)],
                  ignore_index=True)
    S = pd.concat([P.groupby(["area", "thr", "frac"]).apply(scores).reset_index().assign(layer="all"),
                   P.groupby(["area", "thr", "frac", "layer"]).apply(scores).reset_index()], ignore_index=True)
    S.to_csv(RES / "f1_v6_calibration.csv", index=False)
    # every prediction, so that the audit can recompute Table A9 and the reliability bins independently
    P[["area", "thr", "frac", "block", "rep", "layer", "y", "p_gp", "p_prior"]].to_csv(
        RES / "f1_v6_calibration_predictions.csv.gz", index=False, float_format="%.6f", compression="gzip")
    pd.set_option("display.width", 200)
    print(S.round(4).to_string())
    P["bin"] = np.minimum((P.p_gp * 10).astype(int), 9)
    B = P.groupby(["area", "thr", "frac", "bin"]).agg(pred=("p_gp", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index()
    B.to_csv(RES / "f1_v6_calibration_bins.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for ax, area in zip(axes, ("census", "upazila")):
        for (thr, frac), g in B[B.area == area].groupby(["thr", "frac"]):
            g = g[g.n >= 30]
            ax.plot(g.pred, g.obs, "o-", ms=3.5, lw=1.2, label=f"{int(thr)} µg/L, {int(100 * frac)}% tested")
        ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability safe (Gaussian process)", fontsize=9)
        ax.set_ylabel("Observed share safe", fontsize=9)
        ax.set_title("2000-01 census" if area == "census" else "2012-13 upazila survey", fontsize=10)
        ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIG / "figA2_calibration.png", dpi=300); plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
