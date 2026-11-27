"""Pilot: development blocks only. Compares the four policies at 100 m radius, 50 ppb, perfect lab results."""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from data import load_wells, ROOT
from sim import fit_prior, fit_gp, run_campaign

RADIUS, THR, BATCH, FRAC = 100, 50, 20, 0.30
d = load_wells()
dev = d[d.split == "dev"]
prior = fit_prior(dev)
gp = fit_gp(dev, prior)
print("prior bins:", {int(k): round(v, 2) for k, v in prior.mean_by_bin.items()})
print("gp:", gp)

rows = []
for b, blk in dev.groupby("block"):
    blk = blk.reset_index(drop=True)
    budget = int(FRAC * len(blk))
    for pol in ["random", "static", "adaptive", "learning"]:
        seeds = range(5) if pol == "random" else [0]
        for s in seeds:
            t = time.time()
            c = run_campaign(blk, prior, gp, pol, budget, BATCH, RADIUS, THR, seed=s) / len(blk)
            for f in (0.05, 0.10, 0.20, 0.30):
                k = max(1, int(f * len(blk))) - 1
                rows.append(dict(block=b, n=len(blk), policy=pol, seed=s, budget_frac=f,
                                 secured=c[min(k, len(c) - 1)], auc=c[:k + 1].mean()))
            print(f"{b} n={len(blk)} {pol} s{s}: secured@30%={c[-1]:.3f} ({time.time()-t:.1f}s)", flush=True)

r = pd.DataFrame(rows)
out = ROOT / "results" / "f1_pilot.csv"
r.to_csv(out, index=False)
print(r.groupby(["budget_frac", "policy"])[["secured", "auc"]].mean().unstack().round(3))
