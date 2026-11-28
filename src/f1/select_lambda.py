"""Select the look-ahead weight per policy from the development run.   python -X utf8 src/f1/select_lambda.py [tag ...]

Rule (fixed before the final runs): for each policy, the lambda with the highest mean share of exposed households
protected at 5, 10 and 20 team-days, averaged over both thresholds and all development blocks and runs.
dev6_lam is the run on which the weights were chosen (earlier development version of the simulator); dev6_lam_rerun
repeats it with the final code. Both select the same weights.
"""
import pathlib, sys
import pandas as pd

RES = pathlib.Path(__file__).resolve().parents[2] / "results"

for tag in (sys.argv[1:] or ["dev6_lam", "dev6_lam_rerun"]):
    f = RES / f"f1_{tag}.csv"
    if not f.exists():
        continue
    d = pd.read_csv(f)
    d = d[d.days.isin([5, 10, 20])]
    score = d.groupby(["policy", "lam"]).protected_share.mean().unstack("lam")
    print(f"--- {tag}\n" + (100 * score).round(2).to_string())
    print("chosen lambda:", score.idxmax(axis=1).to_dict(), "\n")
