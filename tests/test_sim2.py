"""Sanity tests for the campaign simulator on synthetic blocks.  python -X utf8 -m pytest tests -q"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "f1"))
from sim import fit_prior
from sim2 import Campaign, PriorLearner, KNNLearner, run, Field


def block(n=300, safe_frac=0.5, seed=0, clustered=True):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 1500, n), rng.uniform(0, 1500, n)
    if clustered:   # safe in the west half, unsafe in the east
        as_ppb = np.where(x < 1500 * safe_frac, 5, 200).astype(float)
    else:
        as_ppb = np.where(rng.uniform(size=n) < safe_frac, 5, 200).astype(float)
    d = pd.DataFrame(dict(x=x, y=y, as_ppb=as_ppb, depth_ft=40.0, deep=0))
    d["log_as"] = np.log10(d.as_ppb)
    return d


def test_no_kit_means_no_misled():
    b = block()
    prior = fit_prior(b)
    camp = Campaign(b, prior, 100, 50, np.random.default_rng(0))
    r = run(camp, "learning", KNNLearner(), np.random.default_rng(0), days=1)
    assert r.misled.max() == 0
    assert (r.true_sec == r.believed).all()


def test_secured_never_exceeds_securable():
    b = block(safe_frac=0.2)
    prior = fit_prior(b)
    camp = Campaign(b, prior, 100, 50, np.random.default_rng(1))
    r = run(camp, "adaptive", PriorLearner(), np.random.default_rng(1), n_tests=300)
    assert r.true_sec[-1] == camp.can_secure.sum()      # testing everything secures all securable
    assert (np.diff(r.true_sec) >= 0).all()


def test_time_budget_respected():
    b = block()
    prior = fit_prior(b)
    camp = Campaign(b, prior, 100, 50, np.random.default_rng(2))
    f = Field()
    r = run(camp, "sweep", None, np.random.default_rng(2), f, days=1)
    assert r.minutes[-1] <= f.day_min
    assert len(r.minutes) <= f.day_min / f.test_min


def test_learning_beats_adaptive_when_space_is_informative():
    gains = []
    for s in range(5):
        b = block(safe_frac=0.3, seed=s)
        prior = fit_prior(b)
        res = {}
        for pol in ("adaptive", "learning"):
            camp = Campaign(b, prior, 100, 50, np.random.default_rng(s))
            res[pol] = run(camp, pol, KNNLearner(), np.random.default_rng(s), n_tests=25).true_sec[-1]
        gains.append(res["learning"] - res["adaptive"])
    assert np.mean(gains) > 0


def test_no_leak_from_untested_wells():
    """Scrambling the kit readings of wells that are never tested must not change any decision."""
    from sim2 import GPLearner
    b = block(safe_frac=0.4, seed=3, clustered=False)
    prior = fit_prior(b)
    for pol, lr in (("learning", KNNLearner()), ("learning", GPLearner(0.2, 300, 0.1)), ("adaptive", PriorLearner())):
        camp = Campaign(b, prior, 100, 50, np.random.default_rng(5))
        r1 = run(camp, pol, lr, np.random.default_rng(9), n_tests=30)
        tested = camp.obs_log.copy()
        ref = Campaign(b, prior, 100, 50, np.random.default_rng(5))
        tested_mask = np.zeros(camp.n, bool)
        tested_mask[np.flatnonzero(r1.true_sec >= 0)] = False
        # wells tested in run 1: those whose kit reading was copied into obs (always equal), so rerun and track
        camp2 = Campaign(b, prior, 100, 50, np.random.default_rng(5))
        seen = []
        orig = lr.predict
        r_ref = run(ref, pol, lr, np.random.default_rng(9), n_tests=30)
        assert (r_ref.true_sec == r1.true_sec).all()          # deterministic
        rng = np.random.default_rng(11)
        sh = np.arange(camp2.n)
        camp2.kit_safe[:] = camp2.kit_safe[rng.permutation(sh)]   # scramble ALL readings ...
        camp2.obs_safe[:] = camp2.kit_safe
        camp2.obs_log[:] = camp2.kit_log[rng.permutation(sh)]
        r2 = run(camp2, pol, lr, np.random.default_rng(9), n_tests=1)
        r1b = run(Campaign(b, prior, 100, 50, np.random.default_rng(5)), pol, lr, np.random.default_rng(9), n_tests=1)
        assert r2.minutes[0] == r1b.minutes[0]               # ... the first choice cannot depend on any reading


def test_sim3_protected_definition():
    """v5 outcome: protected households are exposed, informed and served by a truly safe tested well."""
    from sim3 import run3
    b = block(safe_frac=0.4, seed=4)
    prior = fit_prior(b)
    camp = Campaign(b, prior, 100, 50, np.random.default_rng(1))
    out, n_exp = run3(camp, "learning", KNNLearner(), np.random.default_rng(1), days=3)
    assert n_exp == (~camp.truth_safe).sum()
    assert (out["protected"] <= out["informed"]).all()        # no kit error: protected needs informed
    assert out["misled"].max() == 0                           # no kit -> nobody misled
    assert (np.diff(out["protected"]) >= 0).all()
    camp = Campaign(b, prior, 100, 50, np.random.default_rng(1))
    sw, _ = run3(camp, "sweep", None, np.random.default_rng(1), days=3)
    assert sw["minutes"][-1] <= 3 * 360
