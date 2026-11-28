"""Tests of the v6 campaign simulator used in the article (sim4.run4) on small synthetic blocks.

    python -X utf8 -m pytest tests -q
"""
import sys, pathlib
import numpy as np, pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "f1"))
from sim import fit_prior
from sim2 import Campaign, Field, GPLearner, KitModel, PriorLearner
from sim4 import run4, _capacity_protected, _capacity_optimal

FIELD = Field(test_min=20)
MARKS = (0.5, 1, 2, 3)


def block(n=250, safe_frac=0.45, seed=0, size=1500.0):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, size, n), rng.uniform(0, size, n)
    as_ppb = np.where(rng.uniform(size=n) < safe_frac, 5.0, 200.0)
    d = pd.DataFrame(dict(x=x, y=y, as_ppb=as_ppb, depth_ft=40.0, deep=0))
    d["log_as"] = np.log10(d.as_ppb)
    return d


def kit_model():
    """A field kit that misreads about one well in six, in both directions."""
    rng = np.random.default_rng(0)
    lab = np.r_[rng.uniform(1, 40, 400), rng.uniform(60, 400, 400)]
    kit = np.where(rng.uniform(size=800) < 0.17, np.where(lab < 50, 100, 25), np.where(lab < 50, 10, 200))
    return KitModel(pd.DataFrame(dict(lab=lab, kit=kit)))


def campaign(b, seed=0, kit=None, **kw):
    return Campaign(b, fit_prior(b), 100, 50, np.random.default_rng(seed), kit=kit, **kw)


def go(camp, policy="learning", learner=None, seed=0, **kw):
    learner = learner or GPLearner(0.2, 300, 0.1)
    out, n_exp = run4(camp, policy, learner, np.random.default_rng(seed), FIELD, days=max(MARKS), lam=0.1,
                      marks_days=MARKS, **kw)
    return pd.DataFrame(out), n_exp


@pytest.mark.parametrize("policy", ["sweep", "static", "adaptive", "learning", "oracle"])
def test_protection_bounded_and_monotone(policy):
    b = block(seed=1)
    camp = campaign(b, kit=kit_model())
    r, n_exp = go(camp, policy)
    assert n_exp == int((~camp.truth_safe).sum())
    assert (r.protected <= n_exp).all() and (r.protected <= r.informed).all()
    assert (np.diff(r.protected) >= 0).all() and (np.diff(r.tests) >= 0).all()


def test_no_kit_error_means_nobody_misled_and_no_false_alarms():
    camp = campaign(block(seed=2))
    r, _ = go(camp)
    assert (r.misled_own == 0).all() and (r.misled_src == 0).all() and (r.false_alarm == 0).all()


def test_outcome_definitions_when_every_well_is_tested():
    """After testing every well, misled-own and false alarms equal their definitions on the full block."""
    b = block(n=40, seed=3, size=500)
    camp = campaign(b, kit=kit_model())
    r, _ = go(camp, "sweep")
    last = r.iloc[-1]
    assert last.tests == camp.n
    exposed, ks = ~camp.truth_safe, camp.kit_safe
    assert last.misled_own == int((exposed & ks).sum())
    assert last.false_alarm == int((~exposed & ~ks).sum())
    # protected: exposed, read unsafe, and a truly safe well within r read safe
    src = ks & camp.truth_safe
    has_src = np.asarray(camp.A @ src.astype(float)).ravel() > 0
    assert last.protected == int((exposed & ~ks & has_src).sum())


def test_capacity_rules():
    b = block(seed=4)
    camp = campaign(b, kit=kit_model())
    r, _ = go(camp, cap=1)
    assert (r.prot_cap <= r.prot_cap_opt).all()           # greedy never beats the maximum matching
    assert (r.prot_cap_opt <= r.protected).all()          # capacity can only reduce protection
    # a tiny hand-made case where nearest-first greedy is not optimal
    X = np.array([[0, 0], [10, 0], [60, 0], [140, 0]], float)   # households 0,1 ; sources 2,3 ; radius 100
    d = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], as_ppb=[200, 200, 5, 5], depth_ft=40.0, deep=0))
    d["log_as"] = np.log10(d.as_ppb)
    c = campaign(d)
    assert _capacity_optimal(c, np.array([0, 1]), np.array([2, 3]), 1) == 1 + (np.hypot(*(X[1] - X[3])) <= 100)
    assert _capacity_protected(c, np.array([0, 1]), np.array([2, 3]), 1) <= _capacity_optimal(c, np.array([0, 1]), np.array([2, 3]), 1)


def test_daily_return_charges_the_walk_home():
    """One well at the block centre (the base) and two clusters 3 km away: every day must include the walk out and
    back (78 min each way at 3 km/h with detour 1.3), so at most 1 + (360 - 2 * 78) / 20 = 11 wells fit in day 1."""
    rng = np.random.default_rng(5)
    left = np.c_[-3000 + rng.uniform(-50, 50, 30), rng.uniform(-50, 50, 30)]
    right = np.c_[3000 + rng.uniform(-50, 50, 30), rng.uniform(-50, 50, 30)]
    X = np.vstack([[[0.0, 0.0]], left, right])
    d = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], as_ppb=rng.uniform(100, 300, len(X)), depth_ft=40.0, deep=0))
    d["log_as"] = np.log10(d.as_ppb)
    camp = campaign(d)
    assert int(np.argmin(((camp.X - camp.X.mean(0)) ** 2).sum(1))) == 0      # the centre well is the base
    r_home, _ = go(camp, "sweep", daily_return=True)
    per_mark = r_home.set_index("days").tests
    assert per_mark[1.0] <= 11
    assert per_mark[3.0] <= 3 * 11
    r_free, _ = go(campaign(d), "sweep", daily_return=False)
    assert per_mark[3.0] < r_free.set_index("days").tests[3.0]


def test_absent_owners_use_time():
    b = block(seed=6)
    r0, _ = go(campaign(b), "sweep", absent=0.0)
    r_all, _ = go(campaign(b), "sweep", absent=1.0)
    r_half, _ = go(campaign(b), "sweep", absent=0.5)
    assert (r_all.tests == 0).all() and (r_all.protected == 0).all()
    assert r_half.tests.iloc[-1] < r0.tests.iloc[-1]


def test_identical_seeds_identical_campaigns():
    b = block(seed=7)
    r1, _ = go(campaign(b, seed=3, kit=kit_model()), seed=8)
    r2, _ = go(campaign(b, seed=3, kit=kit_model()), seed=8)
    pd.testing.assert_frame_equal(r1, r2)


class Spy(GPLearner):
    """GP learner that records the set of wells it was conditioned on at every decision."""
    def __init__(self):
        super().__init__(0.2, 300, 0.1)
        self.log = []

    def predict(self, camp, tested, untested):
        self.log.append(tuple(sorted(tested)))
        return super().predict(camp, tested, untested)


def test_no_information_from_untested_wells():
    b = block(seed=9)
    s1 = Spy()
    go(campaign(b, kit=kit_model()), "learning", s1)
    ever = set(s1.log[-1])
    camp2 = campaign(b, kit=kit_model())
    rng = np.random.default_rng(1)
    hidden = np.array([i for i in range(camp2.n) if i not in ever])
    hidden = hidden[: len(hidden) - 5]                       # leave a few candidates untouched for the last pick
    camp2.kit_log[hidden] = rng.permutation(camp2.kit_log[hidden])
    camp2.obs_log[hidden] = rng.uniform(0, 3, len(hidden))
    s2 = Spy()
    go(camp2, "learning", s2)
    assert s1.log == s2.log                                  # identical decisions


def test_earlier_survey_values_do_not_enter_the_gp():
    """Wells with an old measurement use only the calibrated P(safe now | old status); old values are not GP data."""
    b = block(seed=10)
    b["old_as"] = np.where(np.random.default_rng(0).uniform(size=len(b)) < 0.3, b.as_ppb, np.nan)
    kp = (0.8, 0.2)
    c1 = campaign(b, known_frac=-1, known_p=kp)
    s1 = Spy(); go(c1, "learning", s1)
    c2 = campaign(b, known_frac=-1, known_p=kp)
    c2.obs_log[c2.known] = np.random.default_rng(2).uniform(0, 3, len(c2.known))   # scramble the old values
    s2 = Spy(); go(c2, "learning", s2)
    assert s1.log == s2.log


def test_gp_probabilities_match_scikit_learn():
    """The Gaussian-process probability of being safe equals the one implied by scikit-learn's GP regressor with the
    same fixed kernel (exponential covariance = Matern nu 0.5, plus a white-noise nugget)."""
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    rng = np.random.default_rng(11)
    b = block(n=120, seed=11)
    b["as_ppb"] = 10 ** rng.normal(1.6, 0.6, len(b)); b["log_as"] = np.log10(b.as_ppb.clip(lower=1))
    camp = campaign(b)
    sill, rho, nug = 0.2, 300.0, 0.1
    tested, untested = np.arange(40), np.arange(40, len(b))
    p = GPLearner(sill, rho, nug).predict(camp, tested, untested)
    k = ConstantKernel(sill, "fixed") * Matern(rho, "fixed", nu=0.5) + WhiteKernel(nug, "fixed")
    gpr = GaussianProcessRegressor(k, optimizer=None, normalize_y=False).fit(camp.X[tested], camp.obs_log[tested] - camp.m0[tested])
    mean, sd = gpr.predict(camp.X[untested], return_std=True)          # sd includes the nugget
    expected = norm.cdf((np.log10(50) - camp.m0[untested] - mean) / (camp.sd0[untested] * sd / np.sqrt(sill + nug)))
    assert np.allclose(p, expected, atol=1e-8)


@pytest.mark.parametrize("daily_return", [False, True])
def test_counts_at_day_boundaries(daily_return):
    """All wells at one spot (no walking, base included): with 20-minute tests exactly 18 wells fit in a 360-minute day,
    and the counts recorded at 360, 720 and 1,080 minutes are 18, 36 and 54, with or without daily return."""
    n = 80
    d = pd.DataFrame(dict(x=np.zeros(n), y=np.zeros(n), as_ppb=np.random.default_rng(0).uniform(100, 300, n),
                          depth_ft=40.0, deep=0))
    d["log_as"] = np.log10(d.as_ppb)
    r, _ = go(campaign(d), "sweep", daily_return=daily_return)
    t = r.set_index("days").tests
    assert (t[1.0], t[2.0], t[3.0]) == (18, 36, 54)
    assert t[0.5] == 9


def test_walk_home_is_recorded_at_the_day_mark():
    """50 wells at the base and 10 wells 1,300 m away (walk 33.8 min each way). On day 3 the last 14 base wells take
    280 min; a far visit would end the day at 280 + 33.8 + 20 + 33.8 = 367.6 > 360 min, so with daily return the
    team stops and the count at 1,080 min is 50. Without daily return two far wells fit (52)."""
    X = np.vstack([np.zeros((50, 2)), np.c_[np.full(10, 1300.0), np.zeros(10)]])
    d = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], as_ppb=np.random.default_rng(1).uniform(100, 300, len(X)), depth_ft=40.0, deep=0))
    d["log_as"] = np.log10(d.as_ppb)
    camp = campaign(d)
    assert int(np.argmin(((camp.X - camp.X.mean(0)) ** 2).sum(1))) < 50      # the base is in the large cluster
    t_home = go(camp, "sweep", daily_return=True)[0].set_index("days").tests
    t_free = go(campaign(d), "sweep", daily_return=False)[0].set_index("days").tests
    assert (t_home[1.0], t_home[2.0], t_home[3.0]) == (18, 36, 50)
    assert t_free[3.0] == 52
