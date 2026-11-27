"""Replay well-testing campaigns on the Araihazar data.

Every well stands for the household(s) that drink from it. A household is *secured* once a well within
`radius` metres of it (its own included) has been tested and found safe. A campaign is one spatial block;
the policy picks which wells to test, in batches, and only sees the results of wells it has tested.

Policies differ in one thing at a time:
  static    plans the whole budget up front from the depth prior (no results used)
  adaptive  re-plans after each batch using which households are now secured, still with the depth prior
  learning  re-plans after each batch and also updates P(safe) with a spatial model of the results
  random    tests wells in random order
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import norm


@dataclass
class Prior:
    mean_by_bin: dict          # depth bin -> mean log10 As
    sd_by_bin: dict            # depth bin -> sd of log10 As
    bins: np.ndarray           # depth bin edges (ft)

    def bin_of(self, depth):
        b = np.digitize(np.nan_to_num(depth, nan=-1), self.bins)
        b[np.isnan(depth)] = -1                     # unknown depth gets its own bin
        return b

    def mean_sd(self, depth):
        b = self.bin_of(depth)
        return (np.array([self.mean_by_bin[k] for k in b]), np.array([self.sd_by_bin[k] for k in b]))


BINS = np.array([0, 30, 50, 70, 100, 150])


def fit_prior(dev):
    tmp = Prior({}, {}, BINS)
    b = tmp.bin_of(dev["depth_ft"].to_numpy())
    la = dev["log_as"].to_numpy()
    for k in np.unique(b):
        tmp.mean_by_bin[k], tmp.sd_by_bin[k] = la[b == k].mean(), la[b == k].std(ddof=1)
    return tmp


@dataclass
class GP:
    """Exponential covariance on the depth-prior residual; deep and shallow wells are separate fields."""
    sill: float
    range_m: float
    nugget: float

    def cov(self, A, B):
        dd = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
        return self.sill * np.exp(-dd / self.range_m)

    def posterior(self, X, layer, resid, tested, untested):
        mu = np.zeros(len(untested))
        var = np.full(len(untested), self.sill)
        for L in (0, 1):
            t = tested[layer[tested] == L]
            u_mask = layer[untested] == L
            if len(t) == 0 or not u_mask.any():
                continue
            u = untested[u_mask]
            K = self.cov(X[t], X[t]) + self.nugget * np.eye(len(t))
            Ks = self.cov(X[u], X[t])
            c = np.linalg.cholesky(K)
            alpha = np.linalg.solve(c.T, np.linalg.solve(c, resid[t]))
            v = np.linalg.solve(c, Ks.T)
            mu[u_mask] = Ks @ alpha
            var[u_mask] = np.maximum(self.sill - (v ** 2).sum(0), 1e-6)
        return mu, var


def fit_gp(dev, prior, n_sub=1500, seed=0):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    m, _ = prior.mean_sd(dev["depth_ft"].to_numpy())
    r = dev["log_as"].to_numpy() - m
    sh = np.flatnonzero(dev["deep"].to_numpy() == 0)
    idx = np.random.default_rng(seed).choice(sh, size=min(n_sub, len(sh)), replace=False)
    k = ConstantKernel(0.2, (1e-3, 5)) * Matern(300, (10, 5000), nu=0.5) + WhiteKernel(0.1, (1e-4, 2))
    g = GaussianProcessRegressor(k, normalize_y=False).fit(dev[["x", "y"]].to_numpy()[idx], r[idx])
    p = g.kernel_.get_params()
    return GP(sill=p["k1__k1__constant_value"], range_m=p["k1__k2__length_scale"], nugget=p["k2__noise_level"])


def run_campaign(block, prior, gp, policy, budget, batch, radius, thr=50, seed=0, kit=None):
    """Return the number of secured households after each test (array of length `budget`)."""
    rng = np.random.default_rng(seed)
    X = block[["x", "y"]].to_numpy()
    n = len(X)
    layer = block["deep"].to_numpy()
    truth_safe = block["as_ppb"].to_numpy() <= thr
    m0, sd0 = prior.mean_sd(block["depth_ft"].to_numpy())
    p_prior = norm.cdf((np.log10(thr) - m0) / sd0)
    nbrs = cKDTree(X).query_ball_point(X, radius)          # nbrs[j]: households a safe well j secures
    observed_safe = (lambda j: truth_safe[j]) if kit is None else (lambda j: kit(block["as_ppb"].iat[j], rng) <= thr)

    secured = np.zeros(n, bool)
    tested = np.zeros(n, bool)
    curve = []
    resid = block["log_as"].to_numpy() - m0

    def plan(p, k):
        """Greedy expected-coverage choice of k untested wells given P(safe) = p."""
        w = (~secured).astype(float)                      # expected 'still unsecured' weight per household
        chosen = []
        cand = np.flatnonzero(~tested)
        gain = np.array([p[j] * w[nbrs[j]].sum() for j in cand])
        for _ in range(min(k, len(cand))):
            i = int(np.argmax(gain))
            j = cand[i]
            chosen.append(j)
            gain[i] = -np.inf
            w[nbrs[j]] *= (1 - p[j])
            # refresh gains of wells whose neighbourhoods overlap j's (lazy: neighbours within 2*radius)
            touched = set()
            for h in nbrs[j]:
                touched.update(nbrs[h])
            for jj in touched:
                if not tested[jj] and jj not in chosen:
                    ii = np.searchsorted(cand, jj)
                    gain[ii] = p[jj] * w[nbrs[jj]].sum()
        return chosen

    if policy == "random":
        order = rng.permutation(n)[:budget]
        queue = list(order)
    elif policy == "static":
        queue = plan(p_prior, budget)
    else:
        queue = []

    while tested.sum() < budget and tested.sum() < n:
        if policy in ("adaptive", "learning"):
            if policy == "adaptive":
                p = p_prior
            else:
                t_idx, u_idx = np.flatnonzero(tested), np.flatnonzero(~tested)
                p = p_prior.copy()
                if len(t_idx):
                    mu, var = gp.posterior(X, layer, resid, t_idx, u_idx)
                    p[u_idx] = norm.cdf((np.log10(thr) - m0[u_idx] - mu) / np.sqrt(var + gp.nugget))
            queue = plan(p, min(batch, budget - int(tested.sum())))
        step = queue[:batch] if policy in ("adaptive", "learning") else queue[int(tested.sum()):int(tested.sum()) + batch]
        if not step:
            break
        for j in step:
            tested[j] = True
            if observed_safe(j) and truth_safe[j]:        # kit says safe and it really is: household moves safely
                secured[nbrs[j]] = True
            curve.append(secured.sum())
    return np.array(curve)
