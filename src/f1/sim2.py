"""Well-testing campaign simulator, version 2 (field-time budget, kit error, pluggable learners).

A campaign is one spatial block. Every well stands for the household(s) drinking from it. A field team starts
at a random well, walks to wells, tests them, and sees each result immediately (field kits read in minutes).
A household is
  believed secured  if some tested well within `radius` read safe on the kit (the team tells them to switch);
  truly secured     if such a well is also safe by the lab value;
  misled            if believed secured but not truly secured.

Policies form an ablation ladder; all use the same score  p_j * (unsecured households near j) / (minutes to test j):
  sweep     test the nearest untested well (door-to-door blanket testing, no ranking)
  static    plan with the depth prior, discounting expected coverage, never looking at results
  adaptive  as static, but uses the observed results to know which households are already secured
  learning  as adaptive, and also updates P(safe) of untested wells with a learner fed by the results
  random    (count mode only) test wells in random order
"""
import os
from dataclasses import dataclass, field
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import norm

LOG_FLOOR = 0.0          # log10(1 ppb)


@dataclass
class Field:
    test_min: float = 10.0      # minutes at a well: find owner, sample, kit reading, record
    walk_m_per_min: float = 50.0  # 3 km/h
    detour: float = 1.3         # path length / straight-line distance
    day_min: float = 360.0      # productive minutes per team-day


class KitModel:
    """Draw a kit category for a lab value from the k nearest (in log lab value) of the 944 paired wells."""

    def __init__(self, kit_df, k=25):
        self.lab = np.log10(kit_df["lab"].clip(lower=0.1).to_numpy())
        self.kit = kit_df["kit"].to_numpy().astype(float)
        self.order = np.argsort(self.lab)
        self.lab_sorted = self.lab[self.order]
        self.k = k

    def sample(self, as_ppb, rng):
        q = np.log10(np.clip(as_ppb, 0.1, None))
        out = np.empty(len(q))
        for i, v in enumerate(q):
            pos = np.searchsorted(self.lab_sorted, v)
            lo, hi = max(0, pos - self.k), min(len(self.lab_sorted), pos + self.k)
            cand = np.arange(lo, hi)
            near = cand[np.argsort(np.abs(self.lab_sorted[cand] - v))[: self.k]]
            out[i] = self.kit[self.order[rng.choice(near)]]
        return out


class Campaign:
    def __init__(self, blk, prior, radius, thr, rng, kit=None, advise_frac=1.0, known_frac=0.0, known_noise=0.0,
                 known_p=None):
        self.blk = blk
        self.X = blk[["x", "y"]].to_numpy()
        self.n = len(self.X)
        self.layer = blk["deep"].to_numpy()
        self.radius, self.thr = radius, thr
        self.m0, self.sd0 = prior.mean_sd(blk["depth_ft"].to_numpy())
        self.p_prior = norm.cdf((np.log10(thr) - self.m0) / self.sd0)
        lab = blk["as_ppb"].to_numpy().astype(float)
        self.truth_safe = lab <= thr
        if "kit_obs" in blk:                             # recorded field-kit reading is the observation
            reading = blk["kit_obs"].to_numpy().astype(float)
        elif kit is None:
            reading = lab
        else:
            reading = kit.sample(lab, rng)
        self.kit_safe = reading <= thr * advise_frac   # team advises switching only below this reading
        zero_as = float(os.environ.get("F1_ZERO_AS", "1"))   # value given to kit readings of 0 (sensitivity analysis)
        self.kit_log = np.log10(np.clip(np.where(reading == 0, zero_as, reading), 1, None))
        # what learners may use: kit results of tested wells (filled in by run) and old results of known wells
        self.obs_safe = self.kit_safe.copy()
        self.obs_log = self.kit_log.copy()
        self.tree = cKDTree(self.X)
        pairs = self.tree.query_pairs(radius, output_type="ndarray")
        i = np.r_[pairs[:, 0], pairs[:, 1], np.arange(self.n)]
        j = np.r_[pairs[:, 1], pairs[:, 0], np.arange(self.n)]
        self.A = sparse.csr_matrix((np.ones(len(i)), (i, j)), shape=(self.n, self.n))
        dist = np.sqrt(((self.X[i] - self.X[j]) ** 2).sum(1))
        self.D = sparse.csr_matrix((dist + 1e-9, (i, j)), shape=(self.n, self.n))   # same pattern as A
        self.can_secure = np.asarray(self.A @ self.truth_safe.astype(float)).ravel() > 0
        # earlier survey: a random share of wells whose old laboratory value is known to the planner (risk map),
        # but which must be re-tested before any household can be advised
        if known_frac < 0 and "old_as" in blk:          # real earlier survey: wells with an old measurement
            self.known = np.flatnonzero(blk["old_as"].notna().to_numpy())
        else:
            self.known = np.sort(rng.choice(self.n, int(known_frac * self.n), replace=False)) if known_frac > 0 \
                else np.array([], int)
        if len(self.known):
            if known_frac < 0:
                old = blk["old_as"].to_numpy()[self.known].astype(float)
            else:
                # old values: today's lab value times a log-normal change (sd in log10 units) since the old survey
                old = lab[self.known] * 10 ** rng.normal(0, known_noise, len(self.known))
            self.obs_log[self.known] = np.log10(np.clip(old, 1, None))
            old_safe = old <= thr
            self.obs_safe[self.known] = old_safe
            if known_p is None:
                self.obs_safe_known = old_safe
            else:                                        # calibrated P(safe now | old status)
                self.obs_safe_known = np.where(old_safe, known_p[0], known_p[1])


# ---------------------------------------------------------------- learners: P(safe) for untested wells
class PriorLearner:
    name = "prior"

    def fit_block(self, camp):
        pass

    def predict(self, camp, tested, untested):
        return camp.p_prior[untested]


@dataclass
class GPLearner:
    sill: float
    range_m: float
    nugget: float
    name: str = "gp"

    def fit_block(self, camp):
        pass

    def predict(self, camp, tested, untested):
        p = camp.p_prior[untested].copy()
        if len(tested) == 0:
            return p
        X, layer = camp.X, camp.layer
        resid = camp.obs_log - camp.m0
        for L in (0, 1):
            t = tested[layer[tested] == L]
            um = layer[untested] == L
            if len(t) == 0 or not um.any():
                continue
            u = untested[um]
            K = self._cov(X[t], X[t]) + self.nugget * np.eye(len(t))
            Ks = self._cov(X[u], X[t])
            c = np.linalg.cholesky(K)
            alpha = np.linalg.solve(c.T, np.linalg.solve(c, resid[t]))
            v = np.linalg.solve(c, Ks.T)
            mu = Ks @ alpha
            var = np.maximum(self.sill - (v ** 2).sum(0), 1e-6)
            # predictive sd: GP variance + nugget, scaled so the no-data case matches the depth prior
            scale = camp.sd0[u] / np.sqrt(self.sill + self.nugget)
            p[um] = norm.cdf((np.log10(camp.thr) - camp.m0[u] - mu) / (np.sqrt(var + self.nugget) * scale))
        return p

    def _cov(self, A, B):
        d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
        return self.sill * np.exp(-d / self.range_m)


@dataclass
class KNNLearner:
    """Shrink the depth prior towards the kit results of nearby tested wells in the same layer."""
    k: int = 8
    scale_m: float = 150.0
    strength: float = 1.0
    name: str = "knn"

    def fit_block(self, camp):
        pass

    def predict(self, camp, tested, untested):
        p = camp.p_prior[untested].copy()
        for L in (0, 1):
            t = tested[camp.layer[tested] == L]
            um = camp.layer[untested] == L
            if len(t) == 0 or not um.any():
                continue
            k = min(self.k, len(t))
            d, idx = cKDTree(camp.X[t]).query(camp.X[untested[um]], k=k)
            d, idx = d.reshape(-1, k), idx.reshape(-1, k)
            w = np.exp(-d / self.scale_m)
            s = (w * camp.obs_safe[t][idx]).sum(1)
            p[um] = (self.strength * p[um] + s) / (self.strength + w.sum(1))
        return p


K_FEAT = 8


def neighbour_features(camp, tested, untested):
    """Features of untested wells given the current tested set (same-layer nearest tested wells)."""
    m = len(untested)
    F = np.zeros((m, 6 + 2 * K_FEAT))
    F[:, 0] = camp.m0[untested]
    F[:, 1] = camp.sd0[untested]
    F[:, 2] = camp.layer[untested]
    F[:, 3] = np.log10(camp.blk["depth_ft"].to_numpy()[untested] + 1) if True else 0
    F[:, 4:] = np.nan
    resid = camp.obs_log - camp.m0
    for L in (0, 1):
        t = tested[camp.layer[tested] == L]
        um = np.flatnonzero(camp.layer[untested] == L)
        if len(t) == 0 or len(um) == 0:
            continue
        k = min(K_FEAT, len(t))
        d, idx = cKDTree(camp.X[t]).query(camp.X[untested[um]], k=k)
        d, idx = d.reshape(-1, k), idx.reshape(-1, k)
        w = 1 / (d + 20)
        F[um, 4] = (w * resid[t][idx]).sum(1) / w.sum(1)
        F[um, 5] = (w * camp.obs_safe[t][idx]).sum(1) / w.sum(1)
        F[um, 6:6 + k] = np.log10(d + 1)
        F[um, 6 + K_FEAT:6 + K_FEAT + k] = resid[t][idx]
    return F


@dataclass
class GBMLearner:
    model: object = None
    name: str = "gbm"

    def fit_block(self, camp):
        pass

    def predict(self, camp, tested, untested):
        return self.model.predict_proba(neighbour_features(camp, tested, untested))[:, 1]


def train_gbm(blocks, prior, thr, rng, kind="lgbm", reps=12, kit=None):
    """Train on fitting blocks: random tested subsets (2-30% of wells), target = lab safe.
    With `kit`, features of tested wells use simulated kit readings, as in the field."""
    Xs, ys = [], []
    for blk in blocks:
        for _ in range(reps):
            camp = Campaign(blk, prior, 100, thr, rng, kit=kit)
            f = rng.uniform(0.02, 0.30)
            perm = rng.permutation(camp.n)
            k = max(2, int(f * camp.n))
            t, u = np.sort(perm[:k]), np.sort(perm[k:])
            Xs.append(neighbour_features(camp, t, u))
            ys.append(camp.truth_safe[u])
    X, y = np.vstack(Xs), np.concatenate(ys)
    if kind == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    elif kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=1, random_state=0)
        X = np.nan_to_num(X, nan=-9)
        m.fit(X, y)
        return _NanWrap(m)
    m.fit(X, y)
    return m


class _NanWrap:
    def __init__(self, m):
        self.m = m

    def predict_proba(self, X):
        return self.m.predict_proba(np.nan_to_num(X, nan=-9))


# ---------------------------------------------------------------- campaign loop
@dataclass
class Result:
    minutes: np.ndarray
    true_sec: np.ndarray
    believed: np.ndarray
    misled: np.ndarray
    n: int
    can_secure: int
    nozone_auc: float = np.nan
    nozone_prec: float = np.nan
    nozone_rec: float = np.nan
    extra: dict = field(default_factory=dict)


DECAY_M = 100 / np.log(1 / 0.3)     # P(switch) = exp(-d / DECAY_M): 1 at own well, 0.3 at 100 m (Jameel et al. 2021)


def run(camp, policy, learner, rng, fieldp=Field(), days=None, n_tests=None, cost_aware=True, cap=None):
    """policy: sweep | static | static_map | adaptive | learning | uncertainty | oracle | random.
    cap: max households (wells) one safe well can serve (None = unlimited)."""
    n = camp.n
    tested = np.zeros(n, bool)
    believed = np.zeros(n, bool)
    true_sec = np.zeros(n, bool)
    best_d = np.full(n, np.inf)              # distance to the nearest truly-safe well the household was sent to
    load = np.zeros(n)                       # households assigned to each safe well (capacity)
    w_exp = np.ones(n)                       # static plans: expected 'still unsecured' weight
    pos = camp.X[rng.integers(n)]
    minutes, out_t, out_true, out_bel, out_mis, out_sw = 0.0, [], [], [], [], []
    limit_min = np.inf if days is None else days * fieldp.day_min
    limit_n = n if n_tests is None else n_tests
    order = rng.permutation(n) if policy == "random" else None
    p_all = camp.p_prior.copy()
    known = camp.known
    p_base = camp.p_prior.copy()               # depth prior, with the old survey result where one exists
    if len(known):
        p_base[known] = camp.obs_safe_known
    if policy == "static_map":                # advance plan from a risk map built on the earlier survey
        p_map = camp.p_prior.copy()
        rest = np.setdiff1d(np.arange(n), known)
        if len(known):
            p_map[rest] = learner.predict(camp, known, rest)
            p_map[known] = camp.obs_safe_known
    while tested.sum() < limit_n:
        cand = np.flatnonzero(~tested)
        if len(cand) == 0:
            break
        walk = np.sqrt(((camp.X[cand] - pos) ** 2).sum(1)) * fieldp.detour / fieldp.walk_m_per_min
        cost = fieldp.test_min + walk
        if policy == "random":
            j = order[int(tested.sum())]
            jc = np.searchsorted(cand, j)
        elif policy == "sweep":
            jc = int(np.argmin(walk + rng.uniform(0, 1e-6, len(cand))))
        else:
            if policy in ("learning", "uncertainty"):
                info = np.union1d(np.flatnonzero(tested), known)
                info_c = np.setdiff1d(cand, info)
                p = camp.p_prior[cand].copy()
                if len(info_c):
                    p[np.searchsorted(cand, info_c)] = learner.predict(camp, info, info_c)
                if len(known):                              # untested wells from the old survey: old result
                    kc = np.intersect1d(cand, known)
                    p[np.searchsorted(cand, kc)] = camp.obs_safe_known[np.searchsorted(known, kc)]
                p_all[cand] = p
            elif policy == "oracle":
                p = camp.truth_safe[cand].astype(float)
            elif policy == "static_map":
                p = p_map[cand]
            else:
                p = p_base[cand]
            w = w_exp if policy in ("static", "static_map") else (~believed).astype(float)
            cover = np.asarray(camp.A[cand] @ w).ravel()
            if cap is not None:
                cover = np.minimum(cover, cap)
            # 'uncertainty': adaptive-design baseline (test where the model is least sure), Chipeta et al. 2016 style
            score = p * (1 - p) * cover if policy == "uncertainty" else p * cover
            if cost_aware:
                score = score / cost
            jc = int(np.argmax(score + rng.uniform(0, 1e-9, len(cand))))
        j = cand[jc]
        if minutes + cost[jc] > limit_min:
            break
        minutes += cost[jc]
        pos = camp.X[j]
        tested[j] = True
        nb = camp.A[j].indices
        dj = np.asarray(camp.D[j, nb].todense()).ravel()
        if policy in ("static", "static_map"):
            pj = p_base[j] if policy == "static" else p_map[j]
            w_exp[nb] *= (1 - pj)
        camp.obs_safe[j], camp.obs_log[j] = camp.kit_safe[j], camp.kit_log[j]
        if camp.kit_safe[j]:
            new = nb                                        # unlimited: every household within r is told
            if cap is not None:                             # capacity: serve the nearest unsecured households first
                new = nb[~believed[nb]]
                dn = dj[~believed[nb]]
                new = new[np.argsort(dn)][: max(0, int(cap - load[j]))]
                load[j] += len(new)
            believed[new] = True
            if camp.truth_safe[j]:
                true_sec[new] = True
                dnew = np.asarray(camp.D[j, new].todense()).ravel() if len(new) else np.array([])
                dnew[new == j] = 0.0
                best_d[new] = np.minimum(best_d[new], dnew)
        out_t.append(minutes)
        out_true.append(true_sec.sum())
        out_bel.append(believed.sum())
        out_mis.append((believed & ~true_sec).sum())
        out_sw.append(np.exp(-best_d[np.isfinite(best_d)] / DECAY_M).sum())
    res = Result(np.array(out_t), np.array(out_true), np.array(out_bel), np.array(out_mis), n,
                 int(camp.can_secure.sum()))
    res.extra["switch"] = np.array(out_sw)
    # no-safe-option zones: unsecured households, P(no safe well within radius) from final P(safe)
    if policy in ("learning", "adaptive", "static"):
        cand = np.flatnonzero(~tested)
        if policy == "learning" and len(cand):
            p_all[cand] = learner.predict(camp, np.flatnonzero(tested), cand)
        p_un = np.where(tested, 0.0, p_all)
        log_none = np.asarray(camp.A @ np.log(np.clip(1 - p_un, 1e-9, 1))).ravel()
        q = np.exp(log_none)                               # P(no safe well within r | data)
        h = ~believed
        truth_none = ~camp.can_secure
        if h.sum() > 10 and 0 < truth_none[h].sum() < h.sum():
            from sklearn.metrics import roc_auc_score
            res.nozone_auc = roc_auc_score(truth_none[h], q[h])
            flag = h & (q > 0.5)
            res.nozone_prec = truth_none[flag].mean() if flag.any() else np.nan
            res.nozone_rec = flag[truth_none].mean() if truth_none.any() else np.nan
    return res
