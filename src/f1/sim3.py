"""Campaign simulator v5: public-health outcome (exposed households protected).

A household (well) is EXPOSED if its own well is truly unsafe. Testing well j has two possible values:
  * j reads unsafe  -> its household is INFORMED that it must switch (useful only if a safe source is within r);
  * j reads safe    -> j becomes a SOURCE that informed neighbours within r can switch to.
An exposed household is PROTECTED once it is informed (own well tested, kit unsafe) and a tested well within r reads
safe and is truly safe. It is MISLED if its own unsafe well read safe (false reassurance), or if every source it
was sent to is truly unsafe.

All targeted policies score a candidate well j by one-step expected gain per minute
   g_j = p_j (E_j + lam U_j) + (1 - p_j) (S_j + lam Q_j),
   E_j = informed, not-yet-served neighbours that j would serve if safe
   S_j = P(a source already exists within r of j)            (j's household is served at once if j is unsafe)
   U_j = untested neighbours without a source that are likely unsafe (future households j could serve)
   Q_j = P(at least one untested neighbour of j is safe)      (future source for j's household)
using beliefs that differ by policy:
   static     p from depth prior; outcomes of earlier tests replaced by their expectations (never looks at results)
   geometry   p constant (no depth, no results about safety) but observed outcomes used: pure re-planning
   adaptive   p from depth prior, observed outcomes
   learning   p from a learner updated with every result, observed outcomes
   oracle     p = true status (reference ceiling)
   sweep      nearest untested well (door-to-door)
"""
import numpy as np
from sim2 import Field

EPS = 1e-9


def run3(camp, policy, learner, rng, fieldp=Field(), days=10, lam=0.5, tertile=None):
    n = camp.n
    A = camp.A
    exposed = ~camp.truth_safe
    tested = np.zeros(n, bool)
    kit_safe = camp.kit_safe
    src_any = np.zeros(n, bool)       # a tested well within r reads safe
    src_true = np.zeros(n, bool)      # ... and is truly safe
    logno = np.zeros(n)               # sum over tested k within r of log(1 - q_k): belief that no source exists
    q_exp = np.zeros(n)               # static: expected 'reads safe' of tested wells
    known = camp.known
    p_base = camp.p_prior.copy()
    if len(known):
        p_base[known] = camp.obs_safe_known
    p_const = np.full(n, float(np.mean(camp.p_prior)))
    pos = camp.X[rng.integers(n)]
    minutes = 0.0
    limit = days * fieldp.day_min
    out = {"minutes": [], "protected": [], "misled": [], "informed": [], "access": []}
    if tertile is not None:
        for k in range(3):
            out[f"prot_t{k}"] = []
    while True:
        cand = np.flatnonzero(~tested)
        if len(cand) == 0:
            break
        walk = np.sqrt(((camp.X[cand] - pos) ** 2).sum(1)) * fieldp.detour / fieldp.walk_m_per_min
        cost = fieldp.test_min + walk
        if policy == "sweep":
            jc = int(np.argmin(walk + rng.uniform(0, 1e-6, len(cand))))
        else:
            if policy == "learning":
                info = np.union1d(np.flatnonzero(tested), known)
                info_c = np.setdiff1d(cand, info)
                p_c = p_base[cand].copy()
                if len(info_c):
                    p_c[np.searchsorted(cand, info_c)] = learner.predict(camp, info, info_c)
                p = p_base.copy()
                p[cand] = p_c
            elif policy == "oracle":
                p = camp.truth_safe.astype(float)
            elif policy == "geometry":
                p = p_const
            else:
                p = p_base
            if policy == "static":
                q = np.where(tested, q_exp, 0.0)
                informed = np.where(tested, 1 - q_exp, 0.0)
            else:
                q = np.where(tested, kit_safe.astype(float), 0.0)
                informed = (tested & ~kit_safe).astype(float)
            s = 1 - np.exp(logno)                                   # P(source within r)
            untested = (~tested).astype(float)
            pc = np.clip(p, EPS, 1 - EPS)
            E = np.asarray(A[cand] @ (informed * (1 - s))).ravel()
            U = np.asarray(A[cand] @ (untested * (1 - s) * (1 - pc))).ravel() - (1 - s[cand]) * (1 - pc[cand])
            lq = np.asarray(A[cand] @ (untested * np.log(1 - pc))).ravel() - np.log(1 - pc[cand])
            Q = 1 - np.exp(lq)
            S = s[cand]
            pj = pc[cand]
            g = pj * (E + lam * U) + (1 - pj) * (S + lam * Q)
            jc = int(np.argmax(g / cost + rng.uniform(0, 1e-12, len(cand))))
        j = cand[jc]
        if minutes + cost[jc] > limit:
            break
        minutes += cost[jc]
        pos = camp.X[j]
        tested[j] = True
        nb = A[j].indices
        camp.obs_safe[j], camp.obs_log[j] = camp.kit_safe[j], camp.kit_log[j]
        if policy == "static":
            pj = p_base[j]
            q_exp[j] = pj
            logno[nb] += np.log(max(1 - pj, EPS))
        elif kit_safe[j]:
            logno[nb] += np.log(EPS)
        if kit_safe[j]:
            src_any[nb] = True
            if camp.truth_safe[j]:
                src_true[nb] = True
        inf_unsafe = tested & ~kit_safe
        prot = exposed & inf_unsafe & src_true
        mis = exposed & ((tested & kit_safe) | (inf_unsafe & src_any & ~src_true))
        out["minutes"].append(minutes)
        out["protected"].append(prot.sum())
        out["misled"].append(mis.sum())
        out["informed"].append((exposed & inf_unsafe).sum())
        out["access"].append(src_true.sum())
        if tertile is not None:
            for k in range(3):
                out[f"prot_t{k}"].append((prot & (tertile == k)).sum())
    return {k: np.array(v) for k, v in out.items()}, int(exposed.sum())
