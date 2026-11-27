"""Campaign simulator v6 (version used in the article).

Same outcome and score as v5 (sim3.py) plus:
  * per-policy look-ahead weight lam;
  * 'uncertainty' policy (variance-driven exploration, adaptive-design style);
  * metrics computed at fixed minute marks: protected, misled split into (a) own unsafe well read safe and
    (b) directed only to wells that read safe but are unsafe, false alarms (truly safe own well read unsafe),
    expected switchers with distance decay, protected under a capacity limit per safe well, equity tertiles;
  * field realism options: daily return to a base well, and owners absent with probability q (visit fails,
    time is spent, the well can be retried on a later day).
"""
import numpy as np
from sim2 import Field

EPS = 1e-9
DECAY_M = 100 / np.log(1 / 0.3)          # P(switch) = exp(-d / DECAY_M): 1 at 0 m, 0.3 at 100 m


def _capacity_protected(camp, cand_h, src, cap):
    """Greedy nearest-first assignment of informed exposed households to truly safe sources (<= cap each)."""
    if len(cand_h) == 0 or len(src) == 0:
        return 0
    sub = camp.D[cand_h][:, src].tocoo()
    if sub.nnz == 0:
        return 0
    order = np.argsort(sub.data)
    load = np.zeros(len(src), int)
    done = np.zeros(len(cand_h), bool)
    n = 0
    for k in order:
        h, s = sub.row[k], sub.col[k]
        if not done[h] and load[s] < cap:
            done[h] = True
            load[s] += 1
            n += 1
    return n


def run4(camp, policy, learner, rng, fieldp=Field(), days=10, lam=0.1, tertile=None, marks_days=(1, 5, 10),
         cap=5, daily_return=False, absent=0.0):
    n = camp.n
    A, D = camp.A, camp.D
    exposed = ~camp.truth_safe
    tested = np.zeros(n, bool)
    kit_safe = camp.kit_safe
    src_any = np.zeros(n, bool)
    src_true = np.zeros(n, bool)
    best_d = np.full(n, np.inf)               # distance to the nearest truly safe source that read safe
    logno = np.zeros(n)
    q_exp = np.zeros(n)
    blocked_until = np.full(n, -1)            # owner absent: unavailable until this day index
    known = camp.known
    p_base = camp.p_prior.copy()
    if len(known):
        p_base[known] = camp.obs_safe_known
    p_const = np.full(n, float(np.mean(camp.p_prior)))
    base = int(np.argmin(((camp.X - camp.X.mean(0)) ** 2).sum(1)))
    pos = camp.X[base] if daily_return else camp.X[rng.integers(n)]
    minutes, day = 0.0, 0
    limit = days * fieldp.day_min
    marks = [m * fieldp.day_min for m in sorted(marks_days) if m <= days]
    out = []

    def record(mark_day):
        inf_unsafe = tested & ~kit_safe
        prot = exposed & inf_unsafe & src_true
        mis_own = exposed & tested & kit_safe
        mis_src = exposed & inf_unsafe & src_any & ~src_true
        false_alarm = (~exposed) & tested & ~kit_safe
        cand = np.flatnonzero(exposed & inf_unsafe & np.isfinite(best_d))
        src = np.flatnonzero(tested & kit_safe & camp.truth_safe)
        r = dict(days=mark_day, tests=int(tested.sum()), protected=int(prot.sum()), misled_own=int(mis_own.sum()),
                 misled_src=int(mis_src.sum()), false_alarm=int(false_alarm.sum()),
                 informed=int((exposed & inf_unsafe).sum()), access=int(src_true.sum()),
                 switch=float(np.exp(-best_d[prot] / DECAY_M).sum()),
                 prot_cap=_capacity_protected(camp, cand, src, cap))
        if tertile is not None:
            for k in range(3):
                r[f"prot_t{k}"] = int((prot & (tertile == k)).sum())
        out.append(r)

    mi = 0
    while mi < len(marks):
        cur_day = int(minutes // fieldp.day_min)
        if daily_return and cur_day > day:
            pos = camp.X[base]
        day = cur_day
        avail = (~tested) & (blocked_until < day)
        cand = np.flatnonzero(avail)
        if len(cand) == 0:
            if (~tested).any():                   # everyone left is absent today: go to next day
                minutes = (day + 1) * fieldp.day_min
                while mi < len(marks) and minutes >= marks[mi]:
                    record(marks[mi] / fieldp.day_min); mi += 1
                continue
            break
        walk = np.sqrt(((camp.X[cand] - pos) ** 2).sum(1)) * fieldp.detour / fieldp.walk_m_per_min
        cost = fieldp.test_min + walk
        if policy == "sweep":
            jc = int(np.argmin(walk + rng.uniform(0, 1e-6, len(cand))))
        else:
            if policy in ("learning", "uncertainty"):
                info = np.union1d(np.flatnonzero(tested), known)
                untested_all = np.flatnonzero(~tested)
                info_c = np.setdiff1d(untested_all, info)
                p = p_base.copy()
                if len(info_c):
                    p[info_c] = learner.predict(camp, info, info_c)
            elif policy == "oracle":
                p = camp.truth_safe.astype(float)
            elif policy == "geometry":
                p = p_const
            else:
                p = p_base
            pc = np.clip(p, EPS, 1 - EPS)
            untested = (~tested).astype(float)
            if policy == "uncertainty":
                nb_untested = np.asarray(A[cand] @ untested).ravel()
                g = pc[cand] * (1 - pc[cand]) * nb_untested
            else:
                if policy == "static":
                    informed = np.where(tested, 1 - q_exp, 0.0)
                else:
                    informed = (tested & ~kit_safe).astype(float)
                s = 1 - np.exp(logno)
                E = np.asarray(A[cand] @ (informed * (1 - s))).ravel()
                U = np.asarray(A[cand] @ (untested * (1 - s) * (1 - pc))).ravel() - (1 - s[cand]) * (1 - pc[cand])
                lq = np.asarray(A[cand] @ (untested * np.log(1 - pc))).ravel() - np.log(1 - pc[cand])
                Q = 1 - np.exp(lq)
                pj = pc[cand]
                g = pj * (E + lam * U) + (1 - pj) * (s[cand] + lam * Q)
            jc = int(np.argmax(g / cost + rng.uniform(0, 1e-12, len(cand))))
        j = cand[jc]
        nxt = minutes + cost[jc]
        while mi < len(marks) and nxt > marks[mi]:
            record(marks[mi] / fieldp.day_min); mi += 1
        if nxt > limit or mi >= len(marks):
            break
        minutes = nxt
        pos = camp.X[j]
        if absent > 0 and rng.uniform() < absent:     # owner absent: time spent, retry on a later day
            blocked_until[j] = day
            continue
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
                dj = np.asarray(D[j, nb].todense()).ravel()
                dj[nb == j] = 0.0
                best_d[nb] = np.minimum(best_d[nb], dj)
    while mi < len(marks):
        record(marks[mi] / fieldp.day_min); mi += 1
    return out, int(exposed.sum())
