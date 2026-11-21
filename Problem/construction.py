"""Constructive heuristic, used here only by the buffer extension rule: the buffer is extended by
one period until the heuristic places every job for some fill-level offset theta.
"""
import numpy as np

from .configurations import CONSTRUCTION_OFFSETS
from .instance import pla_value

try:
    from numba import njit
except ImportError:  # same code in pure Python (much slower on the large configurations)
    def njit(*args, **kwargs):
        return args[0] if len(args) == 1 and callable(args[0]) else (lambda f: f)


def construction_data(instance):
    """arrays of construct(): padded resources (capacity never binding, processing time 0)"""
    F, J, T, R = instance.num_facilities, instance.num_jobs, instance.periods, instance.regular_periods
    K = max(len(c) for c in instance.capacity)
    C = max(instance.num_critical(i) for i in range(F))

    cap = np.full((F, K, T), np.iinfo(np.int64).max // 4, dtype=np.int64)
    P = np.zeros((J, F, K), dtype=np.int64)
    crit = np.full((F, C), -1, dtype=np.int64)
    ctab = np.full((F, C, T), -1, dtype=np.int64)
    target = np.zeros((F, C, T))
    crit_cap = np.ones((F, C, T))
    tables = {}  # (target, scale, buffer period) -> index of its load cost table
    for i in range(F):
        n = len(instance.capacity[i])
        cap[i, :n] = instance.capacity[i]
        P[:, i, :n] = instance.processing_times(i)
        for q in range(instance.num_critical(i)):
            w = instance.target(i, q)
            crit[i, q] = q
            target[i, q] = w
            crit_cap[i, q] = cap[i, q]
            scale = cap[i, q] - w
            for t in range(T):
                ctab[i, q, t] = tables.setdefault((w[t], scale[t], t >= R), len(tables))

    rem_cap = np.cumsum(crit_cap[:, :, ::-1], axis=2)[:, :, ::-1].copy()
    rem_target = np.cumsum(target[:, :, ::-1], axis=2)[:, :, ::-1]
    w_hat = (rem_target / rem_cap).max(axis=1).mean(axis=0)
    last_fit = np.full((J, F), -1, dtype=np.int64)
    for t in range(T):
        last_fit[(P <= cap[None, :, :, t]).all(axis=2)] = t

    # load cost of a critical resource as a function of its load y (only above target in the buffer)
    y = np.arange(max(int(cap[i, :instance.num_critical(i)].max()) for i in range(F)) + 1)
    G = np.empty((len(tables), len(y)))
    for h, (w, scale, buffer) in enumerate(tables):
        G[h] = instance.load_coefficient() * pla_value(np.maximum(0, y - w) if buffer else np.abs(y - w), scale)

    # delay cost of job j in period t
    delay = np.maximum(0, np.arange(T)[None, :] - instance.due[:, None])
    Dly = instance.delay_coefficient() * pla_value(delay, instance.max_delay[:, None])
    return P, cap, crit, ctab, G, Dly, instance.due.astype(np.int64), target, rem_cap, w_hat, last_fit


def places_every_job(instance):
    """the constructive heuristic places every job for at least one offset theta"""
    data = construction_data(instance)
    return any(construct(*data, float(theta))[2] for theta in CONSTRUCTION_OFFSETS)


@njit(cache=True)
def construct(P, cap, crit, ctab, G, Dly, due, target, rem_cap, w_hat, last_fit, offset):
    """period by period; every critical resource is filled up to the level target + lam_t * (capacity - target) of period t
    (lam_t at least offset), trading load deviation for delay. Returns the facility and period of each job and whether every
    job was placed."""
    J, F, K = P.shape
    C = crit.shape[1]
    T = cap.shape[2]
    L = np.zeros((F, K, T), dtype=np.int64)
    fac = np.full(J, -1, dtype=np.int64)
    per = np.full(J, -1, dtype=np.int64)
    by_due = np.argsort(due, kind='mergesort')
    left = J
    for t in range(T):
        # jobs that can be placed in t, the most urgent first (ties: earliest due period)
        cand = np.empty(left, dtype=np.int64)
        wait = np.empty(left)
        n = 0
        for j in by_due:
            if fac[j] < 0:
                cand[n] = j
                wait[n] = Dly[j, t + 1] - Dly[j, t] if t + 1 < T else np.inf
                n += 1
        prio = np.argsort(-wait[:n], kind='mergesort')
        # share of the remaining capacity needed by the jobs still to place, each in its best facility
        need = 0.0
        for q in range(n):
            j = cand[q]
            best = np.inf
            for i in range(F):
                if last_fit[j, i] >= t:
                    p = 0.0
                    for r in range(C):
                        c = crit[i, r]
                        if c < 0:
                            break
                        p = max(p, P[j, i, c] / rem_cap[i, r, t])
                    best = min(best, p)
            if best < np.inf:
                need += best
        lam = max(offset, (need / F - w_hat[t]) / (1 - w_hat[t]))
        for q in prio:
            j = cand[q]
            best_i, best_p, best_dg = -1, np.inf, np.inf
            alt_i, alt_dg = -1, np.inf
            for i in range(F):
                feasible = True
                for k in range(K):
                    if L[i, k, t] + P[j, i, k] > cap[i, k, t]:
                        feasible = False
                        break
                if not feasible:
                    continue
                within, p, dg = True, 0.0, 0.0
                for r in range(C):
                    c = crit[i, r]
                    if c < 0:
                        break
                    y = L[i, c, t]
                    p = max(p, P[j, i, c] / cap[i, c, t])
                    h = ctab[i, r, t]
                    dg = dg + G[h, y + P[j, i, c]] - G[h, y]
                    w = target[i, r, t]
                    if y + P[j, i, c] > w + lam * (cap[i, c, t] - w) + 1e-9:
                        within = False
                if within and (p < best_p or (p == best_p and dg < best_dg)):
                    best_i, best_p, best_dg = i, p, dg
                if dg < alt_dg:
                    alt_i, alt_dg = i, dg
            # beyond the level only if waiting costs more (always in the last period)
            if best_i < 0 and alt_i >= 0 and alt_dg < wait[q]:
                best_i = alt_i
            if best_i >= 0:
                for k in range(K):
                    L[best_i, k, t] += P[j, best_i, k]
                fac[j], per[j] = best_i, t
                left -= 1

    # jobs left out: cheapest feasible period of the horizon
    for j in by_due:
        if fac[j] >= 0:
            continue
        best_i, best_t, best = -1, -1, np.inf
        for t in range(T):
            for i in range(F):
                feasible = True
                for k in range(K):
                    if L[i, k, t] + P[j, i, k] > cap[i, k, t]:
                        feasible = False
                        break
                if feasible:
                    d = 0.0
                    for r in range(C):
                        c = crit[i, r]
                        if c < 0:
                            break
                        d = d + G[ctab[i, r, t], L[i, c, t] + P[j, i, c]] - G[ctab[i, r, t], L[i, c, t]]
                    d = d + Dly[j, t]
                    if d < best:
                        best_i, best_t, best = i, t, d
        if best_i < 0:
            return fac, per, False
        for k in range(K):
            L[best_i, k, best_t] += P[j, best_i, k]
        fac[j], per[j] = best_i, best_t
    return fac, per, True
