"""Generation of instance `run` of a configuration for a weight alpha.

The random draws depend only on the configuration and on `run`.
The constructive heuristic must place every job, otherwise the buffer is extended by one period
(capacities of the last period) and the heuristic is run again.
"""
import numpy as np

from . import configurations as cfg
from .construction import places_every_job
from .instance import Instance


def generate(case, run, alpha):
    F, R = cfg.CASES[case]
    B = cfg.buffer_periods(R)
    T = R + B
    rs = np.random.RandomState(cfg.instance_seed(run))

    # resources per facility: 1 + Binomial(4, 0.5)
    n_res = rs.binomial(*cfg.RESOURCES_BINOMIAL, size=F) + 1

    # processing time of each job type on each resource of each facility
    type_times = [np.zeros((cfg.N_JOB_TYPES, n), dtype=np.int64) for n in n_res]
    for n in range(cfg.N_JOB_TYPES):
        for i in range(F):
            for k in range(n_res[i]):
                values, p = cfg.FIRST_RESOURCE_TIMES if k == 0 else cfg.OTHER_RESOURCE_TIMES
                type_times[i][n, k] = int(rs.choice(list(values), p=list(p)))

    # jobs: |F| * U{15..30} due in each regular period, each with a job type drawn uniformly; then the maximum delays
    job_types, due = [], []
    for t in range(R):
        for _ in range(rs.randint(cfg.JOBS_PER_PERIOD[0], cfg.JOBS_PER_PERIOD[1] + 1) * F):
            job_types.append(rs.choice(cfg.N_JOB_TYPES))
            due.append(t)
    max_delay = rs.choice(list(cfg.MAX_DELAY_VALUES), size=len(due)).astype(np.int64)

    # capacities (hours, then time units): reduced in 10% of the (resource, period) pairs, regular periods and buffer
    rng = np.random.default_rng(cfg.capacity_seed(run))
    capacity = []
    for i in range(F):
        c = np.empty((n_res[i], T), dtype=np.int64)
        for k in range(n_res[i]):
            reduced = rng.random(T) < cfg.CAPACITY_REDUCTION['p']
            factor = rng.uniform(cfg.CAPACITY_REDUCTION['low'], cfg.CAPACITY_REDUCTION['high'], T)
            hours = np.where(reduced, np.floor(cfg.NOMINAL_CAPACITY_HOURS * factor), cfg.NOMINAL_CAPACITY_HOURS).astype(int)
            c[k] = hours * cfg.UNITS_PER_HOUR
        capacity.append(c)

    # number of critical resources per facility
    critical = int(np.random.default_rng(cfg.critical_seed(run)).integers(min(cfg.CRITICAL_CHOICES),
                                                                         max(cfg.CRITICAL_CHOICES) + 1))

    instance = Instance(case, run, float(alpha), R, B, 0, critical, capacity, np.array(job_types, dtype=np.int64), type_times,
                        np.array(due, dtype=np.int64), max_delay)

    # buffer extension rule
    while not places_every_job(instance):
        if instance.buffer_extension >= R:
            raise RuntimeError(f"{case} run {run}: no complete construction with {R} buffer periods added")
        instance.capacity = [np.concatenate([c, c[:, -1:]], axis=1) for c in instance.capacity]
        instance.buffer_periods += 1
        instance.buffer_extension += 1
    return instance
