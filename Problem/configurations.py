"""The eleven instance configurations and the parameters they share.

Time unit: 15 minutes (a quarter of an hour). Capacities are drawn in hours and converted (x4); processing times are drawn
directly in quarters of an hour (7..11 -> 105..165 minutes, 6..9 -> 90..135 minutes).
"""

# |F| x |T_R|: number of facilities and of regular periods. The buffer is round(|T_R| / 6) periods before any extension.
CASES = {
    '5x6': (5, 6), '5x12': (5, 12), '5x18': (5, 18),
    '10x6': (10, 6), '10x12': (10, 12), '10x18': (10, 18),
    '15x6': (15, 6), '15x12': (15, 12), '15x18': (15, 18),
    '20x24': (20, 24), '20x48': (20, 48),
}
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)  # weight of the load term; 1 - alpha weighs the delay term
RUNS = tuple(range(10))  # ten instances per configuration and alpha: 11 x 6 x 10 = 660 runs

UNITS_PER_HOUR = 4  # time unit: 15 minutes
N_JOB_TYPES = 100
NOMINAL_CAPACITY_HOURS = 45  # every resource, every period
TARGET_SHARE = 0.8  # target workload of a critical resource: 80% of its capacity in each period

# resources per facility: 1 + Binomial(4, 0.5); the first k are critical (all of them if fewer), k uniform in {1, 2, 3}
RESOURCES_BINOMIAL = (4, 0.5)
CRITICAL_CHOICES = (1, 2, 3)

# processing time of a job type on a resource (time units, with probabilities): first resource of a facility / the others
FIRST_RESOURCE_TIMES = ((7, 8, 9, 10, 11), (0.1, 0.2, 0.4, 0.2, 0.1))
OTHER_RESOURCE_TIMES = ((6, 7, 8, 9), (0.1, 0.6, 0.2, 0.1))

JOBS_PER_PERIOD = (15, 30)  # jobs due in each regular period: uniform in {15, ..., 30}, times |F|; none due in the buffer
MAX_DELAY_VALUES = (1, 2, 3, 4)  # maximum tolerated delay D_j of a job (periods), uniform

# reduced capacity: in each (resource, period) pair with probability 0.1, floor(45 * U(0.5, 0.9)) hours
CAPACITY_REDUCTION = {'p': 0.1, 'low': 0.5, 'high': 0.9}

# breakpoints of the piecewise linear approximation of u^2, as fractions of the scale
PLA_BREAKPOINTS = (0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1)

# fill-level offsets theta tried by the constructive heuristic in the buffer extension rule
CONSTRUCTION_OFFSETS = tuple(i / 9 for i in range(10))


# seeds of instance `run` (the same for every configuration and alpha)
def instance_seed(run):
    """job types, resources per facility, jobs, due periods and maximum delays (numpy legacy MT19937)"""
    return 41 + run ** 2


def capacity_seed(run):
    """reduced capacities (numpy default_rng)"""
    return run


def critical_seed(run):
    """number k of critical resources per facility (numpy default_rng)"""
    return 10_000 + run


def buffer_periods(regular_periods):
    return round(regular_periods / 6)
