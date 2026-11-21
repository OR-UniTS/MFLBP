"""An instance of the multi-facility load balancing problem (MFLBP), with its JSON format.

Time unit: 15 minutes. Facilities i = 0..F-1 have resources k = 0..K_i-1; the first `critical` ones of each facility (all of
them if it has fewer) are critical. Periods t = 0..T-1: regular periods 0..T_R-1, then the buffer T_R..T-1.
"""
import json
from dataclasses import dataclass

import numpy as np

from .configurations import PLA_BREAKPOINTS, TARGET_SHARE, UNITS_PER_HOUR

_BREAKPOINTS = np.array(PLA_BREAKPOINTS, dtype=float)


def pla_value(x, scale):
    """piecewise linear approximation of (x / scale)^2: interpolated between the breakpoints, linear with the
    last slope beyond the scale"""
    u = np.asarray(x, dtype=float) / scale
    return np.interp(u, _BREAKPOINTS, _BREAKPOINTS ** 2) + np.maximum(0, u - 1) * (_BREAKPOINTS[-2] + _BREAKPOINTS[-1])


@dataclass
class Instance:
    case: str
    run: int
    alpha: float
    regular_periods: int  # T_R
    buffer_periods: int  # T_B, extension included
    buffer_extension: int  # periods added to the buffer by the extension rule
    critical: int  # k: the first k resources of each facility are critical
    capacity: list  # capacity[i]: int array (K_i, T), time units
    job_types: np.ndarray  # job type of each job (J,)
    type_times: list  # type_times[i]: int array (N_JOB_TYPES, K_i), processing time of each job type on each resource of i
    due: np.ndarray  # due period of each job (J,)
    max_delay: np.ndarray  # maximum tolerated delay D_j of each job (J,)

    @property
    def periods(self):
        return self.regular_periods + self.buffer_periods

    @property
    def num_facilities(self):
        return len(self.capacity)

    @property
    def num_jobs(self):
        return len(self.due)

    def is_critical(self, i, k):
        return k < self.critical

    def num_critical(self, i):
        return min(self.critical, len(self.capacity[i]))

    def target(self, i, k):
        """target workload of critical resource k of facility i in each period (time units)"""
        return self.capacity[i][k] * TARGET_SHARE

    def processing_times(self, i):
        """int array (J, K_i): processing time of each job on each resource of facility i"""
        return self.type_times[i][self.job_types]

    def load_coefficient(self):
        """alpha / (number of (critical resource, period) pairs)"""
        return self.alpha / (sum(self.num_critical(i) for i in range(self.num_facilities)) * self.periods)

    def delay_coefficient(self):
        """(1 - alpha) / number of jobs"""
        return (1 - self.alpha) / self.num_jobs

    def to_dict(self):
        return {
            'case': self.case, 'run': self.run, 'alpha': self.alpha,
            'time_unit_minutes': 60 // UNITS_PER_HOUR,
            'periods': {'regular': self.regular_periods, 'buffer': self.buffer_periods,
                        'buffer_extension': self.buffer_extension},
            'critical_per_facility': self.critical,
            'target_share': TARGET_SHARE,
            'pla_breakpoints': list(PLA_BREAKPOINTS),
            'facilities': [{'capacity': c.tolist(), 'job_type_times': tt.tolist()}
                           for c, tt in zip(self.capacity, self.type_times)],
            'jobs': {'type': self.job_types.tolist(), 'due': self.due.tolist(), 'max_delay': self.max_delay.tolist()},
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d['case'], d['run'], d['alpha'], d['periods']['regular'], d['periods']['buffer'],
                   d['periods']['buffer_extension'], d['critical_per_facility'],
                   [np.array(f['capacity'], dtype=np.int64) for f in d['facilities']],
                   np.array(d['jobs']['type'], dtype=np.int64),
                   [np.array(f['job_type_times'], dtype=np.int64) for f in d['facilities']],
                   np.array(d['jobs']['due'], dtype=np.int64), np.array(d['jobs']['max_delay'], dtype=np.int64))

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, separators=(',', ':'))

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))
