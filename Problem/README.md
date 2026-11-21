# Problem: instance generator

Generates the instances of the multi-facility load balancing problem: 11 configurations × 6 values of α × 10 instances
(660 in total). The package is self-contained. It needs only `numpy`, plus `numba`, which is optional but strongly
recommended.

```
python -m Problem.generate_instances                       # all 660 -> instances/<case>_a<alpha>_run<run>.json
python -m Problem.generate_instances --cases 5x6 --alphas 0.6 --runs 0 1 --out-dir my_dir
```

Run the commands from the folder that contains `Problem/`. In Python:

```python
from Problem import generate, Instance
instance = generate('10x12', run=3, alpha=0.4)
instance.save('10x12_a0.4_run3.json'); instance = Instance.load('10x12_a0.4_run3.json')
```

Generating all 660 instances takes about 4 minutes with numba, and the JSON files take about 34 MB.

## Files

| file | content |
|---|---|
| `configurations.py` | the 11 configurations, every generation parameter, the seeds |
| `generator.py` | `generate(case, run, alpha)`: random draws, capacities, critical resources, buffer extension |
| `construction.py` | the constructive heuristic, used only by the buffer extension rule |
| `instance.py` | `Instance`, the JSON format, the piecewise linear approximation and the coefficients of the two objectives |
| `generate_instances.py` | command line: writes the instances |

## Configurations

| case | \|F\| | \|T_R\| | \|T_B\| (before extension) |
|---|---|---|---|
| 5x6, 10x6, 15x6 | 5, 10, 15 | 6 | 1 |
| 5x12, 10x12, 15x12 | 5, 10, 15 | 12 | 2 |
| 5x18, 10x18, 15x18 | 5, 10, 15 | 18 | 3 |
| 20x24 | 20 | 24 | 4 |
| 20x48 | 20 | 48 | 8 |

α ∈ {0, 0.2, 0.4, 0.6, 0.8, 1}, runs 0–9.

## Generation

Time unit: 15 minutes. Capacities and processing times are in this unit (a capacity of 45 h is 180).

| element | rule | code |
|---|---|---|
| resources per facility | 1 + Binomial(4, 0.5) | `RESOURCES_BINOMIAL` |
| critical resources | the first k of each facility (all if fewer), k ∈ {1, 2, 3} per instance | `CRITICAL_CHOICES` |
| capacity | 45 h per period; in 10% of the (resource, period) pairs ⌊45·U(0.5, 0.9)⌋ h, regular periods and buffer | `NOMINAL_CAPACITY_HOURS`, `CAPACITY_REDUCTION` |
| target | 80% of the capacity of each period | `TARGET_SHARE` |
| job types | 100, uniform | `N_JOB_TYPES` |
| processing time, first resource | {105, …, 165} min with probabilities (0.1, 0.2, 0.4, 0.2, 0.1) | `FIRST_RESOURCE_TIMES` (7–11 units) |
| processing time, other resources | {90, …, 135} min with probabilities (0.1, 0.6, 0.2, 0.1) | `OTHER_RESOURCE_TIMES` (6–9 units) |
| jobs | due in each regular period: U{15, …, 30} × \|F\|; none due in the buffer | `JOBS_PER_PERIOD` |
| maximum delay D_j | uniform in {1, 2, 3, 4} | `MAX_DELAY_VALUES` |
| buffer | round(\|T_R\|/6) periods | `buffer_periods` |
| piecewise linear approximation | breakpoints 0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1 | `PLA_BREAKPOINTS` |
| buffer extension | one period at a time (capacities of the last one) until the construction places every job | `generator.generate`, `construction.places_every_job` |

**Seeds** (instance `run`, the same for every configuration and α): `numpy.random.RandomState(41 + run²)` draws the
resources, the job types, the jobs and the maximum delays, in that order. `default_rng(run)` draws the reduced capacities,
and `default_rng(10000 + run)` draws k.


## JSON format

```
case, run, alpha, time_unit_minutes (15), target_share (0.8), pla_breakpoints,
periods: {regular: T_R, buffer: T_B (extension included), buffer_extension},
critical_per_facility: k,
facilities: [{capacity: [K_i][T] int, job_type_times: [100][K_i] int}],
jobs: {type: [J], due: [J], max_delay: [J]}
```

The processing time of job j on resource k of facility i is `job_type_times[type[j]][k]` of facility i. A resource is
critical if its index is smaller than `critical_per_facility`, and its target in period t is `0.8 · capacity[k][t]`.
