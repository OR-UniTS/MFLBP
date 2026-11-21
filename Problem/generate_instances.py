"""Generate the instances and write them as JSON, one file per configuration, alpha and run:
<out-dir>/<case>_a<alpha>_run<run>.json

usage (from the folder that contains Problem/):
    python -m Problem.generate_instances [--cases 5x6 20x48] [--alphas 0 0.6] [--runs 0 1] [--out-dir instances]

Default: all the 660 instances (11 configurations x 6 alphas x 10 runs).
"""
import argparse
import os

from .configurations import ALPHAS, CASES, RUNS
from .generator import generate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='+', default=list(CASES), choices=list(CASES))
    parser.add_argument('--alphas', nargs='+', type=float, default=list(ALPHAS))
    parser.add_argument('--runs', nargs='+', type=int, default=list(RUNS))
    parser.add_argument('--out-dir', default='instances')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for case in args.cases:
        for alpha in args.alphas:
            for run in args.runs:
                generate(case, run, alpha).save(os.path.join(args.out_dir, f"{case}_a{alpha:.1f}_run{run}.json"))
