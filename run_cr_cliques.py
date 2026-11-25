"""
Camera-ready k-clique study: same five monitors and stopping rule as
run_cr_grid.py, with the edge-sampling k-clique estimator (avsc/fast_samplers.py).

Cells: jazz, pgp, celegans, power at k in {3,4,5,6}; com-dblp, com-amazon at
k in {3,4,5,6}; eps in {0.10, 0.30}; RUNS runs each. Exact counts for every
cell come from tools_gt/ground_truth_kcliques.csv (kClist and Pivoter agree
exactly). Uniform SAMPLE_CAP; no wall-clock censoring.
Seeds: 2_000_000 + 10_000 * graph_index + 1_000 * (2 * (k - 3) + eps_index) + run.

Usage: run_cr_cliques.py [n_workers]
"""
import os

# One BLAS thread per worker: avoids oversubscription and keeps floating-point
# summation order (hence results) identical from run to run.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import csv
import math
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from avsc.run_loop import result_row, run_monitors
from avsc.fast_samplers import CliqueEdgeSampler, csr_from_cache

ALPHA = 0.10
T0 = 100
CHECK_EVERY = 100
BATCH = 10_000
SAMPLE_CAP = 20_000_000
RUNS = 80
RUN_BLOCK = 10
EPS_LIST = [0.10, 0.30]
GRAPHS = ["jazz", "pgp", "celegans", "power", "com-dblp", "com-amazon"]
KS = [3, 4, 5, 6]
OUT_DIR = os.path.join(HERE, "results_cr", "cliques")
FIELDS = ["graph", "k", "eps", "run", "seed", "monitor", "outcome", "T", "L_y", "U_y",
          "nu", "covered", "mid_rel_err", "mean_at_T", "mean_in_interval", "mean_rel_err",
          "sampling_s_at_T", "update_s", "uncertified", "B", "rho", "count",
          "alpha", "sample_cap", "build_s"]

_CACHE = {}


def ground_truth():
    table = {}
    with open(os.path.join(HERE, "tools_gt", "ground_truth_kcliques.csv")) as f:
        for row in csv.DictReader(f):
            table[(row["graph"], int(row["k"]))] = int(row["count"])
    return table


def sampler_template(name, k):
    key = (name, k)
    if key not in _CACHE:
        g = csr_from_cache(os.path.join(HERE, "data", "csr", name + ".npz"))
        t0 = time.perf_counter()
        g.build()
        sampler = CliqueEdgeSampler(g, k, seed=0)
        sampler.build()
        _CACHE[key] = (sampler, time.perf_counter() - t0)
    return _CACHE[key]


def run_one(name, k, eps, run, seed, template, build_s, count):
    sampler = template
    sampler.rng = np.random.default_rng(seed)
    nu = math.comb(k, 2) * count / sampler.bound
    rows = []
    for monitor, info in run_monitors(sampler, nu, eps, ALPHA, T0, CHECK_EVERY, BATCH, SAMPLE_CAP):
        row = {"graph": name, "k": k, "eps": eps, "run": run, "seed": seed}
        row.update(result_row(monitor, info, nu))
        row.update({"B": repr(sampler.bound), "rho": repr(1.0 / nu), "count": count,
                    "alpha": ALPHA, "sample_cap": SAMPLE_CAP, "build_s": repr(build_s)})
        rows.append(row)
    return rows


def run_task(task):
    name, k, eps, runs = task
    path = os.path.join(OUT_DIR, f"{name}_k{k}_eps{eps:.2f}_runs{runs[0]:03d}-{runs[-1]:03d}.csv")
    if os.path.exists(path):
        return path, "skipped (exists)"
    count = ground_truth()[(name, k)]
    template, build_s = sampler_template(name, k)
    gi = GRAPHS.index(name)
    ei = EPS_LIST.index(eps)
    rows = []
    t0 = time.time()
    for run in runs:
        seed = 2_000_000 + 10_000 * gi + 1_000 * (2 * (k - 3) + ei) + run
        rows.extend(run_one(name, k, eps, run, seed, template, build_s, count))
    tmp = path + ".partial"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)
    return path, f"{time.time() - t0:.1f}s"


def all_tasks():
    tasks = []
    for name in GRAPHS:
        for k in KS:
            for eps in EPS_LIST:
                for start in range(0, RUNS, RUN_BLOCK):
                    tasks.append((name, k, eps, list(range(start, start + RUN_BLOCK))))
    tasks.sort(key=lambda task: (-task[1], task[2]))
    return tasks


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with Pool(workers, maxtasksperchild=4) as pool:
        for path, info in pool.imap_unordered(run_task, all_tasks()):
            print(time.strftime("%H:%M:%S"), os.path.basename(path), info, flush=True)


if __name__ == "__main__":
    main()
