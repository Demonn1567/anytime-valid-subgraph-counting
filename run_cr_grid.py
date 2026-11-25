"""
Camera-ready triangle grid: nine graphs x eps in {0.10, 0.30} x RUNS runs.

Every run draws ONE i.i.d. wedge-sample stream (in batches of BATCH) and feeds
it to five monitors that evaluate their interval every CHECK_EVERY samples and
stop independently (shared loop: avsc/run_loop.py):
  betting-eb, betting-hoeffding, prpl-eb, prpl-hoeffding  (anytime-valid, see avsc/cs_arms.py)
  wald-peeking  (NOT anytime-valid: 90% normal-approximation interval, re-checked every chunk)
A monitor stops at the first checkpoint t >= T0 with (U - L) / ((U + L) / 2) <= eps.
Outcomes: "stopped"; "cap" (SAMPLE_CAP reached; the final interval is still a
valid CS value and its coverage is recorded); "empty" (running intersection
became empty, which certifies a miss). There is no wall-clock censoring.

Settings (unchanged from the submission): alpha=0.10, T0=100, CHECK_EVERY=100.
Seeds: 1_000_000 + 10_000 * graph_index + 1_000 * eps_index + run.

Usage: run_cr_grid.py [n_workers]
"""
import os

# One BLAS thread per worker: avoids oversubscription and keeps floating-point
# summation order (hence results) identical from run to run.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import csv
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from avsc.run_loop import result_row, run_monitors
from avsc.fast_samplers import WedgeSampler, csr_from_cache

ALPHA = 0.10
T0 = 100
CHECK_EVERY = 100
BATCH = 10_000
SAMPLE_CAP = 300_000_000
RUNS = 80
RUN_BLOCK = 10
EPS_LIST = [0.10, 0.30]
GRAPHS = ["jazz", "pgp", "celegans", "power", "com-dblp", "com-amazon",
          "roadnet-ca", "com-youtube", "as-skitter"]
TRIANGLES = {"jazz": 17899, "pgp": 54788, "celegans": 3284, "power": 651,
             "com-dblp": 2224385, "com-amazon": 667129, "roadnet-ca": 120676,
             "com-youtube": 3056386, "as-skitter": 28769868}
OUT_DIR = os.path.join(HERE, "results_cr", "triangles")
FIELDS = ["graph", "eps", "run", "seed", "monitor", "outcome", "T", "L_y", "U_y",
          "nu", "covered", "mid_rel_err", "mean_at_T", "mean_in_interval", "mean_rel_err",
          "sampling_s_at_T", "update_s", "uncertified", "W", "rho", "alpha",
          "sample_cap", "build_s"]

_GRAPH_CACHE = {}


def graph_for(name):
    if name not in _GRAPH_CACHE:
        g = csr_from_cache(os.path.join(HERE, "data", "csr", name + ".npz"))
        t0 = time.perf_counter()
        g.build()
        _GRAPH_CACHE[name] = (g, time.perf_counter() - t0)
    return _GRAPH_CACHE[name]


def run_one(name, eps, run, seed, g, build_s):
    sampler = WedgeSampler(g, seed)
    t0 = time.perf_counter()
    sampler.build()
    build_total = build_s + (time.perf_counter() - t0)
    nu = 3.0 * TRIANGLES[name] / sampler.total_wedges
    rows = []
    for monitor, info in run_monitors(sampler, nu, eps, ALPHA, T0, CHECK_EVERY, BATCH, SAMPLE_CAP):
        row = {"graph": name, "eps": eps, "run": run, "seed": seed}
        row.update(result_row(monitor, info, nu))
        row.update({"W": sampler.total_wedges, "rho": repr(1.0 / nu), "alpha": ALPHA,
                    "sample_cap": SAMPLE_CAP, "build_s": repr(build_total)})
        rows.append(row)
    return rows


def run_task(task):
    name, eps, runs = task
    path = os.path.join(OUT_DIR, f"{name}_eps{eps:.2f}_runs{runs[0]:03d}-{runs[-1]:03d}.csv")
    if os.path.exists(path):
        return path, "skipped (exists)"
    g, build_s = graph_for(name)
    gi = GRAPHS.index(name)
    ei = EPS_LIST.index(eps)
    rows = []
    t0 = time.time()
    for run in runs:
        seed = 1_000_000 + 10_000 * gi + 1_000 * ei + run
        rows.extend(run_one(name, eps, run, seed, g, build_s))
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
        for eps in EPS_LIST:
            for start in range(0, RUNS, RUN_BLOCK):
                tasks.append((name, eps, list(range(start, start + RUN_BLOCK))))
    heavy_first = {"as-skitter": 0, "com-youtube": 1, "roadnet-ca": 2}
    tasks.sort(key=lambda task: (heavy_first.get(task[0], 9), task[1]))
    return tasks


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    with Pool(workers, maxtasksperchild=4) as pool:
        for path, info in pool.imap_unordered(run_task, all_tasks()):
            print(time.strftime("%H:%M:%S"), os.path.basename(path), info, flush=True)


if __name__ == "__main__":
    main()
