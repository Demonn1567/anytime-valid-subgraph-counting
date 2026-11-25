"""
Dedicated wall-clock measurement for the camera-ready paper.

Runs in ONE process with no other experiment running, after the grids finish.
For each large graph:
  build: BUILD_REPEATS timed builds of the CSR graph plus the wedge table
         (loading the cached edge arrays from disk is excluded and reported
         separately as load time); median reported.
  runs:  for each eps, RUNS timed runs through the same code path as the main
         grid (run_cr_grid.run_one), recording per monitor the sampling time
         up to its stop and its confidence-sequence update time.
Seeds: 4_000_000 + 10_000 * graph_index + 1_000 * eps_index + run.
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import csv
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from avsc.fast_samplers import WedgeSampler, csr_from_cache
from run_cr_grid import EPS_LIST, GRAPHS, run_one

TIMED_GRAPHS = ["com-dblp", "com-amazon", "roadnet-ca", "com-youtube", "as-skitter"]
BUILD_REPEATS = 5
RUNS = 10
OUT_BUILD = os.path.join(HERE, "results_cr", "timing_build.csv")
OUT_RUNS = os.path.join(HERE, "results_cr", "timing_runs.csv")


def timed_build(name):
    path = os.path.join(HERE, "data", "csr", name + ".npz")
    loads = []
    builds = []
    graph = None
    for _ in range(BUILD_REPEATS):
        t0 = time.perf_counter()
        graph = csr_from_cache(path)
        loads.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        graph.build()
        sampler = WedgeSampler(graph, seed=0)
        sampler.build()
        builds.append(time.perf_counter() - t0)
    return graph, statistics.median(loads), statistics.median(builds), builds


def main():
    build_rows = []
    run_rows = []
    for name in TIMED_GRAPHS:
        graph, load_s, build_s, builds = timed_build(name)
        build_rows.append({"graph": name, "load_s_median": repr(load_s), "build_s_median": repr(build_s),
                           "build_s_all": ";".join(f"{b:.4f}" for b in builds)})
        print(f"{name}: load {load_s:.3f}s build {build_s:.3f}s", flush=True)
        gi = GRAPHS.index(name)
        for eps in EPS_LIST:
            ei = EPS_LIST.index(eps)
            for run in range(RUNS):
                seed = 4_000_000 + 10_000 * gi + 1_000 * ei + run
                rows = run_one(name, eps, run, seed, graph, build_s)
                run_rows.extend(rows)
            print(f"  {name} eps={eps} done {time.strftime('%H:%M:%S')}", flush=True)
    with open(OUT_BUILD, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(build_rows[0]))
        writer.writeheader()
        writer.writerows(build_rows)
    with open(OUT_RUNS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)


if __name__ == "__main__":
    main()
