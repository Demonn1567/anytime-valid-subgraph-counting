"""
Cost of the k-clique edge sampler and of cheaper range bounds.

For each (graph, k):
  preprocessing  median of BUILD_REPEATS timed builds (CSR + common-neighbour
                 lists + exact range B), excluding loading the cached edges
  per-sample     median time per sample over PER_SAMPLE_DRAWS draws
  bounds         B_exact = M * max_e C(|CN(e)|, k-2)                (needs all CN(e))
                 B_mindeg = M * C(max_e min(deg u, deg v) - 1, k-2)   (O(M), no intersections)
                 B_maxdeg = M * C(max degree, k-2)                     (O(n))
                 all are valid ranges since |CN(e)| <= min(deg u, deg v) - 1 <= max degree.
  inflation      Bet-EB samples to certify eps=EPS with each bound (same streams:
                 a looser bound B' only rescales Y to Y * B / B'), RUNS runs,
                 SAMPLE_CAP cap; reported as the ratio of mean stopping times.
                 Stated rule: if the first EARLY_RUNS runs for a bound all reach
                 the cap, that bound is reported as capped (no ratio exists) and
                 its remaining runs are skipped.
Seeds: 5_000_000 + 10_000 * graph_index + 100 * k + run.
Run ALONE (it times preprocessing): run_cr_clique_cost.py 1
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import csv
import math
import statistics
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from avsc.cs_arms import ConfidenceSequenceArm
from avsc.fast_samplers import CliqueEdgeSampler, csr_from_cache

ALPHA = 0.10
EPS = 0.30
T0 = 100
CHECK_EVERY = 100
BATCH = 10_000
SAMPLE_CAP = 20_000_000
RUNS = 80
EARLY_RUNS = 5
BUILD_REPEATS = 3
PER_SAMPLE_DRAWS = 200_000
CELLS = [("power", k) for k in (3, 4, 5)] + [("com-dblp", k) for k in (3, 4, 5)] + \
        [("com-amazon", k) for k in (3, 4, 5)]
GRAPH_INDEX = {"power": 3, "com-dblp": 4, "com-amazon": 5}
OUT = os.path.join(HERE, "results_cr", "clique_cost.csv")


def stop_time(sampler, scale, seed):
    sampler.rng = np.random.default_rng(seed)
    arm = ConfidenceSequenceArm("betting", "eb", ALPHA)
    while True:
        y = sampler.draw(BATCH) * scale
        event = arm.advance(y, CHECK_EVERY, EPS, T0, SAMPLE_CAP)
        if event:
            return event[0], event[1]


def run_cell(cell):
    name, k = cell
    path = os.path.join(HERE, "data", "csr", name + ".npz")
    builds = []
    sampler = None
    for _ in range(BUILD_REPEATS):
        g = csr_from_cache(path)
        t0 = time.perf_counter()
        g.build()
        sampler = CliqueEdgeSampler(g, k, seed=0)
        sampler.build()
        builds.append(time.perf_counter() - t0)
    g = sampler.graph
    s = k - 2
    t0 = time.perf_counter()
    sampler.draw(PER_SAMPLE_DRAWS)
    per_sample = (time.perf_counter() - t0) / PER_SAMPLE_DRAWS
    min_deg_cn = int(np.max(np.minimum(g.degree[g.edge_u], g.degree[g.edge_v]))) - 1
    bounds = {"exact": sampler.bound,
              "mindeg": float(g.m) * math.comb(min_deg_cn, s),
              "maxdeg": float(g.m) * math.comb(int(g.degree.max()), s)}
    row = {"graph": name, "k": k, "preprocessing_s_median": repr(statistics.median(builds)),
           "per_sample_s": repr(per_sample), "B_exact": repr(bounds["exact"]),
           "B_mindeg": repr(bounds["mindeg"]), "B_maxdeg": repr(bounds["maxdeg"])}
    for label in ("exact", "mindeg", "maxdeg"):
        scale = bounds["exact"] / bounds[label]
        stops = []
        capped = 0
        runs_done = 0
        for run in range(RUNS):
            seed = 5_000_000 + 10_000 * GRAPH_INDEX[name] + 100 * k + run
            outcome, T = stop_time(sampler, scale, seed)
            runs_done += 1
            if outcome == "stopped":
                stops.append(T)
            else:
                capped += 1
            if runs_done == EARLY_RUNS and capped == EARLY_RUNS:
                break
        row[f"T_mean_{label}"] = repr(float(np.mean(stops))) if stops else ""
        row[f"runs_{label}"] = runs_done
        row[f"stopped_{label}"] = len(stops)
        row[f"capped_or_empty_{label}"] = capped
    print(time.strftime("%H:%M:%S"), name, k, "done", flush=True)
    return row


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with Pool(workers) as pool:
        rows = pool.map(run_cell, CELLS)
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
