"""
Camera-ready non-anytime-valid baselines on the triangle (wedge) sampler.

(a) fixed-budget Hoeffding (pilot, then commit): draw PILOT samples, estimate
    nu_hat0; commit to n* = ceil(2 log(2/alpha) / (eps^2 nu_hat0^2)), the
    classical single-look Hoeffding sample size that gives relative width eps
    if nu_hat0 were exact; draw n* FRESH samples and report
    mean +/- sqrt(log(2/alpha) / (2 n*)). Valid at its single look because n*
    depends only on the independent pilot. If nu_hat0 = 0 or n* > SAMPLE_CAP,
    the run is reported as infeasible (no extrapolation).
(b) fixed-n empirical Bernstein (Maurer & Pontil 2009, Thm. 4) at an ORACLE n:
    the median stopping time of Bet-EB in the same (graph, eps) cell of the
    main grid. Two-sided at level alpha, so each side uses delta = alpha/2:
    mean +/- (sqrt(2 V log(4/alpha) / n) + 7 log(4/alpha) / (3 (n - 1))).
Both report coverage and the realised relative width (whether eps was met).
The naive peeking rule (Wald interval re-checked every batch) is run inside
the main grid itself (monitor "wald-peeking").

Seeds: 3_000_000 + 10_000 * graph_index + 1_000 * eps_index + run.
Usage: run_cr_baselines.py [n_workers]
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import csv
import glob
import math
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from avsc.fast_samplers import WedgeSampler, csr_from_cache
from run_cr_grid import EPS_LIST, GRAPHS, TRIANGLES

ALPHA = 0.10
PILOT = 100
SAMPLE_CAP = 300_000_000
DRAW_CHUNK = 1_000_000
RUNS = 80
OUT = os.path.join(HERE, "results_cr", "baselines_triangles.csv")
FIELDS = ["graph", "eps", "run", "seed", "method", "feasible", "n_used", "L_y", "U_y",
          "nu", "covered", "rel_width", "met_target", "oracle_n", "sampling_s"]


def oracle_n_table():
    table = {}
    rows = []
    for path in glob.glob(os.path.join(HERE, "results_cr", "triangles", "*.csv")):
        with open(path) as f:
            rows.extend(r for r in csv.DictReader(f) if r["monitor"] == "betting-eb")
    for name in GRAPHS:
        for eps in EPS_LIST:
            ts = [int(r["T"]) for r in rows if r["graph"] == name and float(r["eps"]) == eps
                  and r["outcome"] == "stopped"]
            table[(name, eps)] = int(np.median(ts)) if ts else None
    return table


def mean_and_sq(sampler, n):
    total = 0.0
    total_sq = 0.0
    left = n
    while left > 0:
        y = sampler.draw(min(DRAW_CHUNK, left))
        total += float(y.sum())
        total_sq += float(np.dot(y, y))
        left -= len(y)
    return total / n, total_sq / n


def interval_row(base, method, L, U, nu, eps, n_used, oracle_n, seconds, feasible=True):
    mid = 0.5 * (L + U)
    width = (U - L) / mid if mid > 0 else float("inf")
    return dict(base, method=method, feasible=feasible, n_used=n_used, L_y=repr(L), U_y=repr(U),
                nu=repr(nu), covered=bool(L <= nu <= U), rel_width=repr(width),
                met_target=bool(width <= eps), oracle_n=oracle_n, sampling_s=repr(seconds))


def run_task(task):
    name, eps, oracle_n = task
    g = csr_from_cache(os.path.join(HERE, "data", "csr", name + ".npz"))
    g.build()
    gi = GRAPHS.index(name)
    ei = EPS_LIST.index(eps)
    rows = []
    for run in range(RUNS):
        seed = 3_000_000 + 10_000 * gi + 1_000 * ei + run
        sampler = WedgeSampler(g, seed)
        sampler.build()
        nu = 3.0 * TRIANGLES[name] / sampler.total_wedges
        base = {"graph": name, "eps": eps, "run": run, "seed": seed}
        t0 = time.perf_counter()
        pilot_mean = float(sampler.draw(PILOT).mean())
        n_star = math.ceil(2.0 * math.log(2.0 / ALPHA) / (eps ** 2 * pilot_mean ** 2)) if pilot_mean > 0 else None
        if n_star is None or n_star > SAMPLE_CAP:
            rows.append(dict(base, method="fixed-budget-hoeffding", feasible=False, n_used=PILOT,
                             L_y="", U_y="", nu=repr(nu), covered="", rel_width="", met_target=False,
                             oracle_n="", sampling_s=repr(time.perf_counter() - t0)))
        else:
            mean, _ = mean_and_sq(sampler, n_star)
            half = math.sqrt(math.log(2.0 / ALPHA) / (2.0 * n_star))
            rows.append(interval_row(base, "fixed-budget-hoeffding", mean - half, mean + half, nu, eps,
                                     PILOT + n_star, "", time.perf_counter() - t0))
        if oracle_n:
            t0 = time.perf_counter()
            mean, mean_sq = mean_and_sq(sampler, oracle_n)
            var = max(mean_sq - mean * mean, 0.0) * oracle_n / (oracle_n - 1)
            log_term = math.log(4.0 / ALPHA)
            half = math.sqrt(2.0 * var * log_term / oracle_n) + 7.0 * log_term / (3.0 * (oracle_n - 1))
            rows.append(interval_row(base, "fixed-n-eb-oracle", max(mean - half, 0.0), min(mean + half, 1.0),
                                     nu, eps, oracle_n, oracle_n, time.perf_counter() - t0))
    return rows


def main():
    oracle = oracle_n_table()
    tasks = [(name, eps, oracle[(name, eps)]) for name in GRAPHS for eps in EPS_LIST]
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    all_rows = []
    with Pool(workers) as pool:
        for rows in pool.imap_unordered(run_task, tasks):
            all_rows.extend(rows)
            print(time.strftime("%H:%M:%S"), rows[0]["graph"], rows[0]["eps"], "done", flush=True)
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=lambda r: (r["graph"], r["eps"], r["run"], r["method"])))


if __name__ == "__main__":
    main()
