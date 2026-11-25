"""
Exact mean and variance of the k-clique edge sampler (Y = X / B) from
Pivoter's per-edge k-clique counts, and of the wedge sampler (closed form).

For edge e: c_e = |CN(e)| = number of triangles on e; w_e = C(c_e, k-2);
cl_e = number of k-cliques containing e. The sampler outputs
Y = (w_e / w_max) * 1[clique] with P(clique | e) = cl_e / w_e, so
  E[Y]   = sum_e cl_e / (M w_max)
  E[Y^2] = sum_e w_e cl_e / (M w_max^2).
rho_eff = 1 + Var(Y) / E[Y]^2 is the concentration ratio of a 0/1 sampler with
the same relative variance (for the wedge sampler it equals rho exactly).
Checks: sum_e cl_e = C(k,2) theta_k against the global counts.
"""
import csv
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPHS = ["jazz", "pgp", "celegans", "power", "com-dblp", "com-amazon"]
KS = [3, 4, 5, 6]
LINE = re.compile(r"^\((\d+), (\d+), (\d+)\): (\d+)$")


def load_ground_truth():
    table = {}
    with open(os.path.join(HERE, "ground_truth_kcliques.csv")) as f:
        for row in csv.DictReader(f):
            table[(row["graph"], int(row["k"]))] = int(row["count"])
    return table


def edge_count(name):
    with open(os.path.join(HERE, "edgelists", name + ".edges")) as f:
        return int(f.readline().split()[1])


def per_edge_counts(name):
    counts = defaultdict(dict)
    with open(os.path.join(HERE, "results", f"{name}_6_E.txt")) as f:
        for line in f:
            m = LINE.match(line.strip())
            if m:
                u, v, k, c = (int(x) for x in m.groups())
                if c:
                    counts[k][(min(u, v), max(u, v))] = c
    return counts


def main():
    gt = load_ground_truth()
    rows = []
    for name in GRAPHS:
        M = edge_count(name)
        counts = per_edge_counts(name)
        triangles_on_edge = counts[3]
        c_max = max(triangles_on_edge.values())
        for k in KS:
            s = k - 2
            cl = counts.get(k, {})
            total = sum(cl.values())
            assert total == math.comb(k, 2) * gt[(name, k)], (name, k, total)
            w_max = math.comb(c_max, s)
            mean = sum(cl.values()) / (M * w_max)
            second = sum(math.comb(triangles_on_edge[e], s) * c for e, c in cl.items()) / (M * w_max ** 2)
            var = second - mean ** 2
            B = M * w_max
            rows.append({"graph": name, "k": k, "B": B, "nu": repr(mean), "rho": repr(1 / mean),
                         "var_Y": repr(var), "bound_var": repr(mean * (1 - mean)),
                         "var_over_bound": repr(var / (mean * (1 - mean))),
                         "rho_eff": repr(1 + var / mean ** 2)})
            print(f"{name:10s} k={k} rho={1/mean:12.2f} Var/bound={var/(mean*(1-mean)):.4f} rho_eff={1+var/mean**2:10.2f}")
    out = os.path.join(HERE, "..", "results_cr", "exact_edge_sampler_variance.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


main()
