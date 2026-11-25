"""Correctness tests for avsc/fast_samplers.py.

Run:  .venv/bin/python tests/test_fast_samplers.py
"""
import csv
import itertools
import math
import os
import sys
from collections import Counter

import networkx as nx
import numpy as np
from scipy.stats import chisquare

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from avsc.fast_samplers import CliqueEdgeSampler, WedgeSampler, csr_from_networkx
from avsc.realgraphs import load_graph
from avsc.bigraphs import load_snap_graph

GT_PATH = os.path.join(HERE, "..", "tools_gt", "ground_truth_kcliques.csv")


def ground_truth():
    table = {}
    with open(GT_PATH) as f:
        for row in csv.DictReader(f):
            table[(row["graph"], int(row["k"]))] = int(row["count"])
    return table


def built_graph(G):
    g = csr_from_networkx(G)
    g.build()
    return g


def z_score(y, target):
    se = y.std(ddof=1) / math.sqrt(len(y))
    return (y.mean() - target) / se if se > 0 else float("nan")


def test_wedge_small(gt):
    for name in ("jazz", "pgp", "celegans", "power"):
        G = load_graph(name)
        g = built_graph(G)
        sampler = WedgeSampler(g, seed=1)
        sampler.build()
        expected_w = sum(d * (d - 1) // 2 for _, d in G.degree())
        assert sampler.total_wedges == expected_w
        nu = 3 * gt[(name, 3)] / sampler.total_wedges
        y = sampler.draw(4_000_000)
        z = z_score(y, nu)
        print(f"    wedge {name:9s} W={expected_w} rho={1/nu:.4f} mean={y.mean():.6f} nu={nu:.6f} z={z:+.2f}")
        assert abs(z) < 4.5
    print("ok  wedge sampler: W exact, unbiased on the four small graphs")


def test_wedge_center_distribution():
    G = load_graph("jazz")
    g = built_graph(G)
    sampler = WedgeSampler(g, seed=2)
    sampler.build()
    r = sampler.rng.integers(0, sampler.total_wedges, size=3_000_000)
    center = np.searchsorted(sampler.cumulative, r, side="right")
    counts = np.bincount(center, minlength=g.n)
    weights = g.degree * (g.degree - 1) / 2
    mask = weights > 0
    expected = weights[mask] / weights.sum() * counts.sum()
    stat, p = chisquare(counts[mask], expected)
    assert counts[~mask].sum() == 0
    print(f"ok  wedge centres follow w_v/W exactly (chi-square p={p:.3f}, vertices without wedges never drawn)")
    assert p > 1e-4


def exact_clique_distribution(G, k):
    """Brute-force distribution of Y for the edge sampler on a tiny graph."""
    edges = [tuple(sorted(e)) for e in G.edges()]
    M = len(edges)
    s = k - 2
    cn = {e: sorted(set(G[e[0]]) & set(G[e[1]])) for e in edges}
    max_weight = max(math.comb(len(c), s) for c in cn.values())
    dist = Counter()
    for e in edges:
        c = cn[e]
        if len(c) < s:
            dist[0.0] += 1.0 / M
            continue
        subsets = list(itertools.combinations(c, s))
        for subset in subsets:
            ok = all(G.has_edge(a, b) for a, b in itertools.combinations(subset, 2))
            value = math.comb(len(c), s) / max_weight if ok else 0.0
            dist[round(value, 12)] += 1.0 / (M * len(subsets))
    return dist


def test_clique_exact_distribution():
    rng = np.random.default_rng(5)
    G = nx.complete_graph(6)
    G.add_edges_from([(5, 6), (6, 7), (7, 5), (7, 8), (8, 9), (9, 6), (2, 9)])
    G.add_edges_from((int(a), int(b)) for a, b in rng.integers(0, 14, size=(10, 2)) if a != b)
    for k in (3, 4, 5, 6):
        g = built_graph(G)
        sampler = CliqueEdgeSampler(g, k, seed=10 + k)
        sampler.build()
        exact = exact_clique_distribution(G, k)
        y = np.round(sampler.draw(2_000_000), 12)
        observed = Counter(y.tolist())
        support = sorted(exact)
        assert set(observed) <= set(support), (k, set(observed) - set(support))
        obs = np.array([observed.get(v, 0) for v in support], dtype=float)
        exp = np.array([exact[v] for v in support]) * len(y)
        stat, p = chisquare(obs, exp)
        print(f"    clique k={k} support={len(support)} chi-square p={p:.3f}")
        assert p > 1e-4
    print("ok  clique sampler reproduces the brute-force output distribution exactly (k=3..6)")


def test_clique_bounds_match_logged():
    logged = {("com-amazon", 3): 149065392.0, ("com-dblp", 3): 223621458.0}
    for (name, k), value in logged.items():
        g = built_graph(load_snap_graph(name))
        sampler = CliqueEdgeSampler(g, k, seed=0)
        sampler.build()
        print(f"    {name} k={k}: B={sampler.bound:.1f} (logged {value:.1f})")
        assert sampler.bound == value
    print("ok  clique range B matches the values logged by the original experiments")


def test_clique_unbiased(gt):
    cases = [("jazz", k) for k in (3, 4, 5, 6)] + [("pgp", k) for k in (3, 4, 5, 6)] + \
            [("celegans", k) for k in (3, 4, 5)] + [("power", k) for k in (3, 4, 5)]
    graphs = {}
    for name, k in cases:
        if name not in graphs:
            graphs[name] = built_graph(load_graph(name))
        sampler = CliqueEdgeSampler(graphs[name], k, seed=100 + k)
        sampler.build()
        nu = math.comb(k, 2) * gt[(name, k)] / sampler.bound
        y = sampler.draw(4_000_000)
        z = z_score(y, nu)
        nonzero = float((y > 0).mean())
        print(f"    clique {name:9s} k={k} rho={1/nu:12.2f} rel_err={(y.mean()-nu)/nu:+.4f} z={z:+.2f} nonzero={nonzero:.4f}")
        assert abs(z) < 4.5 or nonzero * len(y) < 2000
    print("ok  clique sampler unbiased against exact kClist/Pivoter counts")


def main():
    gt = ground_truth()
    test_wedge_center_distribution()
    test_wedge_small(gt)
    test_clique_exact_distribution()
    test_clique_unbiased(gt)
    test_clique_bounds_match_logged()
    print("ALL TESTS PASSED")


main()
