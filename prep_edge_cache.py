"""Cache each experiment graph as canonical edge arrays (from the same
networkx loaders used throughout), and verify that the cached graph is
identical to the edge list given to kClist/Pivoter for ground truth."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from avsc.realgraphs import load_graph
from avsc.bigraphs import load_snap_graph
from avsc.fast_samplers import csr_from_cache, save_edge_cache

SMALL = ["jazz", "pgp", "celegans", "power"]
LARGE = ["com-dblp", "com-amazon", "roadnet-ca", "com-youtube", "as-skitter"]
CACHE = os.path.join(HERE, "data", "csr")


def edgelist_keys(name, n):
    path = os.path.join(HERE, "tools_gt", "edgelists", name + ".txt")
    pairs = np.loadtxt(path, dtype=np.int64, ndmin=2)
    return np.sort(pairs[:, 0] * n + pairs[:, 1])


def main():
    os.makedirs(CACHE, exist_ok=True)
    for name in SMALL + LARGE:
        G = load_graph(name) if name in SMALL else load_snap_graph(name)
        path = os.path.join(CACHE, name + ".npz")
        save_edge_cache(path, G)
        g = csr_from_cache(path)
        g.build()
        canonical = np.sort(g.edge_u * g.n + g.edge_v)
        same = np.array_equal(canonical, edgelist_keys(name, g.n))
        print(f"{name}: n={g.n} m={g.m} identical_to_ground_truth_edgelist={same}", flush=True)
        assert same and g.m == G.number_of_edges()


main()
