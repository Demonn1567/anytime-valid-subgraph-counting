"""
Vectorised bounded unbiased samplers on a CSR graph (same distributions as
the reference samplers in estimators.py, drawn with numpy instead of a
per-sample Python loop).

Both samplers return Y = X / B in [0, 1] directly, so the stream fed to a
confidence sequence carries no rounding from dividing large numbers.

Preprocessing (the `build` step, timed separately by callers):
  CSRGraph:            sort the 2m directed edges, O(m log m); degrees, row
                       pointers, and sorted 64-bit edge keys u*n+v used for
                       O(log m) adjacency tests.
  WedgeSampler:        integer cumulative wedge counts w_v = C(deg v, 2), O(n).
  CliqueEdgeSampler:   common-neighbour list of every edge, O(sum_e min(d_u, d_v) log m),
                       and the range B = M * max_e C(|CN(e)|, k-2).
"""

import math
import numpy as np


class CSRGraph:
    """Simple undirected graph in CSR form, nodes 0..n-1."""

    def __init__(self, n, u, v):
        self.n = int(n)
        self.u_raw = np.asarray(u, dtype=np.int64)
        self.v_raw = np.asarray(v, dtype=np.int64)

    def build(self):
        u = np.minimum(self.u_raw, self.v_raw)
        v = np.maximum(self.u_raw, self.v_raw)
        keep = u != v
        u = u[keep]
        v = v[keep]
        canonical = np.unique(u * self.n + v)
        self.edge_u = canonical // self.n
        self.edge_v = canonical % self.n
        self.m = len(canonical)
        src = np.concatenate([self.edge_u, self.edge_v])
        dst = np.concatenate([self.edge_v, self.edge_u])
        keys = src * self.n + dst
        order = np.argsort(keys, kind="stable")
        self.keys = keys[order]
        self.dst = dst[order]
        self.degree = np.bincount(src, minlength=self.n).astype(np.int64)
        self.indptr = np.zeros(self.n + 1, dtype=np.int64)
        np.cumsum(self.degree, out=self.indptr[1:])

    def adjacent(self, a, b):
        query = a * self.n + b
        pos = np.searchsorted(self.keys, query)
        pos = np.minimum(pos, len(self.keys) - 1)
        return self.keys[pos] == query


def edge_arrays_from_networkx(G):
    """(n, u, v) with nodes relabelled 0..n-1 in sorted node order."""
    nodes = sorted(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    pairs = np.array([(index[a], index[b]) for a, b in G.edges()], dtype=np.int64)
    return len(nodes), pairs[:, 0], pairs[:, 1]


def csr_from_networkx(G):
    n, u, v = edge_arrays_from_networkx(G)
    return CSRGraph(n, u, v)


def save_edge_cache(path, G):
    n, u, v = edge_arrays_from_networkx(G)
    np.savez(path, n=np.array([n]), u=u, v=v)


def csr_from_cache(path):
    data = np.load(path)
    return CSRGraph(int(data["n"][0]), data["u"], data["v"])


class WedgeSampler:
    """Y = 1[sampled wedge is closed]; E[Y] = 3 T / W; B = W."""

    def __init__(self, graph, seed):
        self.graph = graph
        self.rng = np.random.default_rng(seed)

    def build(self):
        g = self.graph
        wedges = g.degree * (g.degree - 1) // 2
        self.cumulative = np.cumsum(wedges)
        self.total_wedges = int(self.cumulative[-1])
        self.bound = float(self.total_wedges)

    def draw(self, size):
        g = self.graph
        r = self.rng.integers(0, self.total_wedges, size=size)
        center = np.searchsorted(self.cumulative, r, side="right")
        d = g.degree[center]
        i = self.rng.integers(0, d)
        j = self.rng.integers(0, d - 1)
        j = j + (j >= i)
        a = g.dst[g.indptr[center] + i]
        b = g.dst[g.indptr[center] + j]
        return g.adjacent(a, b).astype(float)


class CliqueEdgeSampler:
    """Edge sampling with common-neighbour completion for k-cliques.
    Draw edge e uniformly; choose s = k-2 distinct vertices of CN(e) uniformly;
    X = M * C(|CN(e)|, s) * 1[clique]; E[X] = C(k,2) * #k-cliques; B = M * max_e C(|CN(e)|, s)."""

    def __init__(self, graph, k, seed, batch_size=4_000_000):
        assert k >= 3
        self.graph = graph
        self.k = k
        self.s = k - 2
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size

    def build(self):
        g = self.graph
        du = g.degree[g.edge_u]
        dv = g.degree[g.edge_v]
        low = np.where(du <= dv, g.edge_u, g.edge_v)
        high = np.where(du <= dv, g.edge_v, g.edge_u)
        lengths = g.degree[low]
        cn_parts = []
        cn_sizes = np.zeros(g.m, dtype=np.int64)
        start = 0
        while start < g.m:
            stop = start
            budget = 0
            while stop < g.m and (budget + lengths[stop] <= self.batch_size or stop == start):
                budget += lengths[stop]
                stop += 1
            edge_ids = np.repeat(np.arange(start, stop), lengths[start:stop])
            offsets = np.arange(len(edge_ids)) - np.repeat(
                np.cumsum(lengths[start:stop]) - lengths[start:stop], lengths[start:stop])
            candidates = g.dst[g.indptr[low[edge_ids]] + offsets]
            hit = g.adjacent(high[edge_ids], candidates)
            cn_parts.append(candidates[hit])
            cn_sizes[start:stop] = np.bincount(edge_ids[hit] - start, minlength=stop - start)
            start = stop
        self.cn = np.concatenate(cn_parts) if cn_parts else np.zeros(0, dtype=np.int64)
        self.cn_ptr = np.zeros(g.m + 1, dtype=np.int64)
        np.cumsum(cn_sizes, out=self.cn_ptr[1:])
        self.cn_sizes = cn_sizes
        weights = np.array([math.comb(int(c), self.s) for c in range(int(cn_sizes.max()) + 1)], dtype=float)
        self.edge_weight = weights[cn_sizes]
        self.max_weight = float(self.edge_weight.max())
        self.bound = float(g.m) * self.max_weight
        self.edges_per_clique = math.comb(self.k, 2)

    def draw(self, size):
        g = self.graph
        e = self.rng.integers(0, g.m, size=size)
        c = self.cn_sizes[e]
        y = np.zeros(size)
        valid = np.nonzero(c >= self.s)[0]
        if len(valid) == 0:
            return y
        ev = e[valid]
        cv = c[valid]
        chosen = np.zeros((len(valid), 0), dtype=np.int64)
        for step in range(self.s):
            j = self.rng.integers(0, cv - step)
            for col in range(chosen.shape[1]):
                j = j + (j >= chosen[:, col])
            chosen = np.sort(np.column_stack([chosen, j]), axis=1)
        vertices = self.cn[self.cn_ptr[ev][:, None] + chosen]
        ok = np.ones(len(valid), dtype=bool)
        for a in range(self.s):
            for b in range(a + 1, self.s):
                ok &= g.adjacent(vertices[:, a], vertices[:, b])
        y[valid] = (self.edge_weight[ev] / self.max_weight) * ok
        return y
