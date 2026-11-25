"""
Loaders for real-world graphs used in the experiments.

Graphs are fetched from public mirrors (the NetworKit input collection on
GitHub) in METIS adjacency format and cached locally. These are standard
real-world networks: a jazz-musician collaboration network, the US Western
power grid, the PGP web-of-trust giant component, and the C. elegans metabolic
network. They span a wide range of size, density, and triangle concentration,
which is exactly the axis along which our adaptive sample complexity varies.
"""

import os
import urllib.request
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "data")
os.makedirs(CACHE, exist_ok=True)

BASE = "https://raw.githubusercontent.com/networkit/networkit/master/input/"
GRAPHS = {
    "jazz": "jazz.graph",
    "power": "power.graph",
    "pgp": "PGPgiantcompo.graph",
    "celegans": "celegans_metabolic.graph",
}


def _parse_metis(text):
    lines = text.strip().split("\n")
    header = lines[0].split()
    n = int(header[0])
    G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    for i, line in enumerate(lines[1:n + 1], start=1):
        for tok in line.split():
            j = int(tok)
            if j > i:
                G.add_edge(i, j)
    return G


def load_graph(name):
    """Load a named real graph, fetching+caching the raw file if needed."""
    fn = GRAPHS[name]
    path = os.path.join(CACHE, fn)
    if not os.path.exists(path):
        url = BASE + fn
        txt = urllib.request.urlopen(url, timeout=30).read().decode()
        with open(path, "w") as f:
            f.write(txt)
    with open(path) as f:
        return _parse_metis(f.read())


def load_all():
    return {name: load_graph(name) for name in GRAPHS}
