"""
Loaders for larger public real-world graphs (SNAP edge-list format), used for
Experiment 1 (scale) and later experiments. Separate from realgraphs.py (which
holds the four original METIS-format graphs already used in the paper) so the
existing pipeline is untouched.

Each entry in SNAP_GRAPHS names a graph, its source URL, and the on-disk cache
path. Files are gzip-compressed plain edge lists with '#'-prefixed comment
lines, the standard SNAP format.
"""

import os
import time
import gzip
import urllib.request
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "data", "big")
os.makedirs(CACHE, exist_ok=True)

SNAP_GRAPHS = {
    "com-amazon": {
        "url": "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz",
        "file": "com-amazon.ungraph.txt.gz",
        "citation": "J. Yang and J. Leskovec, 'Defining and Evaluating Network "
                    "Communities based on Ground-truth', ICDM 2012. "
                    "https://snap.stanford.edu/data/com-Amazon.html",
    },
    "com-dblp": {
        "url": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz",
        "file": "com-dblp.ungraph.txt.gz",
        "citation": "J. Yang and J. Leskovec, 'Defining and Evaluating Network "
                    "Communities based on Ground-truth', ICDM 2012. "
                    "https://snap.stanford.edu/data/com-DBLP.html",
    },
    "com-youtube": {
        "url": "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz",
        "file": "com-youtube.ungraph.txt.gz",
        "citation": "J. Yang and J. Leskovec, 'Defining and Evaluating Network "
                    "Communities based on Ground-truth', ICDM 2012. "
                    "https://snap.stanford.edu/data/com-Youtube.html",
    },
    "roadnet-ca": {
        "url": "https://snap.stanford.edu/data/roadNet-CA.txt.gz",
        "file": "roadNet-CA.txt.gz",
        "citation": "J. Leskovec, K. Lang, A. Dasgupta, M. Mahoney, "
                    "'Community Structure in Large Networks', 2008. "
                    "https://snap.stanford.edu/data/roadNet-CA.html",
    },
    "as-skitter": {
        "url": "https://snap.stanford.edu/data/as-skitter.txt.gz",
        "file": "as-skitter.txt.gz",
        "citation": "SNAP, 'AS-Skitter internet topology graph'. "
                    "https://snap.stanford.edu/data/as-Skitter.html",
    },
}


def download_if_missing(name):
    spec = SNAP_GRAPHS[name]
    path = os.path.join(CACHE, spec["file"])
    if os.path.exists(path):
        return path, 0.0
    t0 = time.time()
    urllib.request.urlretrieve(spec["url"], path)
    return path, time.time() - t0


def load_snap_graph(name, timing=None):
    """Load a SNAP undirected edge-list graph, timing parse time if `timing`
    dict is passed (key 'parse_s' set to elapsed seconds)."""
    path, dl_s = download_if_missing(name)
    t0 = time.time()
    G = nx.Graph()
    edges = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            u, v = line.split()
            u = int(u); v = int(v)
            if u != v:
                edges.append((u, v))
    G.add_edges_from(edges)
    parse_s = time.time() - t0
    if timing is not None:
        timing["download_s"] = dl_s
        timing["parse_s"] = parse_s
    return G
