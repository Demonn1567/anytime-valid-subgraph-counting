"""Write each experiment graph, exactly as our loaders build it (simple,
undirected, self-loops dropped, duplicates merged), as a 0-indexed edge list
for the external exact clique counters kClist and Pivoter."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from avsc.realgraphs import load_graph
from avsc.bigraphs import load_snap_graph

SMALL = ["jazz", "pgp", "celegans", "power"]
LARGE = ["com-dblp", "com-amazon", "roadnet-ca", "com-youtube", "as-skitter"]
OUT = os.path.join(HERE, "edgelists")


def write_graph(name, G):
    index = {v: i for i, v in enumerate(sorted(G.nodes()))}
    edges = []
    for u, v in G.edges():
        a, b = index[u], index[v]
        if a > b:
            a, b = b, a
        edges.append((a, b))
    edges.sort()
    with open(os.path.join(OUT, name + ".txt"), "w") as f:
        for a, b in edges:
            f.write(f"{a} {b}\n")
    with open(os.path.join(OUT, name + ".edges"), "w") as f:
        f.write(f"{len(index)} {len(edges)}\n")
        for a, b in edges:
            f.write(f"{a} {b}\n")
    print(f"{name} n={G.number_of_nodes()} m={G.number_of_edges()}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in SMALL:
        write_graph(name, load_graph(name))
    for name in LARGE:
        write_graph(name, load_snap_graph(name))


main()
