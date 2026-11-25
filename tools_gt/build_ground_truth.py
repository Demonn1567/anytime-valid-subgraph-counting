"""Parse kClist and Pivoter logs, require exact agreement, write
ground_truth_kcliques.csv (graph, k, count, kclist_s, pivoter_s)."""
import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "gt_logs")
GRAPHS = ["jazz", "pgp", "celegans", "power", "com-dblp", "com-amazon",
          "roadnet-ca", "com-youtube", "as-skitter"]
KS = [3, 4, 5, 6]


def read(path):
    with open(path) as f:
        return f.read()


def wall_seconds(text):
    return float(re.search(r"^real ([0-9.]+)", text, re.M).group(1))


def kclist_count(graph, k):
    text = read(os.path.join(LOGS, f"kclist_{graph}_k{k}.log"))
    count = int(re.search(rf"Number of {k}-cliques: (\d+)", text).group(1))
    return count, wall_seconds(text)


def pivoter_counts(graph):
    text = read(os.path.join(LOGS, f"pivoter_{graph}.log"))
    counts = {}
    for k_str, value in re.findall(r"^(\d+), ([0-9.]+)$", text, re.M):
        counts[int(k_str)] = value
    return counts, wall_seconds(text)


def main():
    rows = []
    mismatches = 0
    for graph in GRAPHS:
        piv, piv_s = pivoter_counts(graph)
        for k in KS:
            kc, kc_s = kclist_count(graph, k)
            piv_value = piv.get(k, "0.000000")
            whole, frac = piv_value.split(".")
            exact = frac.strip("0") == "" and int(whole) == kc and kc < 2 ** 53
            if not exact:
                mismatches += 1
            print(f"{graph:12s} k={k} kClist={kc:>14d} Pivoter={piv_value:>22s} {'OK' if exact else 'MISMATCH'}")
            rows.append([graph, k, kc, kc_s, piv_s, exact])
    with open(os.path.join(HERE, "ground_truth_kcliques.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["graph", "k", "count", "kclist_wall_s", "pivoter_wall_s_all_k", "tools_agree_exactly"])
        writer.writerows(rows)
    print("mismatches:", mismatches)


main()
