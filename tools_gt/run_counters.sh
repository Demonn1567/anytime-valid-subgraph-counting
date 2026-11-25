#!/bin/bash
# Run from anywhere: works relative to this script; tools are built in ../../tools/.
cd "$(dirname "$0")"
TOOLS="$(cd ../.. && pwd)/tools"
# Exact k-clique counts with kClist (Pivoter: run_pivoter.sh); log wall time.
KCL="$TOOLS/kClist/kClist"
for g in jazz pgp celegans power com-dblp com-amazon roadnet-ca com-youtube as-skitter; do
  for k in 3 4 5 6; do
    /usr/bin/time -p $KCL $k edgelists/$g.txt > gt_logs/kclist_${g}_k$k.log 2>&1
  done
  echo "done $g"
done
