#!/bin/bash
# Run from anywhere: works relative to this script; tools are built in ../../tools/.
cd "$(dirname "$0")"
TOOLS="$(cd ../.. && pwd)/tools"
# Pivoter (Jain & Seshadhri, WSDM 2020) reads ./src/nCr.txt, so run from this folder.
PIV="$TOOLS/Pivoter/bin/degeneracy_cliques"
mkdir -p src && [ -f src/nCr.txt ] || cp "$TOOLS/Pivoter/src/nCr.txt" src/
for g in jazz pgp celegans power com-dblp com-amazon roadnet-ca com-youtube as-skitter; do
  /usr/bin/time -p $PIV -i edgelists/$g.edges -t A -k 0 -d 0 > gt_logs/pivoter_$g.log 2>&1
done
