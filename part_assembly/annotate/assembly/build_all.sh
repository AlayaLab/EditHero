#!/bin/bash
# Step 4 for every host of the host list that has a plan: build, verify and render its chain (skips rendered ones).
cd "$(dirname "$0")"
W=${PXFORM_ASSEMBLY_WORK:-./work/assembly}
for oid in $(python -c "import paths; print(' '.join(paths.hosts()))"); do
  [ -f "$W/plans/$oid.json" ] || { echo "$oid: no plan"; continue; }
  [ -f "$W/chains/$oid/img/turn00/f0000.png" ] && { echo "$oid: done"; continue; }
  python build_recipe.py "$oid" 2>&1 | grep -v Warn | sed "s/^/$oid: /"
done
