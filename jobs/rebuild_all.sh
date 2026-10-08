#!/bin/bash
# rebuild_all.sh DATA OUT: rebuild and verify every chain of the data package (DATA = its data/ directory) into OUT.
set -u
DATA=${1:?data directory of the data package}; OUT=${2:?out dir}; P=$(cd "$(dirname "$0")/.." && pwd)
python -c "import json,sys;print('\n'.join(c['chain'] for c in json.load(open(sys.argv[1]))))" "$DATA/chains.json" | while read c; do
  python $P/part_assembly/tools/assemble.py --pv-root "$DATA/parts/partverse_xl" --hy3d-root "$DATA/parts/hy3d" --chain "$DATA/recipes/$c" --out "$OUT/${c/\//_}" | tail -1
done
