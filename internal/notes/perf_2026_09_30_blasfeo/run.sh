#!/usr/bin/env bash
# Build the harness against a Linux BLASFEO build and time every (op, n) that gen.py built.
#   BLASFEO=$HOME/blasfeo_build bash run.sh [ops] [ns]  > results/raw_<tag>.txt
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BLASFEO="${BLASFEO:-$HOME/blasfeo_build}"
BIN="${BIN:-$HOME/blasfeo_bench}"
gcc -O2 -Wall -I"$BLASFEO/include" "$HERE/bench.c" "$BLASFEO/lib/libblasfeo.a" -ldl -lm -o "$BIN"
OPS="${1:-gemm potrf trsm}"
NS="${2:-4 8 12 16 24 32 48 64 96 128}"
for op in $OPS; do
  for n in $NS; do
    d="$HERE/build/${op}_$n"
    ws=$(python3 -c "import json;print(json.load(open('$d/meta.json'))['workspace'])")
    "$BIN" "$op" "$n" "$d/jit.so" "$d/o3.so" "bf_${op}_$n" "$ws" 20 9
  done
done
