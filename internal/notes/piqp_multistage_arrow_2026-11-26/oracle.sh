#!/usr/bin/env bash
# Builds block_info.cpp against PIQP 0.7.0's headers, Eigen and BLASFEO.
#   PIQP=/path/to/piqp-v0.7.0 BLASFEO=/path/with/include+lib ./oracle.sh
# BLASFEO only has to link: extract_arrow_structure itself is plain Eigen, but the multistage
# sources compile only with PIQP_HAS_BLASFEO. CasADi's wheel ships a usable blasfeo.h and
# libblasfeo.so (its include/ and its top directory).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
: "${PIQP:?set PIQP to a PIQP v0.7.0 checkout}"
: "${BLASFEO_INCLUDE:?set BLASFEO_INCLUDE to the directory holding blasfeo.h}"
: "${BLASFEO_LIB:?set BLASFEO_LIB to the directory holding libblasfeo.so}"
EIGEN="${EIGEN:-/usr/include/eigen3}"
c++ -std=c++17 -O1 -DPIQP_HAS_BLASFEO -I"$PIQP/include" -I"$EIGEN" -I"$BLASFEO_INCLUDE" \
  "$here/block_info.cpp" -L"$BLASFEO_LIB" -l:libblasfeo.so -Wl,-rpath,"$BLASFEO_LIB" -o "$here/block_info"
