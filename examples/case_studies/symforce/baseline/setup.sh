#!/bin/sh
# Set up the SymForce study's baseline in baseline/third_party (gitignored):
#   symforce/            symforce-org/symforce at 6582930 (2026-08-27), with two C++ builds (Release, -mcpu=native):
#     build_released/    as released: SymForce's internal timers (SYM_TIME_SCOPE) on, as its benchmark runs
#     build_no_timers/   the timers compiled out (SYMFORCE_CUSTOM_TIC_TOCS with a no-op macro)
#   venv/                Python 3.13 with symforce 0.12.0 from PyPI, for SymForce's code generator, and the
#                        packages its CMake step needs to generate its message types (argh, ply, jinja2)
# Eigen comes from find_package (Homebrew's here) or SymForce's FetchContent. Writes baseline/env.sh.
set -e
here=$(cd "$(dirname "$0")" && pwd)
tp="$here/third_party"
mkdir -p "$tp"
[ -d "$tp/symforce" ] || git clone -q https://github.com/symforce-org/symforce.git "$tp/symforce"
git -C "$tp/symforce" checkout -q 658293082085486f564ca2c7244a778d6db8278f
[ -x "$tp/venv/bin/python" ] || uv venv -q --python 3.13 "$tp/venv"
uv pip install -q --python "$tp/venv/bin/python" symforce==0.12.0 argh ply jinja2 numpy six
common="-DCMAKE_BUILD_TYPE=Release -DSYMFORCE_PYTHON_OVERRIDE=$tp/venv/bin/python -DSYMFORCE_BUILD_OPT=ON -DSYMFORCE_BUILD_CC_SYM=OFF
  -DSYMFORCE_BUILD_EXAMPLES=ON -DSYMFORCE_BUILD_TESTS=OFF -DSYMFORCE_ADD_PYTHON_TESTS=OFF -DSYMFORCE_BUILD_SYMENGINE=OFF
  -DSYMFORCE_GENERATE_MANIFEST=OFF -DSYMFORCE_BUILD_BENCHMARKS=OFF"
cd "$tp/symforce"
[ -f build_released/build.ninja ] || cmake -S . -B build_released -G Ninja $common -DCMAKE_CXX_FLAGS=-mcpu=native >/dev/null
cmake --build build_released -j 8 >/dev/null
mkdir -p build_no_timers/no_tic_toc
printf '#pragma once\n#define SYM_TIME_SCOPE(...) do {} while (0)\n' > build_no_timers/no_tic_toc/noop_tic_toc.h
[ -f build_no_timers/build.ninja ] || cmake -S . -B build_no_timers -G Ninja $common \
  "-DCMAKE_CXX_FLAGS=-mcpu=native -I$tp/symforce/build_no_timers/no_tic_toc" \
  -DSYMFORCE_CUSTOM_TIC_TOCS=ON "-DSYMFORCE_TIC_TOC_HEADER=<noop_tic_toc.h>" >/dev/null
cmake --build build_no_timers -j 8 >/dev/null
eigen=$(sed -n 's/^Eigen3_DIR:PATH=\(.*\)\/share\/eigen3\/cmake$/\1/p' build_released/CMakeCache.txt)
cat > "$here/env.sh" <<ENV
export SYMFORCE="$tp/symforce"
export SYMFORCE_BUILD="$tp/symforce/build_released"
export SYMFORCE_BUILD_NO_TIMERS="$tp/symforce/build_no_timers"
export SYMFORCE_PYTHON="$tp/venv/bin/python"
export EIGEN_INCLUDE="${eigen:-/opt/homebrew}/include/eigen3"
ENV
echo "SymForce built; see $here/env.sh"
