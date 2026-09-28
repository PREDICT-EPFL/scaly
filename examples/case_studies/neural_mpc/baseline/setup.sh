#!/bin/sh
# Pin the upstream code of the Real-time Neural MPC study and build acados.
#   ml-casadi: the Table II scripts and the torch-to-CasADi MLP (used from PYTHONPATH, not installed)
#   acados:    built from source; its Python interface fetches the t_renderer template binary
#              (github.com/acados/tera_renderer, v0.2.1) into acados/bin on first use
# Run from anywhere; everything lands in baseline/third_party (gitignored).
set -e
here=$(cd "$(dirname "$0")" && pwd)
tp="$here/third_party"
mkdir -p "$tp"
[ -d "$tp/ml-casadi" ] || git clone -q https://github.com/TUM-AAS/ml-casadi.git "$tp/ml-casadi"
git -C "$tp/ml-casadi" checkout -q 44ec47f0d14aa8306d0f5dc786babddfbc8964f9
if [ ! -d "$tp/acados" ]; then
  git clone -q https://github.com/acados/acados.git "$tp/acados"
  git -C "$tp/acados" checkout -q 00947ddc394b30c956d9a7794d0bbcbc44c7c050
  git -C "$tp/acados" submodule update -q --recursive --init
fi
if [ ! -f "$tp/acados/lib/libacados.dylib" ] && [ ! -f "$tp/acados/lib/libacados.so" ]; then
  target=GENERIC
  [ "$(uname -sm)" = "Darwin arm64" ] && target=ARMV8A_APPLE_M1
  cmake -S "$tp/acados" -B "$tp/acados/build" -DACADOS_WITH_QPOASES=OFF -DACADOS_WITH_OPENMP=OFF -DBLASFEO_TARGET=$target -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$tp/acados/build" --target install -j 8 >/dev/null
fi
echo "export ACADOS_SOURCE_DIR=$tp/acados"
echo "acados_template: $tp/acados/interfaces/acados_template"
