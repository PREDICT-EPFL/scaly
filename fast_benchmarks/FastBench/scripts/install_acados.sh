#!/usr/bin/env bash
# Build acados (with HPIPM + qpOASES) and install the Python template interface.
# Installs into ./third_party/acados.  Requires git, cmake and a C compiler.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="$PWD/third_party/acados"
mkdir -p third_party

if [ ! -d "$DEST" ]; then
  echo ">> cloning acados"
  git clone https://github.com/acados/acados.git "$DEST"
  git -C "$DEST" submodule update --recursive --init
fi

echo ">> building acados"
mkdir -p "$DEST/build"
cd "$DEST/build"
cmake -DACADOS_WITH_QPOASES=ON -DACADOS_WITH_OSQP=ON \
      -DACADOS_INSTALL_DIR="$DEST" ..
make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
make install
cd "$PWD"

echo ">> installing acados_template (Python interface)"
pip install -e "$DEST/interfaces/acados_template"

# t_renderer is needed for code generation; acados downloads it on first use,
# but you can pre-fetch by running any acados python example once.
echo ">> acados built at $DEST"
echo "   export ACADOS_SOURCE_DIR=$DEST"
echo "   export LD_LIBRARY_PATH=$DEST/lib:\${LD_LIBRARY_PATH:-}"
