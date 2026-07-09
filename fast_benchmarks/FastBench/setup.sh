#!/usr/bin/env bash
# FastBench setup: create a virtual environment and install solvers.
#
#   ./setup.sh                 # core: casadi+ipopt, osqp, piqp, reporting
#   ./setup.sh --do-mpc        # also install do-mpc
#   ./setup.sh --acados        # also build acados + python template
#   ./setup.sh --grampc        # also fetch GRAMPC + pygrampc
#   ./setup.sh --all           # everything
#
# Re-run any time; it is idempotent.
set -euo pipefail
cd "$(dirname "$0")"

WITH_DOMPC=0; WITH_ACADOS=0; WITH_GRAMPC=0
for a in "$@"; do
  case "$a" in
    --do-mpc) WITH_DOMPC=1;;
    --acados) WITH_ACADOS=1;;
    --grampc) WITH_GRAMPC=1;;
    --all) WITH_DOMPC=1; WITH_ACADOS=1; WITH_GRAMPC=1;;
    *) echo "unknown option: $a"; exit 1;;
  esac
done

PY=${PYTHON:-python3}
echo ">> creating virtual environment .venv"
$PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel

echo ">> installing FastBench (core: casadi+ipopt, osqp, piqp, proxqp, reporting)"
pip install -e ".[report,dev]"

if command -v g++ >/dev/null 2>&1; then
  echo ">> g++ found - FastSQP (codegen C++ SQP) will be available"
else
  echo "!! g++ not found - install a C++ compiler to enable the FastSQP backend"
fi

if [ "$WITH_DOMPC" = "1" ]; then
  echo ">> installing do-mpc"
  pip install "do-mpc>=4.6"
fi

if [ "$WITH_GRAMPC" = "1" ]; then
  echo ">> installing pygrampc (requires a C compiler)"
  pip install pygrampc || echo "!! pygrampc install failed - see https://github.com/grampc/pygrampc"
fi

if [ "$WITH_ACADOS" = "1" ]; then
  echo ">> building acados via scripts/install_acados.sh"
  bash scripts/install_acados.sh
  echo ""
  echo "   Add to your shell profile (and 'source' it before running):"
  echo "     export ACADOS_SOURCE_DIR=$PWD/third_party/acados"
  echo "     export LD_LIBRARY_PATH=\$ACADOS_SOURCE_DIR/lib:\${LD_LIBRARY_PATH:-}"
fi

echo ""
echo ">> done.  Activate with:  source .venv/bin/activate"
echo ">> check backends with:   fastbench doctor"
echo ">> run the benchmark:     fastbench run --episodes 10"
