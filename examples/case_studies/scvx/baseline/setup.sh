#!/bin/sh
# Set up the OpenSCvx side of the SCvx study in baseline/third_party (gitignored):
#   OpenSCvx/  OpenSCvx/OpenSCvx at 9f23a62 (2026-09-21), for examples/rocket/6DoF_pdg.py
#   venv/      Python 3.12 with openscvx 0.5.3.dev57 (the PyPI build of that commit) and the versions the
#              study ran: jax/jaxlib 0.11.2, cvxpy 1.9.3, qoco 0.3.2, clarabel 0.11.1, diffrax 0.7.2,
#              equinox 0.13.8, numpy 2.5.3, scipy 1.18.1
# run_openscvx.py points OpenSCvx's compilation cache (OPENSCVX_CACHE_DIR) into a directory it is given,
# so nothing is written to ~/Library/Caches. Writes baseline/env.sh.
set -e
here=$(cd "$(dirname "$0")" && pwd)
tp="$here/third_party"
mkdir -p "$tp"
[ -d "$tp/OpenSCvx" ] || git clone -q https://github.com/OpenSCvx/OpenSCvx.git "$tp/OpenSCvx"
git -C "$tp/OpenSCvx" checkout -q 9f23a62a02aec6e55b7267396e27bf5e6dfa143a
[ -x "$tp/venv/bin/python" ] || uv venv -q --python 3.12 "$tp/venv"
uv pip install -q --python "$tp/venv/bin/python" openscvx==0.5.3.dev57 jax==0.11.2 jaxlib==0.11.2 cvxpy==1.9.3 qoco==0.3.2 \
  clarabel==0.11.1 diffrax==0.7.2 equinox==0.13.8 numpy==2.5.3 scipy==1.18.1
cat > "$here/env.sh" <<ENV
export OPENSCVX="$tp/OpenSCvx"
export OPENSCVX_PYTHON="$tp/venv/bin/python"
ENV
echo "OpenSCvx ready; see $here/env.sh"
