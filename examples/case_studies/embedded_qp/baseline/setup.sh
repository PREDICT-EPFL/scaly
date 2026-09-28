#!/bin/sh
# Pin the upstream code of the embedded-QP study into baseline/third_party (gitignored):
#   qoco-benchmarks: the oscillating-masses problem, its CVXPY-to-QOCO conversion and the QOCOGEN timing harness
#   qoco:            QOCO's C sources, for its vendored QDLDL (the LDL^T microbenchmark's baseline)
# The Python packages come from PyPI through the uv command in compare.py's docstring:
#   qoco 0.3.2, qocogen 0.1.9, clarabel 0.11.1, osqp 1.1.3, cvxpy 1.9.3, qdldl.
# Do not install cvxpygen into that environment: importing it imports pdaqp, whose juliacall downloads
# Julia and writes ~/.julia on first import.
set -e
here=$(cd "$(dirname "$0")" && pwd)
tp="$here/third_party"
mkdir -p "$tp"
[ -d "$tp/qoco-benchmarks" ] || git clone -q https://github.com/qoco-org/qoco-benchmarks.git "$tp/qoco-benchmarks"
git -C "$tp/qoco-benchmarks" checkout -q d7e00f5880bde2503a216afac57646eb3175e053
[ -d "$tp/qoco" ] || git clone -q https://github.com/qoco-org/qoco.git "$tp/qoco"
git -C "$tp/qoco" checkout -q 019862545a79091e9c3f2d4c2c3dedb7cf0b813c
echo "qoco-benchmarks and qoco pinned in $tp"
