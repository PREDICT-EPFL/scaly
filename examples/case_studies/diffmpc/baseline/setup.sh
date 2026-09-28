#!/bin/sh
# Pin the upstream code of the DiffMPC study into baseline/third_party (gitignored):
#   diffmpc      ToyotaResearchInstitute/diffmpc: the solver and the benchmark scripts
#                (benchmarking/reinforcement-learning/benchmark_{diffmpc,mpcpytorch,trajax}.py, utils.py)
#   mpc.pytorch  diffmpc/mpc.pytorch, the fork DiffMPC pins as a submodule (batched LU in lqr_step.py)
#   trajax       google/trajax at the commit DiffMPC's trajax environment pins
# The Python environments come from PyPI through the uv commands in run_baselines.py: JAX 0.5.3 for
# DiffMPC and trajax (DiffMPC needs jnp.matvec, 0.4.38 on; trajax needs jax.tree_map, gone in 0.6),
# PyTorch for mpc.pytorch. Theseus is not set up: it ships manylinux x86_64 wheels only.
set -e
here=$(cd "$(dirname "$0")" && pwd)
tp="$here/third_party"
mkdir -p "$tp"
[ -d "$tp/diffmpc" ] || git clone -q https://github.com/ToyotaResearchInstitute/diffmpc.git "$tp/diffmpc"
git -C "$tp/diffmpc" checkout -q 146be51053fe45675d8f4d773627983f74ec0539
[ -d "$tp/mpc.pytorch" ] || git clone -q https://github.com/diffmpc/mpc.pytorch.git "$tp/mpc.pytorch"
git -C "$tp/mpc.pytorch" checkout -q b78ae8853c10c45b3f208ae47350a8da7705f910
[ -d "$tp/trajax" ] || git clone -q https://github.com/google/trajax.git "$tp/trajax"
git -C "$tp/trajax" checkout -q c94a637c5a397b3d4100153f25b4b165507b5b20
echo "diffmpc, mpc.pytorch and trajax pinned in $tp"
