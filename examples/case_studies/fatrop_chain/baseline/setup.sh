#!/bin/sh
# Pin the authors' benchmark repository and apply the one-line fix it needs under rockit 0.6.
# rockit itself comes from PyPI through `uv run --with rockit-meco==0.6.7`, CasADi 3.8.0 (and the
# Fatrop 1.1.8 and IPOPT 3.14.11 it ships) from the project environment.
set -e
here=$(cd "$(dirname "$0")" && pwd)
dest="$here/third_party/fatrop_benchmarks"
if [ ! -d "$dest" ]; then
  git clone -q https://gitlab.kuleuven.be/robotgenskill/fatrop/fatrop_benchmarks.git "$dest"
fi
cd "$dest"
git checkout -q 9e7025aeb91ca1178b953cfa0f1fc7a0832536e2
git apply --check "$here/fatrop_benchmarks.patch" 2>/dev/null && git apply "$here/fatrop_benchmarks.patch"
echo "fatrop_benchmarks at $(git rev-parse --short HEAD) in $dest"
