#!/bin/sh
# Set up the Altro.jl side of the ALTRO study in baseline/third_party (gitignored):
#   julia/  Julia 1.10.12 (the LTS line Altro 0.5 supports), from julialang.org
#   depot/  its package depot, so nothing is written to ~/.julia
# and instantiate baseline/julia_env, whose Manifest pins Altro 0.5.0, TrajectoryOptimization 0.7.1,
# RobotDynamics 0.4.8, RobotZoo 0.3.1, BenchmarkTools and JSON. Writes baseline/env.sh, which
# compare.py reads. A first run precompiles for several minutes.
set -e
here=$(cd "$(dirname "$0")" && pwd)
tp="$here/third_party"
mkdir -p "$tp"
if [ ! -x "$tp/julia/bin/julia" ]; then
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) url=https://julialang-s3.julialang.org/bin/mac/aarch64/1.10/julia-1.10.12-macaarch64.tar.gz ;;
    Darwin-x86_64) url=https://julialang-s3.julialang.org/bin/mac/x64/1.10/julia-1.10.12-mac64.tar.gz ;;
    Linux-x86_64) url=https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.12-linux-x86_64.tar.gz ;;
    Linux-aarch64) url=https://julialang-s3.julialang.org/bin/linux/aarch64/1.10/julia-1.10.12-linux-aarch64.tar.gz ;;
    *) echo "no Julia build for $(uname -s)-$(uname -m)" >&2; exit 1 ;;
  esac
  curl -fsSL "$url" -o "$tp/julia.tar.gz"
  mkdir -p "$tp/julia" && tar -xzf "$tp/julia.tar.gz" -C "$tp/julia" --strip-components=1 && rm "$tp/julia.tar.gz"
fi
export JULIA_DEPOT_PATH="$tp/depot"
"$tp/julia/bin/julia" --project="$here/julia_env" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
cat > "$here/env.sh" <<ENV
export JULIA="$tp/julia/bin/julia"
export JULIA_DEPOT_PATH="$tp/depot"
export ALTRO_ENV="$here/julia_env"
ENV
echo "Altro.jl environment ready; see $here/env.sh"
