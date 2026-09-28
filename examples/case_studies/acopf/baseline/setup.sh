#!/bin/sh
# Set up the AC-OPF study's baselines in baseline/third_party (gitignored):
#   rosetta-opf  lanl-ansi/rosetta-opf: the JuMP model (jump.jl) and the ExaModels model (examodels.jl)
#   pglib-opf    power-grid-lib/pglib-opf at v23.07: the cases
#   julia/       Julia 1.10.12, and depot/ its package depot, so nothing is written to ~/.julia
# and instantiate baseline/julia_env, whose Manifest pins PowerModels 0.21.6, JuMP 1.31.2, Ipopt.jl 1.16.0,
# Ipopt_jll 300.1400.1902 (Ipopt 3.14.19, the version Scaly's plugin builds), ExaModels 0.12.0 and
# NLPModelsIpopt 0.11.3. rosetta-opf's examodels.jl targets ExaModels 0.7; examodels_v0.12.patch renames
# its calls for 0.12, and compare.py applies it to a copy. Writes baseline/env.sh, which compare.py reads.
set -e
here=$(cd "$(dirname "$0")" && pwd)
tp="$here/third_party"
mkdir -p "$tp"
[ -d "$tp/rosetta-opf" ] || git clone -q https://github.com/lanl-ansi/rosetta-opf.git "$tp/rosetta-opf"
git -C "$tp/rosetta-opf" checkout -q 496d02671eb123368928b8e384e84f900db7cc3c
[ -d "$tp/pglib-opf" ] || git clone -q https://github.com/power-grid-lib/pglib-opf.git "$tp/pglib-opf"
git -C "$tp/pglib-opf" checkout -q dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3
if [ ! -x "$tp/julia/bin/julia" ]; then
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) url=https://julialang-s3.julialang.org/bin/mac/aarch64/1.10/julia-1.10.12-macaarch64.tar.gz ;;
    Linux-x86_64) url=https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.12-linux-x86_64.tar.gz ;;
    *) echo "no Julia build set up for $(uname -s)-$(uname -m)" >&2; exit 1 ;;
  esac
  curl -fsSL "$url" -o "$tp/julia.tar.gz"
  mkdir -p "$tp/julia" && tar -xzf "$tp/julia.tar.gz" -C "$tp/julia" --strip-components=1 && rm "$tp/julia.tar.gz"
fi
export JULIA_DEPOT_PATH="$tp/depot"
"$tp/julia/bin/julia" --project="$here/julia_env" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
cat > "$here/env.sh" <<ENV
export JULIA="$tp/julia/bin/julia"
export JULIA_DEPOT_PATH="$tp/depot"
export OPF_ENV="$here/julia_env"
ENV
echo "AC-OPF baselines ready; see $here/env.sh"
