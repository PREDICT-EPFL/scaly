#!/bin/sh
# Set up the upstream side of the Neural Process MPC study in baseline/third_party (gitignored):
#   neural_process_mpc/  jowaibel/neural_process_mpc at 2d8de66 (2026-09-29): the paper's code, i.e. the
#                        Python CasADi/Opti + IPOPT controller (src/npmpc), its C++ ports (src_cpp), the
#                        trained CNP (model/) and the real-time simulator server (scripts/run_qube_server.py).
#                        patches/*.patch are applied to it (see each file for why)
#   laopt/               PREDICT-EPFL/laopt at 8f9820a (2026-09-25, the feature/discrete_dynamics merge the
#                        upstream laOPT OCPs need), header-only, installed into prefix/
#   src/piqp/            PREDICT-EPFL/piqp v0.6.2 (the version the scaly-piqp plugin vendors), installed
#                        header-only with the -march=native -DBUILD_WITH_EIGEN_MAX_ALIGN_BYTES=ON that laOPT's
#                        docs/_pages/getting_started/installation_piqp.md asks for (both only affect PIQP's
#                        compiled library, which a header-only install does not build)
#   src/eigen/           Eigen 3.4.0: laOPT wraps Eigen internals and Homebrew ships Eigen 5, so the prefix
#                        carries its own, and every configure ignores /opt/homebrew, /usr/local and the
#                        user package registry
#   src/yaml-cpp/        yaml-cpp 0.8.0, built static
#   prefix/              where the four C++ dependencies above are installed
#   venv/                a uv-managed Python 3.12 with the upstream's requirements.txt (torch 2.5.1, casadi
#                        3.7.0, numpy 2.2.4, scipy 1.15.2, matplotlib 3.10.0, pandas 2.3.3) plus pyyaml.
#                        The casadi 3.7.0 wheel doubles as the CasADi C++ SDK (headers, cmake/ package
#                        config) and supplies "CasADi's IPOPT" (libipopt.3.dylib, 3.14.11, headers in
#                        include/coin-or). Its libraries run on the wheel's own libc++.1.0.dylib, next to the
#                        system libc++ of everything compiled here (see patches/casadi_clients_print.patch)
#   build/               baseline/CMakeLists.txt (Release, C++17, the upstream's flags): the upstream's
#                        eval_* and run_*_client programs, IPOPT variants of its laOPT clients, and the
#                        replay_laopt / replay_casadi drivers. laOPT's IPOPT targets are built twice,
#                        against the wheel's IPOPT and, with a _scaly_ipopt suffix, against Scaly's
#                        vendored IPOPT 3.14.19 (plugins/scaly-ipopt, which must be built), the IPOPT
#                        Scaly's generated solvers load. Each binary finds its libraries through its
#                        own rpath; nothing needs DYLD_LIBRARY_PATH
#   work/                scratch for generated inputs and run outputs
# Writes baseline/env.sh with the paths. CMake 4 rejects the old cmake_minimum_required of Eigen and
# yaml-cpp, hence CMAKE_POLICY_VERSION_MINIMUM. Run from anywhere; rerunning skips what is there.
set -e
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../.." && pwd)
tp="$here/third_party"
prefix="$tp/prefix"
src="$tp/src"
mkdir -p "$tp" "$src" "$tp/work"
jobs=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
cmake_common="-DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$prefix -DCMAKE_INSTALL_PREFIX=$prefix
  -DCMAKE_IGNORE_PREFIX_PATH=/opt/homebrew;/usr/local -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
  -DEigen3_DIR=$prefix/share/eigen3/cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  --no-warn-unused-cli -Wno-dev -Wno-deprecated"

# clone_rev DIR URL SHA: a full clone checked out at a commit.
clone_rev() {
  [ -d "$1/.git" ] || git clone -q "$2" "$1"
  git -C "$1" -c advice.detachedHead=false checkout -q "$3"
}
# clone_tag DIR URL TAG: a shallow clone of a tag.
clone_tag() {
  [ -d "$1/.git" ] || git -c advice.detachedHead=false clone -q --depth 1 --branch "$3" "$2" "$1"
}

clone_rev "$tp/neural_process_mpc" https://github.com/jowaibel/neural_process_mpc.git 2d8de66b37667ba37a5c1cc4f47eb506504cca1b
clone_rev "$tp/laopt" https://github.com/PREDICT-EPFL/laopt.git 8f9820ab9e6b3637d29912c593b4c9bb2c4888b1
clone_tag "$src/eigen" https://gitlab.com/libeigen/eigen.git 3.4.0
clone_tag "$src/piqp" https://github.com/PREDICT-EPFL/piqp.git v0.6.2
clone_tag "$src/yaml-cpp" https://github.com/jbeder/yaml-cpp.git 0.8.0
for patch in "$here"/patches/*.patch; do
  git -C "$tp/neural_process_mpc" apply --reverse --check "$patch" 2>/dev/null ||
    git -C "$tp/neural_process_mpc" apply "$patch"
done

if [ ! -f "$prefix/share/eigen3/cmake/Eigen3Config.cmake" ]; then
  cmake -S "$src/eigen" -B "$src/eigen/build" $cmake_common -DBUILD_TESTING=OFF -DEIGEN_BUILD_DOC=OFF \
    -DEIGEN_BUILD_PKGCONFIG=OFF >/dev/null
  cmake --install "$src/eigen/build" >/dev/null
fi
if [ ! -f "$prefix/lib/cmake/yaml-cpp/yaml-cpp-config.cmake" ]; then
  cmake -S "$src/yaml-cpp" -B "$src/yaml-cpp/build" $cmake_common -DYAML_BUILD_SHARED_LIBS=OFF \
    -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF >/dev/null
  cmake --build "$src/yaml-cpp/build" -j "$jobs" >/dev/null
  cmake --install "$src/yaml-cpp/build" >/dev/null
fi
if [ ! -f "$prefix/lib/cmake/piqp/piqpConfig.cmake" ]; then
  cmake -S "$src/piqp" -B "$src/piqp/build" $cmake_common -DCMAKE_CXX_FLAGS=-march=native \
    -DBUILD_WITH_EIGEN_MAX_ALIGN_BYTES=ON -DBUILD_WITH_TEMPLATE_INSTANTIATION=OFF -DBUILD_C_INTERFACE=OFF \
    -DBUILD_PYTHON_INTERFACE=OFF -DBUILD_MATLAB_INTERFACE=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    -DBUILD_BENCHMARKS=OFF >/dev/null
  cmake --install "$src/piqp/build" >/dev/null
fi
if [ ! -f "$prefix/lib/cmake/laopt/laoptConfig.cmake" ]; then
  cmake -S "$tp/laopt" -B "$tp/laopt/build" $cmake_common -DLAOPT_BUILD_TESTS=OFF -DLAOPT_BUILD_EXAMPLES=OFF \
    -DLAOPT_ENABLE_INSTALL=ON >/dev/null
  cmake --install "$tp/laopt/build" >/dev/null
fi

py="$tp/venv/bin/python"
[ -x "$py" ] || uv venv -q --managed-python --python 3.12 "$tp/venv"
uv pip install -q --python "$py" -r "$tp/neural_process_mpc/requirements.txt" pyyaml
casadi_dir=$("$py" -c 'import casadi, os; print(os.path.dirname(casadi.__file__))')

scaly_ipopt_lib="$repo/plugins/scaly-ipopt/src/scaly_ipopt/lib/libipopt.dylib"
scaly_ipopt_include="$repo/plugins/scaly-ipopt/third_party/ipopt_install/include/coin-or"
if [ ! -f "$scaly_ipopt_lib" ] || [ ! -f "$scaly_ipopt_include/IpIpoptApplication.hpp" ]; then
  echo "Scaly's IPOPT is missing ($scaly_ipopt_lib, $scaly_ipopt_include): build the scaly-ipopt plugin" >&2
  exit 1
fi
cmake -S "$here" -B "$tp/build" $cmake_common -DNPMPC_UPSTREAM="$tp/neural_process_mpc" \
  -DCASADI_PY_DIR="$casadi_dir" -DSCALY_IPOPT_LIB="$scaly_ipopt_lib" \
  -DSCALY_IPOPT_INCLUDE_DIR="$scaly_ipopt_include" >/dev/null
cmake --build "$tp/build" -j "$jobs" >/dev/null

cat > "$here/env.sh" <<ENV
export NPMPC_PYTHON="$py"
export NPMPC_UPSTREAM="$tp/neural_process_mpc"
export NPMPC_BUILD="$tp/build"
export NPMPC_WORK="$tp/work"
export NPMPC_CASADI_DIR="$casadi_dir"
ENV
echo "Neural Process MPC baselines built in $tp/build; see $here/env.sh"
