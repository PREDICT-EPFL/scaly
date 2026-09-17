# Installation

Scaly requires Python 3.12 or newer on Linux or macOS. At runtime it depends on NumPy and SciPy,
plus a C compiler.

## The library

```bash
git clone https://github.com/PREDICT-EPFL/scaly.git
cd scaly
uv sync
```

That is enough to build functions, differentiate them, generate C and call it.

Scaly is not on PyPI yet. To depend on it from your own project, point at the repository:

```bash
uv add git+https://github.com/PREDICT-EPFL/scaly.git
```

The solver backends are separate distributions in the same repository, added the same way with a
subdirectory:

```bash
uv add git+https://github.com/PREDICT-EPFL/scaly.git#subdirectory=plugins/scaly-ipopt
```

## A C compiler

Scaly looks for `cc` on your `PATH`, the POSIX name for the system default C compiler on Linux,
macOS and the BSDs. Set `SCALY_CC` to override it.

```bash
uv run scaly_toolchain
```

That prints the active compiler, the cache directory and the state of solver discovery. Run it
first when something will not compile.

## The solvers

`sc.solver(problem, "piqp")` and `sc.solver(problem, "ipopt")` need the PIQP and IPOPT plugins,
which scaly vendors and builds from source. The build needs a Fortran compiler and CMake:

```bash
# macOS
brew install gcc cmake

# Debian / Ubuntu
sudo apt-get install gfortran cmake build-essential
```

Then:

```bash
SCALY_BUILD_SOLVERS=required uv sync
```

A cold build takes 5 to 8 minutes; later syncs reuse the cached artifacts. Without
`SCALY_BUILD_SOLVERS=required`, a missing native toolchain makes `uv sync` skip the solver
libraries instead of failing. Everything except the solver interfaces still works, and a solve
raises `SolverLibraryError` when it needs a library that is not there. See
[Environment variables](env_vars.md).

To force a clean rebuild, delete the plugin's `src/*/lib`, `src/*/include` and `third_party`
directories.

## Checking it works

```bash
uv run python -c "
import numpy as np
import scaly as sc

@sc.function(sc.L('x', 3), sc.L('y', ...))
def f(x):
    return (x.sin() + x * x).sum()

print(f(np.ones(3)))
"
```

If that prints a number, the graph built, the C rendered, the compiler ran and the result came back
through the ABI.
