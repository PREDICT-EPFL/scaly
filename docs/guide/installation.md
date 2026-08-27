# Installation

Alloy requires Python 3.12 or newer and either Linux or macOS. At runtime it only depends on NumPy and a C compiler.

## The library

```bash
git clone https://github.com/PREDICT-EPFL/alloy.git
cd alloy
uv sync
```

That is enough to build functions, differentiate them, generate C and call it.

Alloy is not on PyPI yet, so to depend on it from your own project, point at the repository:

```bash
uv add git+https://github.com/PREDICT-EPFL/alloy.git
```

The solver backends are separate distributions in the same repository and are added the same way,
with a subdirectory:

```bash
uv add git+https://github.com/PREDICT-EPFL/alloy.git#subdirectory=plugins/alloy-ipopt
```

## A C compiler

Alloy looks for `cc` on your `PATH` — the POSIX name for the system default C compiler, which
exists on Linux, macOS and the BSDs. Set `ALLOY_CC` to override it.

```bash
uv run python -m alloy.codegen.toolchain
```

That prints the active compiler, the cache directory and the state of solver discovery. It is the
first thing to run when something will not compile.

## The solvers

`al.solver(problem, "piqp")` and `al.solver(problem, "ipopt")` need the PIQP and IPOPT plugins, which alloy vendors and builds from source. The
build needs a Fortran compiler and CMake:

```bash
# macOS
brew install gcc cmake

# Debian / Ubuntu
sudo apt-get install gfortran cmake build-essential
```

Then:

```bash
ALLOY_BUILD_SOLVERS=required uv sync
```

The first build can take up to 15 minutes; later syncs reuse the cached artifacts. Without
`ALLOY_BUILD_SOLVERS=required`, a missing native toolchain makes `uv sync` skip the solver
libraries rather than fail — everything except the solver interfaces still works, and a solve
raises `SolverLibraryError` when it needs a library that is not there.
See [Environment Variables](env_vars.md) for more details.

To force a clean rebuild, delete the plugin's `src/*/lib`, `src/*/include` and `third_party`
directories.

## Checking it works

```bash
uv run python -c "
import alloy as al
x = al.sym('x', 3)
f = al.Function('f', [x], [(x.sin() + x * x).sum()], ['x'], ['y'])
print(f([1.0, 2.0, 3.0]))
"
```

If that prints a number, the graph built, the C rendered, the compiler ran and the result came back
through the ABI.
