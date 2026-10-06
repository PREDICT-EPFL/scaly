# Installation

Scaly supports Python 3.12 or newer on Linux and macOS. Numerical function calls
compile generated C, so the Python package needs access to a C compiler even
when the application itself is entirely Python.

## The core package and solver plugins

The core package provides symbolic expressions, functions, derivatives, and
C generation. It can be installed in a uv project or a Python environment:

```bash
uv add scaly
# or
pip install scaly
```

Optimization solvers are optional packages:

| Solver | uv command | pip command |
| --- | --- | --- |
| PIQP, for convex quadratic problems | `uv add "scaly[piqp]"` | `pip install "scaly[piqp]"` |
| IPOPT, for nonlinear problems | `uv add "scaly[ipopt]"` | `pip install "scaly[ipopt]"` |
| Scaly SQP, for nonlinear problems | `uv add "scaly[sqp]"` | `pip install "scaly[sqp]"` |
| All three | `uv add "scaly[solvers]"` | `pip install "scaly[solvers]"` |

A solver plugin connects the symbolic problem to a native optimization library.
The solver wheels include those native libraries. If no wheel matches the
platform, a source build needs additional tools, described in
[Contributing](../dev/contributing.md#setup). The
[backend guide](solver_backends.md) compares the supported problems and
solver options.

## Compiler selection

Scaly normally uses `cc` from `PATH`. Apple's command-line developer tools
provide it on macOS, and `build-essential` provides it on Debian and Ubuntu.
`SCALY_CC` selects a different compiler. If it is unset, Scaly also respects
`CC` before falling back to `cc`.

For example, this selects Clang for one process:

```bash
SCALY_CC=clang uv run scaly_toolchain
```

## Tested environments

Scaly is tested every night in these environments, with Python 3.14 and, on Ubuntu 24.04, also 3.12.
Other compilers and systems may work, but they are not tested.

| System | Architectures | C compilers |
| --- | --- | --- |
| Ubuntu 22.04 | x86-64, arm64 | GCC 11, and Clang 14 on x86-64 |
| Ubuntu 24.04 | x86-64, arm64 | GCC 13, and Clang 18 on x86-64 |
| Ubuntu 26.04 | x86-64, arm64 | GCC 15, and Clang 22 on x86-64 |
| macOS 15 | arm64, x86-64 | Apple Clang 17 (Xcode 16.4) |
| macOS 26 | arm64, x86-64 | Apple Clang 21 (Xcode 26) |

The selected executable must exist. An invalid override does not fall back to
the system compiler. [Environment variables](env_vars.md) describes the
compiler and cache settings.

## Reading the toolchain report

`scaly_toolchain` reports the selected compiler, the compiled-function cache,
the native compilation settings, and the installed solver libraries:

```bash
uv run scaly_toolchain
```

In an environment installed with pip, the command is `scaly_toolchain`.
An excerpt from a Linux installation with IPOPT available looks like this.
Paths are shortened, and other installed solvers are omitted:

```text
Scaly native toolchain
  cache root: /home/user/.cache/scaly/jit
  cc: /usr/bin/cc (PATH)
  native build recipe:
/* Scaly build recipe
 * CPU baseline: host-local native CPU
 * lanes=8, dialect=gnu, vector_libm=glibc, reciprocal=False
 * Math library: glibc x86-64; tanh requires glibc >= 2.35
 * gcc -O3 -march=native -fno-math-errno -c module.c
 * clang -O3 -march=native -fno-math-errno -c module.c
 * Link with: -lmvec -lm
 */
  solver source: plugin
  include dirs: .../scaly_ipopt/include, ...
  lib dirs: .../scaly_ipopt/lib, ...
  ipopt: .../scaly_ipopt/lib/libipopt.so
  ipopt discoverable: True
  ipopt loadable: True
  ...
  JIT solver flags: -I... -L... -Wl,-rpath,... -lipopt ...
```

The native build recipe describes the detected CPU and vector math settings.
Its `gcc` and `clang` lines are the suggested commands for building exported
code. Numerical evaluation from Python uses the same CPU and math flags but
compiles with `SCALY_CC_OPT`, `-O2` by default, as described in
[environment variables](env_vars.md#compiler-selection-and-optimization).
Lane width and library selection vary by machine. The
[code generation guide](codegen.md) explains these choices and how to select
them for exported code.

`discoverable` means Scaly found the library and its C headers. `loadable`
also checks that the operating system can load the library. A library can be
present but fail to load because one of its native dependencies is unavailable.
The final line reports the include and link flags used for solver compilation,
not all optimization flags used to compile a model.

A `cc: <missing>` line prevents numerical evaluation of any function. A missing
solver library only affects functions that use that solver. For an installed
plugin, failure to locate or load its native library raises
[`sc.SolverLibraryError`](../api/solvers.md#scaly.solvers.paths.SolverLibraryError).
Missing compiler and solver-library problems can therefore be diagnosed
separately.

## A numerical check

Importing scaly alone does not check C compilation. Evaluating this function
checks the numerical path as well:

```python
import numpy as np
import scaly as sc


@sc.function(sc.arg("x", 3), outputs=sc.arg("energy"))
def energy(x: sc.Expr) -> sc.Expr:
    return sc.sumsqr(x)


value = energy(np.array([1.0, 2.0, 3.0]))
np.testing.assert_allclose(value, 14.0)
print(value)
# 14.0
```

This example needs no solver plugin. [Getting started](getting_started.md)
introduces the symbolic model and adds an optimization solve.
