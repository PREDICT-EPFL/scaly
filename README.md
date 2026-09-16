<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/scaly-wordmark-dark.svg">
    <img alt="scaly" src="docs/assets/scaly-wordmark.svg" width="320">
  </picture>
</p>

Scaly is a symbolic compiler for optimal control, written in Python. You write dynamics, costs and
constraints as named functions over a typed expression graph. Scaly differentiates them, works out
their sparsity, and generates standalone C. The C is compiled and cached on the first call from
Python, and the same C can be written to disk for a build that has no Python in it.

```python
import scaly as sc
import numpy as np

@sc.function(sc.L("x", 2), sc.L("f", ...))
def rosenbrock(x: sc.Expr) -> sc.Expr:
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

rosenbrock(np.array([1.0, 2.0]))          # array(100.)

grad = sc.gradient(rosenbrock, "f", "x")
grad(np.array([1.0, 2.0]))                # array([-400.,  200.])
```

The first call lowered the graph, rendered C, compiled a shared library and cached it. There is no
Python evaluator behind these calls, so the values you test against are computed by the same C you
ship. To render that C to files instead:

```bash
uv run python -m scaly.codegen mymodule:rosenbrock -o generated/
```

## What it does

- Derivatives are graphs. `sc.gradient`, `sc.jacobian`, `sc.hessian`, `sc.lagrangian_hessian` and
  their sparse forms return ordinary functions, which compile, nest and differentiate again.
- Sparsity is computed from the graph structure. Jacobians and Hessians are colored, the generated
  code computes only the nonzeros, and the header carries the pattern as index tables.
- Repetition is preserved. `sc.vmap` evaluates one function over a horizon of stages as a single
  node, and that node stays a loop through differentiation and code generation.
- Solvers are functions. `sc.problem` declares a problem without naming a backend, and `sc.solver`
  turns it into a function that calls PIQP, IPOPT or scaly's own generated SQP, and can sit inside
  a larger graph.
- Every generated function has the same C signature, the one CasADi uses, plus optional typed C++
  wrappers.
- The compiler is about ten thousand lines of Python. NumPy holds values and SciPy is used for
  structural sparsity analysis. There are no other runtime dependencies.

[Benchmark results](docs/results/index.md) compares generated Hessians and closed-loop controllers
against CasADi, and [Are the comparisons fair?](docs/results/fairness.md) says what each comparison
holds constant. The outcome depends on the workload, and both pages say where scaly is slower.

## Status

Scaly is at version 0.1.0a1. The public API can still change between minor versions;
[Versioning](docs/dev/versioning.md) says what a change is allowed to break.

Working today: the full expression set on the host, forward and reverse differentiation, colored
sparse Jacobians and exact sparse Lagrangian Hessians through preserved `vmap` structure, and
PIQP, IPOPT and a generated-C SQP solver callable from inside a compiled graph. Linux and macOS,
Python 3.12 or newer.

Not yet: differentiating through a solver call (its derivatives are zero), warm-start handover
into PIQP, Windows, and any target other than the host CPU.

## Installation

Requirements: Python 3.12 or newer and a C compiler.

```bash
git clone https://github.com/PREDICT-EPFL/scaly.git
cd scaly
uv sync
```

The solver plugins build vendored PIQP and IPOPT, which need CMake and a Fortran compiler:

```bash
# macOS
brew install gcc cmake
# Debian / Ubuntu
sudo apt-get install gfortran cmake build-essential

SCALY_BUILD_SOLVERS=required uv sync     # 5 to 8 minutes cold
```

`uv run python -m scaly.codegen.toolchain` reports the active compiler, the cache directory and
which solvers were found. Full details in [Installation](docs/guide/installation.md).

## Documentation

| | |
| --- | --- |
| [User guide](docs/guide/getting_started.md) | installation, building functions, derivatives, sparsity, solvers, generating C |
| [How it works](docs/how_it_works/architecture.md) | the architecture, the two dialects, lowering, differentiation, the C ABI |
| [Benchmark results](docs/results/index.md) | measured against CasADi SX and MX |
| [Developer guide](docs/dev/contributing.md) | contributing, conventions, solver plugins, versioning |
| [API reference](docs/api/index.md) | the public names, generated from the docstrings |

If you know CasADi, JAX, tinygrad or MLIR, [Next to its neighbours](docs/how_it_works/comparison.md)
says what scaly took from each and what it does differently.

`internal/` holds the todo list and frozen design notes. It is tracked but not published.

## Acknowledgements

Scaly is developed at EPFL in the PREDICT group. The research behind it is funded by the Swiss
National Science Foundation through NCCR Automation (grant agreement 51NF40_180545).

Parts of the code and of the documentation were written with AI coding agents (Claude, Codex and
others), under the direction and review of the authors, who are responsible for the result.

## License

BSD-2-Clause. See [LICENSE.md](LICENSE.md).
