# alloy

**Write optimal-control models in Python. Ship them as C.**

Alloy is a symbolic compiler for optimal control. You describe dynamics, costs and constraints as
named functions over a typed expression graph; alloy differentiates them, exploits their sparsity,
and generates standalone C that runs with no Python anywhere near it.

```python
import alloy as al
import numpy as np

@al.function(al.L("x", 2), al.L("f", ...))
def rosenbrock(x: al.Expr) -> al.Expr:
    return ((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2).scalar()

rosenbrock(np.array([1.0, 2.0]))          # 100.0

grad = al.gradient(rosenbrock, "f", "x")
grad(np.array([1.0, 2.0]))                # array([-400.,  200.])
```

The first call compiled that function to C, built a shared library and cached it. There is no
interpreter behind alloy — what you test from Python is the artifact you deploy.

```bash
uv run python -m alloy.codegen mymodule:rosenbrock -o generated/
```

## What it does

- **Derivatives that stay small.** Gradients, Jacobians, Hessians and Lagrangian Hessians, dense or
  sparse, as efficient and small as possible.
- **Sparsity as a first-class property.** Structural patterns are derived symbolically, colored,
  and turned into code that computes only the nonzeros — with the pattern carried into the
  generated header.
- **Structure that survives codegen.** codegenerating the same function evaluated in a loop preserves the loop through
  differentiation and code generation, so a hundred-stage horizon produces roughly the code of a
  one-stage horizon.
- **Solvers as graph nodes.** `al.problem(...)` declares a backend-free problem and `al.solver(...)` returns a real function, so a solver
  can be nested inside a larger model and the whole thing compiles into one artifact that links
  against PIQP or IPOPT directly.
- **One C ABI.** A single CasADi-style signature per generated function, plus optional typed C++
  wrappers over it.
- **A compiler you can read.** Pure Python, with NumPy for values and SciPy only for structural
  sparsity analysis, two small intermediate representations, and a well-documented architecture.

In our benchmarks, alloy matches CasADi SX on runtime while generating a fraction of the
source — 78 KB against 4.5 MB at a 500-stage horizon, and 417 lines regardless of horizon for a
neural-network-per-node model where CasADi reaches 1.31 million and stops compiling. On
solver-in-the-loop workloads, where oracle evaluation is the thing being compared, alloy is ahead by
1.1–1.4× against a CasADi that is code-generated and compiled the same way. See
[the results](docs/results/index.md) and, for what those comparisons do and do not hold constant,
[the fairness audit](docs/results/fairness.md).

## Status

Pre-1.0 and under active development. The API still moves; breaks are deliberate and documented,
but they happen — see [Versioning](docs/dev/versioning.md).

Working today: the full expression set on the host, forward and reverse differentiation, colored
sparse Jacobians and exact sparse Lagrangian Hessians through preserved `VMAP` structure, and PIQP,
IPOPT and a generated-C SQP solver drivable from inside a compiled graph.

Open: region formation from the scalar and block lowering hints, iterative rather than recursive
passes, warm-start handover into PIQP, differentiating through a solve, and a GPU backend.

## Installation

> [!NOTE]
> Replace this with wheel installation instructions when available.

Requirements:  Python 3.12 or newer, and a C compiler.

```bash
git clone https://github.com/PREDICT-EPFL/alloy.git
cd alloy
uv sync
```

The solver interfaces additionally need the vendored PIQP and IPOPT builds, which want a Fortran
compiler and CMake:

```bash
# macOS
brew install gcc cmake
# Debian / Ubuntu
sudo apt-get install gfortran cmake build-essential

ALLOY_BUILD_SOLVERS=required uv sync     # 5-8 minutes cold
```

`uv run python -m alloy.codegen.toolchain` reports the active compiler, the cache directory and
solver discovery. Windows is not supported in version 1. Full details in
[Installation](docs/guide/installation.md).

## Documentation

The [documentation](docs/index.md) is organized for two readers at once.

| | |
| --- | --- |
| [User Guide](docs/guide/getting_started.md) | building functions, derivatives, sparsity, solvers, generating C |
| [How It Works](docs/how_it_works/architecture.md) | the architecture, the two IRs, lowering, the ABI |
| [Benchmark Results](docs/results/index.md) | measured against CasADi SX and MX, kept current |
| [Developer Guide](docs/dev/contributing.md) | contributing, conventions, solver plugins, versioning |
| [API Reference](docs/api/index.md) | the public surface, generated from source |

If you already know CasADi, JAX, tinygrad or MLIR, start with
[Alloy next to its neighbours](docs/how_it_works/comparison.md): what alloy took from each, and
where it deliberately differs.

`internal/` holds the library roadmap and frozen design notes — kept in the repository for the
record, deliberately not published: `internal/paper.md` is the paper's scope and narrative,
`internal/todo.md` the actionable list, and `internal/notes/` the frozen history.

## License

TBD.
