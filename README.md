<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/scaly-wordmark-dark.svg">
    <img alt="scaly" src="docs/assets/scaly-wordmark.svg" width="320">
  </picture>
</p>

Scaly is a modelling and code-generation library for optimal control. You
describe dynamics, costs and constraints once, and use the same description from
prototyping in Python to deployment using generated C code.

```python
import numpy as np
import scaly as sc

N = 20  # the decision vector w stacks N + 1 states of size 2, then N controls

@sc.function(2, 1, 2)  # the shapes of z, u and znext
def defect(z, u, znext):
    return z + 0.1 * sc.concat([z[1:], u]) - znext

@sc.opt.problem(vars=sc.L("w", 3 * N + 2), params=sc.L("z0", 2))
def multiple_shooting(w, z0):
    zs, us = w[: 2 * N + 2], w[2 * N + 2 :]
    defects = sc.vmap(defect, N, [(zs, 0, 2), (us, 0, 1), (zs, 2, 2)])  # one loop, not N copies
    return sc.opt.ProblemSpec(minimize=sc.sumsqr(zs) + 0.1 * sc.sumsqr(us), eq=(zs[:2] - z0, defects))

solve = sc.opt.solver(multiple_shooting, "ipopt")
w_opt, *_ = solve(np.zeros(3 * N + 2), np.zeros(3 * N + 2), np.zeros(2 * N + 2), np.zeros(0), np.array([1.0, 0.0]))
```

Behind the scenes, scaly traces the costs, constraints and their derivatives,
generates and compiles on the fly C code to call them within the solver. You can
also generate the same C code in a specific directory so you can embed it into
an external application, either from Python:

```python
from pathlib import Path
from scaly.codegen import write_module

write_module(solve, Path("generated/"))   # generated/multiple_shooting_ipopt.h and .c
```

or from the command line, naming the module and the function in it:

```bash
uv run scaly_codegen mymodule:solve -o generated/
```

## Features

- Wrap symbolic expressions in functions that you can compose freely.
  Differentiating (e.g. with `sc.gradient`) or batching (e.g. with `sc.vmap`) a
  function creates another function.
- The input-output dependencies of a function are analyzed statically to
  generate efficient code for sparse Jacobians and Hessians.
- Repeated structure (from `sc.vmap`) stays a loop. A horizon of identical
  stages compiles to one loop body, so the generated code stays small as the
  horizon grows.
- The generated C code has a stable signature that is intentionally close to
  CasADi's for easy integration with existing tools.
- Interfaces to quadratic and nonlinear optimization solvers such as PIQP and
  IPOPT are shipped with scaly. These interfaces are implemented as separate
  packages, allowing anyone to implement a custom solver plugin.
- The core library is pure Python and only depends on NumPy and SciPy. The
  generated code is self-contained C (except for solver code that may depend on
  external libraries).

## Installation

Scaly requires with Python 3.12 or newer, on Linux and macOS, and only requires
a C compiler to be pre-installed. Using [uv](docs.astral.sh/uv):
```bash
# install scaly: the compiler, its numerical methods, optimal control and the tools
uv add scaly
# or with an additional solver interface
uv add "scaly[ipopt]"
# or with all solvers
uv add "scaly[solvers]"
```
You can of course also use pip by replacing `uv add` with `pip install`.

See [Installation](docs/guide/installation.md) for more details.

## Documentation

The [documentation](docs/index.md) has a user guide, a description of how the
compiler works, benchmark results against CasADi, and the API reference.

Scaly is still in development and the public API can still change between minor
versions. See [versioning policy](docs/dev/versioning.md) for more details.

## Acknowledgements 

Scaly is developed at EPFL in the [PREDICT](https://www.epfl.ch/labs/la3/)
group. The research behind it is funded by the [Swiss National Science
Foundation](https://www.snf.ch) through the [NCCR
Automation](https://nccr-automation.ch).

## AI usage disclosure 

This project has been substantially developed using AI coding agents (Claude,
Codex and others), under the direction and review of the authors, who are
responsible for the result.

## License

BSD-2-Clause. See [LICENSE.md](LICENSE.md).
