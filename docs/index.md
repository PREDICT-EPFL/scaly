---
title: Home
---

Scaly is a modelling and code-generation library for optimal control. You
describe dynamics, costs and constraints once, and use the same description from
prototyping in Python to deployment using generated C code.

!!! note "Early release"
    Scaly is still in development and the public API can still change between minor
    versions. See [versioning policy](dev/versioning.md) for more details.

```python
import numpy as np
import scaly as sc

N = 20  # the decision vector w stacks N + 1 states of size 2, then N controls

@sc.function(sc.G(sc.L("z", 2), sc.L("u", 1), sc.L("znext", 2)), sc.L("defect", ...))
def defect(inputs):
    z, u, znext = inputs
    return z + 0.1 * sc.concat([z[1:], u]) - znext

@sc.problem(vars=sc.L("w", 3 * N + 2), params=sc.L("z0", 2))
def multiple_shooting(w, z0):
    zs, us = w[: 2 * N + 2], w[2 * N + 2 :]
    defects = sc.vmap(defect, N, {"z": zs[:-2], "u": us, "znext": zs[2:]})  # one loop, not N copies
    return sc.ProblemSpec(minimize=sc.sumsqr(zs) + 0.1 * sc.sumsqr(us), eq=(zs[:2] - z0, defects))

solve = sc.solver(multiple_shooting, "ipopt")
w_opt, *_ = solve(np.array([1.0, 0.0]))
```

Behind the scenes, scaly traces the costs, constraints and their derivatives,
generates and compiles on the fly C code to call them within the solver. You can
also generate the same C code in a specific directory so you can embed it into
an external application, either from Python:

```python
from pathlib import Path
from scaly.codegen import write_module

write_module(solve, Path("generated/"))  # generated/multiple_shooting_ipopt.h and .c
```

or from the command line, naming the module and the function in it:

```bash
uv run scaly_codegen mymodule:solve -o generated/
```

## Where to start

<div class="grid cards" markdown>

- **User guide**

    After [Installation](guide/installation.md), go to [Getting started](guide/getting_started.md) for
    a quick overview of the library. The rest of the guide
    covers [functions](guide/functions.md), [derivatives](guide/derivatives.md),
    [sparsity](guide/sparsity.md), [solvers](guide/solvers.md) and
    [code generation](guide/codegen.md).

- **How it works**

    Start with the [Compiler architecture](how_it_works/architecture.md), then dive deeper into the
    [intermediate representations](how_it_works/ir.md), [lowering and optimization](how_it_works/lowering.md),
    [differentiation](how_it_works/autodiff.md), the [generated code](how_it_works/generated_interface.md)
    and the [solvers](how_it_works/solvers.md), or check the other [projects that have
    influenced scaly](how_it_works/influences.md).

- **Benchmarks**

    To see how the generated code compares with CasADi's, read the [headline
    results](benchmarks/index.md) on scalability microbenchmarks and full
    closed-loop controller benchmarks on actual systems.

- **Developer guide and API reference**

    Explore the [codebase structure](dev/codebase.md) and our
    [conventions](dev/conventions.md) if you want to start
    [contributing](dev/contributing.md). The [API reference](api/index.md) lists
    the public names, generated from the docstrings.

</div>

## Acknowledgements

Scaly is developed by:

- Tudor Oancea (main developer)
- Colin N. Jones (methods and math)

All contributors are part of the [Predictive control
lab](https://www.epfl.ch/labs/la3/) from EPFL. This project is funded by the
[Swiss National Science Foundation](https://www.snf.ch) through the [NCCR
Automation](https://nccr-automation.ch) (grant agreement 51NF40_180545).
