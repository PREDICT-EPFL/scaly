# alloy

**Write optimal-control models in Python. Ship them as C.**

Alloy is a symbolic compiler for optimal control. You describe dynamics, costs and constraints as
named functions over a typed expression graph; alloy differentiates them, exploits their sparsity,
and generates efficient, small and standalone C that runs with no Python anywhere near it.

!!! warning "Under active development"
    Alloy is pre-1.0 and the API still moves. Breaks are deliberate and documented, but they do
    happen — see [Versioning](dev/versioning.md).

```python
import alloy as al

@al.function("rosenbrock", {"x": 2})
def rosenbrock(x):
    return {"f": ((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2).scalar()}

rosenbrock([1.0, 2.0])                      # 100.0

grad = al.gradient(rosenbrock, "x", "f")
grad([1.0, 2.0])                            # array([-400.,  200.])
```

The first call compiled that function to C, built a shared library and cached it — the just-in-time
path (JIT). There is no interpreter behind alloy — what you test from Python is the artifact you
deploy.

The same compiler renders the same C to files for someone else to build, which is the ahead-of-time
path (AOT):

```bash
uv run python -m alloy.codegen mymodule:rosenbrock -o generated/
```

## What it is for

- **Derivatives that stay small.** Gradients, Jacobians, Hessians and Lagrangian Hessians, dense or
  sparse, as efficient and small as possible.
- **Sparsity as a first-class property.** Structural patterns are derived symbolically, colored,
  and turned into code that computes only the nonzeros — with the pattern carried into the
  generated header.
- **Structure that survives codegen.** codegenerating the same function evaluated in a loop preserves the loop through
  differentiation and code generation, so a hundred-stage horizon produces roughly the code of a
  one-stage horizon.
- **Solvers as graph nodes.** `al.qp(...)` and `al.nlp(...)` return real functions, so a solver
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
[the results](results/index.md) and, for what those comparisons do and do not hold constant,
[the fairness audit](results/fairness.md).

## Where to start

<div class="grid cards" markdown>

- **New here**

    [Installation](guide/installation.md) then
    [Getting started](guide/getting_started.md) — one function, one derivative, the generated C.

- **Building a model**

    [Building functions](guide/functions.md), [Derivatives](guide/derivatives.md),
    [Sparsity](guide/sparsity.md), [Solvers](guide/solvers.md).

- **Curious how it works**

    [Architecture](how_it_works/architecture.md) for the shape of the compiler, or
    [Alloy next to its neighbours](how_it_works/comparison.md) if you already know CasADi, JAX,
    tinygrad or MLIR.

- **Comparing against CasADi**

    [Benchmark results](results/index.md) — measured against CasADi SX and MX, kept current.

</div>

## How it fits together

```mermaid
flowchart LR
  py["Python<br/>@al.function"] --> fn["Function<br/>expression graph"]
  fn -->|"derivatives"| fn
  fn -->|"lowering"| prog["program<br/>loops and buffers"]
  prog -->|"optimization passes"| prog
  prog -->|"rendering"| c["C source"]
  c -->|"compile"| so["shared library<br/>or files on disk"]
```

Two intermediate representations and one direction of travel. The expression graph says *what* to
compute and is what differentiation and simplification work on; the program says *how*, with
explicit loops, buffers and memory. [The architecture](how_it_works/architecture.md) has the full
map.

## Status

Alloy generates C for the full expression set on the host, differentiates it forward and in
reverse, produces colored sparse Jacobians and exact sparse Lagrangian Hessians through preserved
`map` structure, and drives PIQP, IPOPT and its own generated-C SQP solver from inside a compiled
graph.

Open work, roughly in order: region formation from the scalar and block lowering hints, iterative
rather than recursive passes, warm-start handover into PIQP, differentiating through a solve, and a
GPU backend. See [Benchmark results](results/index.md) for where it currently stands against
CasADi.
