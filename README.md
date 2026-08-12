# alloy

A symbolic IR and code-generation framework for optimal control problems. Alloy expresses dynamics, costs, and constraints as named `Function`s over a sparse typed expression graph, builds CasADi-style derivative factories, lowers regions to scalar C or block kernels, and JITs the result through a universal C ABI compatible with AOT C++ consumers.

## Status

Experimental. Phases 0-4 of the roadmap are complete (symbolic core, sparse colored AD, MAP-based loop preservation, JIT-as-default execution). Phase 5 is in progress: PIQP and IPOPT ship as vendored shared libraries with `al.qp(...)` / `al.nlp(...)` opaque solver `Function`s wired on top via ctypes and generated-C solver wrappers for nested JIT/AOT use. Still open: sparse PIQP, warm-start handover, wheel repair/static native dependency cleanup, and end-to-end safety-filter assembly. See `docs/roadmap.md` and `docs/solvers.md`.

## Relationship to anvil

Alloy spun out of [anvil](https://github.com/PREDICT-EPFL/anvil) (a tinygrad-based AOT code-generation framework for optimal-control problems) in May 2026. The two projects share design lineage but no runtime code: alloy is pure-Python with NumPy as its only required dependency. While developing alloy, the anvil repository remains a useful reference for:

- the SQP solver architecture (`src/anvil/optimization/`);
- the multistage OCP formulation pattern (`src/anvil/multistage.py`);
- the tracking-NMPC and unbumpercars example workloads;
- tinygrad UOp internals and graph-rewriting techniques.

If you encounter a design question alloy hasn't answered yet, the anvil source and its `docs/dev/` notes are often a good starting point — particularly `docs/dev/spjacobian_scalability.md`, `docs/dev/vmap.md`, `docs/dev/jit.md`, and `docs/dev/multistage.md`.

## Installation

Requires Python 3.12+. A normal editable install works with only the Python toolchain:

```bash
git clone https://github.com/PREDICT-EPFL/alloy.git
cd alloy
uv sync
```

The solver bindings need native vendored libraries. When the required native toolchain is available, editable `uv sync` builds PIQP and IPOPT into their `plugins/alloy-{piqp,ipopt}/src/*/lib/` package directories; if the toolchain is missing, the editable install skips those libraries and solver calls/tests are unavailable. To require the native build (the CI path):

```bash
# macOS
brew install gcc cmake

# Linux (Debian/Ubuntu)
sudo apt-get install gfortran cmake build-essential

ALLOY_BUILD_SOLVERS=required uv sync
```

Cold solver builds take ~5-8 minutes; subsequent syncs use the cached artifacts. `uv run python -m alloy.toolchain` reports the active compiler, cache directory, and vendored solver discovery state. Windows solver builds are not supported in v1 (open an issue if you need them).

## Getting started

```python
import alloy as al

@al.function("rosenbrock", {"x": 2})
def rosenbrock(x):
  return {"f": ((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2).scalar()}

# Build a gradient through the factory and JIT-call it.
grad = rosenbrock.factory("rosenbrock_grad", ["x"], ["grad:f:x"])
print(grad([1.0, 2.0]))  # -> array of two doubles
```

Functions are sparse-typed, derivatives are pulled through the factory (`jac:*`, `grad:*`, `hess:*`, `lam:*`), and the first call compiles the C source through a universal CasADi-style ABI. The resulting `.so`/`.dylib` is cached under `$XDG_CACHE_HOME/alloy/jit` or `~/.cache/alloy/jit` (override with `ALLOY_CACHE_DIR`).

## Architecture

See `docs/roadmap.md` for the design north star and milestone history. Key documents:

- `docs/roadmap.md` — phased development plan, current status, exit criteria
- `docs/naming.md` — project-name lineage, criteria, and the leading alternative
- `docs/spec.md` — IR semantics, op set, ABI conventions
- `docs/solvers.md` — QP/NLP interfaces (`al.qp`, `al.nlp`) and PIQP/IPOPT wiring
- `docs/safety_filter.md` — Phase 5 driving workload
- `docs/scalability.md` — benchmark results against CasADi SX/MX
- `docs/vendored_solvers.md` — open issues around the vendored solver build
- `docs/native_toolchain_exploration.md` — historical notes from the conda-prefix/delocate exploration

## License

TBD.
