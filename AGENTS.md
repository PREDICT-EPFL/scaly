# Project objective

Alloy is a pure-Python symbolic IR for optimal-control problems: named `Function`s over a sparse typed expression graph, CasADi-style derivative factories (`jac:*` / `grad:*` / `hess:*` / `lam:*`), first-class call nodes, and mixed scalar/block lowering. It generates C through a scalar renderer, JIT-compiles via the universal C ABI on first call, and caches the resulting `.so`. See `README.md` and `docs/roadmap.md` for the design north star.

This repository was extracted from the `anvil` monorepo in May 2026. There is no longer any runtime coupling to anvil or tinygrad — alloy depends only on NumPy at runtime (plus PIQP and IPOPT for the Phase 5 solver bindings, both vendored).

# Where anvil-side context still lives

The anvil monorepo is the place to look when a question outruns alloy's own docs. Especially useful:

- `src/anvil/optimization/` — SQP solver architecture
- `src/anvil/multistage.py` — multistage OCP formulation pattern
- `examples/tracking_nmpc/`, `examples/unbumpercars/` — the two workloads that drove alloy's design
- `docs/dev/spjacobian_scalability.md`, `docs/dev/vmap.md`, `docs/dev/jit.md`, `docs/dev/multistage.md` — design notes on scalability, vmap rewrites, JIT, multistage OCP

If you need to consult those files, ask the user to point you at the right anvil checkout. Do **not** add anvil or tinygrad imports to this repository — alloy is supposed to be self-contained.

# Documentation structure

- `docs/roadmap.md` — phased plan, current status, exit criteria
- `docs/spec.md` — IR semantics, op set, ABI conventions
- `docs/safety_filter.md` — Phase 5 driving workload
- `docs/scalability.md` — benchmark results against CasADi SX/MX

# Tech stack

- language: Python (`>=3.12`, dev runs on 3.14)
- project configuration: `pyproject.toml`
- package manager: uv
- formatter/linter: ruff
- type checker: ty
- build backend: hatchling; each solver plugin has a custom `plugins/*/hatch_build.py` hook

# Cookbook

- Add a dependency: `uv add <name>` (regular) or `uv add --dev <name>` (dev only).
- Sync: `uv sync` (the first sync from a fresh checkout triggers PIQP + IPOPT builds, ~5-8 min).
- Run a module: `uv run python <path>.py`. Always `uv run python` — never activate the venv.
- Type check: `uv run ty check`
- Lint: `uv run ruff check`
- Format: `uv run ruff format`
- Tests: `uv run pytest -n=auto tests/ plugins/`

# Build hook notes

The per-plugin `plugins/*/hatch_build.py` hooks build the vendored solver stacks on first sync:

- PIQP (with Eigen 3.4.1 and Blasfeo) → `plugins/alloy-piqp/src/alloy_piqp/lib/libpiqpc.{dylib,so}`
- METIS → MUMPS → IPOPT → `plugins/alloy-ipopt/src/alloy_ipopt/lib/libipopt.{dylib,so}` with statically linked libgfortran/libgcc/libstdc++ so the resulting library is redistributable.

Each component is skipped if its install marker already exists. To force a clean rebuild, delete the plugin's `src/*/{lib,include}/` and `third_party/` directories, or run its hatch `clean` hook.

Linux uses a built OpenBLAS; macOS uses Apple's Accelerate framework. Windows is unsupported in v1.

# Instructions

- Always format with `uv run ruff format` and run `uv run ruff check` after non-trivial edits.
- Always run unit tests after a change touching the IR, AD, or codegen paths: `uv run pytest -n=auto tests/ plugins/`.
- Code should resemble tinygrad's style — simple, dense, every line earns its place. No speculative abstractions.
- Don't introduce `anvil`, `tinygrad`, or `torch` imports. If a test workload needs PyTorch checkpoints, use `alloy.utils.load_torch_state_dict` instead of adding torch as a dependency.
- Update `docs/` when changing IR-facing behavior or the codegenerated ABI.
- Keep correctness checks layered as described under **Where correctness checks live**. `tests/` must never import `benchmarks.problems`.

# Where correctness checks live

There are two kinds of correctness check in this repo, and each has one correct home.

- **`tests/` covers alloy itself** — the IR, AD, codegen, and solver plumbing. Nothing under `tests/` may import `benchmarks.problems`. When a test needs a specific function shape that a benchmark problem surfaced, copy a minimal reproduction of that shape into the test and check it differentially against an unrolled or NumPy reference. Importing `benchmarks.harness` is allowed, but only to test the harness itself (recorder schemas, scene builders, the artifact writer): drive it through whichever problem is cheapest and assert on harness behaviour, never on a problem's numbers.
- **`benchmarks/problems/<problem>/checks.py` covers that problem** — its input data, its formulation, the layout of its parameter vector, pins on its physical constants, and the agreement between its backends. These gate the measurement, so they run before any timing is recorded.

The split exists because the two churn at different rates. Benchmark problems follow the workload roadmap and are expected to be reshaped or retired, so IR/AD/codegen coverage that rides on one disappears with it. A problem-specific check parked in `tests/` has the mirror-image problem: it gets deleted along with the problem anyway, and it does not run where it is actually needed, which is before the numbers are taken.

So when coverage turns up in the wrong place:

- A benchmark is the only thing exercising some op or composition → copy a small artificial reproduction into `tests/`, then the benchmark is free to churn.
- A test asserts something about a problem rather than about alloy → move it into that problem's `checks.py`. The deciding question is whether retiring the problem would make the check meaningless; if so, it belongs to the problem.

A problem's `checks.py` exposes `run_checks()`, which yields `(name, outcome)` per gate from a `CHECKS` table recording whether each gate needs IPOPT or CasADi; a missing dependency yields `"skipped: ..."` rather than passing quietly. `benchmarks/run.py smoke --select problems` runs them all. Confirm each new gate can actually fail, by perturbing the thing it checks — a gate that cannot fail is worse than no gate, because it reads as coverage. Watch for perturbations that are secretly no-ops (scaling an objective does not move its argmin).

`race_cars` is the worked example. `chain_of_masses` and `bumpercars_filter` are not migrated yet: `tests/alloy/test_chain_of_masses_workload.py`, `test_chain_closed_loop.py`, and `test_bumpercars_filter_workload.py` still import their problem modules directly.

# Naming conventions

- Derivative suffixes: `_grad`, `_jac`, `_hess`. Never spell out `_gradient`, `_jacobian`, `_hessian` in identifiers.
- Multistage stage variables: `zprev`, `z`, `znext`. Never `zm`/`zp`.
