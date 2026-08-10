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
- Tests: `uv run pytest -n=auto` (collects `tests/` and `plugins/`)
- Tests needing a built solver: mark with `@pytest.mark.solver("piqp"|"ipopt")`. The root `conftest.py` skips them when the library is missing, and CI splits the suite with `-m solver` / `-m "not solver"` — never hand-roll a `solver_loadable` skipif.

# Build hook notes

The per-plugin `plugins/*/hatch_build.py` hooks build the vendored solver stacks on first sync:

- PIQP (with Eigen 3.4.1 and Blasfeo) → `plugins/alloy-piqp/src/alloy_piqp/lib/libpiqpc.{dylib,so}`
- METIS → MUMPS → IPOPT → `plugins/alloy-ipopt/src/alloy_ipopt/lib/libipopt.{dylib,so}`. On Linux the Fortran runtime is linked statically (`-static-libgfortran -static-libgcc -static-libstdc++`). macOS cannot pass those flags, so the hook instead copies the Homebrew gcc runtime (`libgfortran`, `libquadmath`, `libgcc_s`) next to `libipopt.dylib`, rewrites every load command to `@rpath/`, and adds an `@loader_path` rpath. Either way the shipped library carries no absolute reference to the build machine's toolchain, so the whole `lib/` directory is what has to travel — never just `libipopt.dylib`.

Each component is skipped if its install marker already exists. To force a clean rebuild, delete the plugin's `src/*/{lib,include}/` and `third_party/` directories, or run its hatch `clean` hook.

Linux uses a built OpenBLAS; macOS uses Apple's Accelerate framework. Windows is unsupported in v1.

# Instructions

- Always format with `uv run ruff format` and run `uv run ruff check` after non-trivial edits.
- Always run unit tests after a change touching the IR, AD, or codegen paths: `uv run pytest -n=auto`.
- Code should resemble tinygrad's style — simple, dense, every line earns its place. No speculative abstractions.
- Don't introduce `anvil`, `tinygrad`, or `torch` imports. If a test workload needs PyTorch checkpoints, use `alloy.utils.load_torch_state_dict` instead of adding torch as a dependency.
- Update `docs/` when changing IR-facing behavior or the codegenerated ABI.
- Correctness checks have two homes and each belongs in exactly one. `tests/` covers alloy itself — IR, AD, codegen, solver plumbing — and must never import `benchmarks.problems`; import `benchmarks.harness` only to test the harness itself. A benchmark problem's own input data, formulation, parameter layout, constant pins, and backend agreement belong to that problem, in `benchmarks/problems/<problem>/checks.py`, where they gate the measurement. Benchmarks churn with the workload roadmap; tests must not. When a check could sit on either side, ask whether retiring the problem would make it meaningless — if so it belongs to the problem.
- Never let a benchmark be the only thing exercising an IR, AD, or codegen path. When it is, copy a small self-contained reproduction into `tests/` (differential against an unrolled or NumPy reference) before changing or retiring the benchmark. Prove every new benchmark gate can actually fail by perturbing what it checks; `benchmarks/README.md` documents how the gates are wired.

# Naming conventions

- Derivative suffixes: `_grad`, `_jac`, `_hess`. Never spell out `_gradient`, `_jacobian`, `_hessian` in identifiers.
- Multistage stage variables: `zprev`, `z`, `znext`. Never `zm`/`zp`.
