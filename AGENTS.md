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

# tinygrad reference (GPU work)

A shallow clone of tinygrad is kept at `/tmp/tinygrad` as the reference for GPU codegen / scheduling / renderer design. Re-clone if missing: `git clone --depth=1 https://github.com/tinygrad/tinygrad.git /tmp/tinygrad`.

Use it for *patterns only* — do not import or copy code wholesale. Useful entry points:

- `tinygrad/renderer/cstyle.py` — `CStyleLanguage`, `CUDARenderer`, `MetalRenderer`, `OpenCLRenderer`. Look here for thread-binding (`code_for_workitem`), shared-memory prefix (`smem_prefix`), barrier syntax, and per-backend dtype maps.
- `tinygrad/runtime/ops_cuda.py` — how to call `cuLaunchKernel` via libcuda + ctypes (no PyCUDA dep), how to detect compute capability via `cuDeviceComputeCapability`.
- `tinygrad/codegen/gpudims.py` — `get_grouped_dims`, `add_gpudims`: how to map logical loop ranges to `(blockIdx, threadIdx)` axes with backend dimension caps. The key idea: schedule passes mark axes as GLOBAL/LOCAL/REDUCE *before* the renderer touches them.
- `tinygrad/schedule/rangeify.py`, `tinygrad/codegen/__init__.py` — how kernel splitting works: reductions over thread-private state get split into separate kernels with a global scratch buffer in between. Relevant whenever a "thread-bound elementwise → cross-thread reduce" pattern shows up in alloy.

# CUDA build setup on this workstation

- Driver: 580.x supports CUDA up to 13.0. `/usr/local/cuda` symlinks to nvcc 13.2 by default, which emits PTX too new for the driver (CUDA error 222 `unsupported toolchain`).
- Use `/usr/local/cuda-13.0/bin/nvcc` explicitly. The alloy CUDA runtime auto-detects this; if you're invoking nvcc manually, don't trust `/usr/local/cuda/bin/nvcc`.
- Do **not** upgrade the driver to get nvcc 13.2 working — driver upgrades are heavyweight and the alloy auto-detection makes them unnecessary.

# Build hook notes (skipping IPOPT during dev)

If `uv sync` fails on OpenBLAS/IPOPT (it does on this workstation), create empty stubs to skip the IPOPT build:

```
mkdir -p src/alloy/include/coin-or && touch src/alloy/include/coin-or/IpStdCInterface.h
touch src/alloy/lib/libipopt.so
```

The build hook checks for these files and skips the IPOPT stack when they exist. Tests that actually need IPOPT (`tests/alloy/test_solvers.py`, `tests/alloy/test_solver_nesting.py`) will fail at import time; skip them while developing GPU code.

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
- build backend: hatchling with a custom hook (`hatch_build.py`) that vendors PIQP and IPOPT as shared libraries

# Cookbook

- Add a dependency: `uv add <name>` (regular) or `uv add --dev <name>` (dev only).
- Sync: `uv sync` (the first sync from a fresh checkout triggers PIQP + IPOPT builds, ~5-8 min).
- Run a module: `uv run python <path>.py`. Always `uv run python` — never activate the venv.
- Type check: `uv run ty check`
- Lint: `uv run ruff check`
- Format: `uv run ruff format`
- Tests: `uv run pytest -n=auto tests/`

# Build hook notes

`hatch_build.py` builds the vendored solver stack on first sync:

- PIQP (with Eigen 3.4.1 and Blasfeo) → `src/alloy/lib/libpiqpc.{dylib,so}`
- METIS → MUMPS → IPOPT → `src/alloy/lib/libipopt.{dylib,so}` with statically linked libgfortran/libgcc/libstdc++ so the resulting library is redistributable.

Each component is skipped if its install marker already exists. To force a clean rebuild, delete `src/alloy/lib/`, `src/alloy/include/`, and `third_party/`, or run the hatch `clean` hook.

Linux uses a built OpenBLAS; macOS uses Apple's Accelerate framework. Windows is unsupported in v1.

# Instructions

- Always format with `uv run ruff format` and run `uv run ruff check` after non-trivial edits.
- Always run unit tests after a change touching the IR, AD, or codegen paths: `uv run pytest -n=auto tests/`.
- Code should resemble tinygrad's style — simple, dense, every line earns its place. No speculative abstractions.
- Don't introduce `anvil` or `tinygrad` imports. If a test workload needs PyTorch checkpoints, use `torch.load` (already in the dev group).
- Update `docs/` when changing IR-facing behavior or the codegenerated ABI.

# Naming conventions

- Derivative suffixes: `_grad`, `_jac`, `_hess`. Never spell out `_gradient`, `_jacobian`, `_hessian` in identifiers.
- Multistage stage variables: `zprev`, `z`, `znext`. Never `zm`/`zp`.
