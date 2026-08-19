# Project objective

Alloy is a pure-Python symbolic compiler for optimal-control problems: named `Function`s over a sparse typed expression graph, typed derivative requests (`al.jac(...)` / `al.grad(...)` / `al.sphess(...)`), first-class call nodes, preserved mapped structure, and mixed scalar/block lowering. It generates C through a scalar renderer, compiles via the universal C ABI on first call, and caches the resulting shared library. See `README.md` for the pitch and `docs/how_it_works/architecture.md` for how it is put together.

This repository was extracted from the `anvil` monorepo in May 2026. There is no longer any runtime coupling to anvil or tinygrad — alloy depends only on NumPy at runtime (plus PIQP and IPOPT for the Phase 5 solver bindings, both vendored). As anvil will remain a stale private project, avoid mentioning it in public surfaces such as documentation or code.

# Where anvil-side context still lives

The anvil monorepo is the place to look when a question outruns alloy's own docs. Especially useful:

- `src/anvil/optimization/` — SQP solver architecture
- `src/anvil/multistage.py` — multistage OCP formulation pattern
- `examples/tracking_nmpc/`, `examples/unbumpercars/` — the two workloads that drove alloy's design
- `docs/dev/spjacobian_scalability.md`, `docs/dev/vmap.md`, `docs/dev/jit.md`, `docs/dev/multistage.md` — design notes on scalability, vmap rewrites, JIT, multistage OCP

If you need to consult those files, ask the user to point you at the right anvil checkout. Do **not** add anvil or tinygrad imports to this repository — alloy is supposed to be self-contained.

# Documentation structure

`docs/` is published as the documentation site (Zensical, `zensical.toml`). It has five sections:

- `docs/guide/` — User Guide: installation, getting started, functions, derivatives, sparsity, solvers, codegen, visualization
- `docs/how_it_works/` — architecture, both IR dialects, lowering, differentiation, the C ABI, solver internals, and how alloy compares to CasADi/tinygrad/MLIR/JAX
- `docs/results/` — benchmark numbers against CasADi SX/MX, kept current
- `docs/dev/` — Developer Guide: contributing, conventions, solver plugins, versioning, fuzzing
- `docs/api/` — API reference, generated from docstrings by mkdocstrings

`internal/` is **not** published: `internal/roadmap.md` is the library roadmap and `internal/notes/` holds frozen design and migration notes. `BENCHMARKS.md` at the root is the benchmark and paper roadmap.

Zensical publishes everything under `docs/` and has no exclusion mechanism, which is why unpublished material lives outside that tree rather than behind a config key.

Build the site with `uv run --only-group docs zensical build`, or `serve` for a live preview.

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
- METIS → MUMPS → IPOPT → `plugins/alloy-ipopt/src/alloy_ipopt/lib/libipopt.{dylib,so}`. On macOS the hook copies the Homebrew gcc runtime (`libgfortran`, `libquadmath`, `libgcc_s`) next to `libipopt.dylib`, rewrites every load command to `@rpath/`, and adds an `@loader_path` rpath, so the library is self-contained. The whole `lib/` directory is therefore what has to travel — never just `libipopt.dylib`.
- On Linux the equivalent is `_bundle_linux_runtime`: IPOPT links with `LDFLAGS=-Wl,-rpath,\$$ORIGIN`, and the hook copies the Fortran runtime (`libgfortran`, plus `libquadmath` if the closure needs it) next to `libipopt.so`, so `ldd` resolves it to the sibling. `libgcc_s` and `libstdc++` stay on the system: they are part of every glibc distribution's base install, and bundling `libstdc++` risks pinning an old one onto other C++ libraries in the same process. As on macOS, the whole `lib/` directory is what has to travel.
  - The `$ORIGIN` escaping is load-bearing. `$$` survives configure's substitution into the Makefiles (make turns it back into `$`), and the backslash stops the recipe shell from expanding it — without the backslash the flag degrades to a bare `-Wl,-rpath` that swallows the next argument, and the link fails with `cannot find libipopt.so.3`.
  - Never pass the solver stacks' static-runtime flags as `FCFLAGS`. COIN-OR's configure sets its default with `: ${FCFLAGS:="-O2 $ADD_FCFLAGS"}`, so any `FCFLAGS=` on the command line silently drops `-O2` from the MUMPS and IPOPT Fortran; `ADD_FCFLAGS` is the variable for adding to the defaults. Those flags also never reached the link, which is C++ (`libipopt_la_LIBADD`) and reads `LDFLAGS`, not `FCFLAGS`.

Each component is skipped if its install marker already exists. To force a clean rebuild, delete the plugin's `src/*/{lib,include}/` and `third_party/` directories, or run its hatch `clean` hook.

Linux uses a built OpenBLAS; macOS uses Apple's Accelerate framework. Windows is unsupported in v1.

# Instructions

- Always format with `uv run ruff format` and run `uv run ruff check` after non-trivial edits.
- Always run unit tests after a change touching the IR, AD, or codegen paths: `uv run pytest -n=auto`.
- Code should resemble tinygrad's style — simple, dense, every line earns its place. No speculative abstractions.
- Don't introduce `anvil`, `tinygrad`, or `torch` imports. If a test workload needs PyTorch checkpoints, use `alloy.utils.load_torch_state_dict` instead of adding torch as a dependency.
- Update `docs/` when changing IR-facing behavior or the generated ABI. A new public name needs a docstring — the API reference is generated from them, and `tests/test_import_boundaries.py` pins the public surface.
- Every module carries a one-line docstring saying what it owns, and every module has a layer in `tests/test_layering.py`. Imports go down, never up; the two sanctioned exceptions are written down in `docs/how_it_works/architecture.md`.
- Prefer the plainest accurate words in prose, comments and docs. "The number stops the check from ever failing" beats "the gate is vacuous". Spell out an acronym on first use unless it is standard outside the field.
- Correctness checks have two homes and each belongs in exactly one. `tests/` covers alloy itself — IR, AD, codegen, solver plumbing — and must never import `benchmarks.problems`; import `benchmarks.harness` only to test the harness itself. A benchmark problem's own input data, formulation, parameter layout, constant pins, and backend agreement belong to that problem, in `benchmarks/problems/<problem>/checks.py`, where they gate the measurement. Benchmarks churn with the workload roadmap; tests must not. When a check could sit on either side, ask whether retiring the problem would make it meaningless — if so it belongs to the problem.
- Never write down a commit hash made on the current branch — not in code, comments, docs, tests, or commit messages. Branches land through `wt merge`, which squashes them into a single new commit on the default branch, so every hash created while working stops existing the moment the work merges and the reference is left pointing at nothing. Only hashes already on the default branch are safe to cite. To point at work done on the branch, describe it instead: name the file, function, or change.
- Never let a benchmark be the only thing exercising an IR, AD, or codegen path. When it is, copy a small self-contained reproduction into `tests/` (differential against an unrolled or NumPy reference) before changing or retiring the benchmark. Prove every new benchmark gate can actually fail by perturbing what it checks; `benchmarks/README.md` documents how the gates are wired.
- **Never edit `benchmarks/problems/*/foxglove-layout.json`.** These are not source files: each is a backup the user exported from Foxglove Desktop after arranging the panels by hand. Their schema has a lot of undocumented semantics — panel ids, the `layout` split tree, camera state, per-topic settings — that agents get wrong constantly, and a hand-written edit is silently lost the next time the user re-exports. If a change to the recorder makes a layout stale (a renamed topic, a new scene topic, a channel that no longer exists), say so and describe what to toggle in Desktop; the user makes the change there and re-exports. This is read-only for you even when the fix looks like a one-line JSON edit.

# Naming conventions

- Derivative suffixes: `_grad`, `_jac`, `_hess`. Never spell out `_gradient`, `_jacobian`, `_hessian` in identifiers.
- Multistage stage variables: `zprev`, `z`, `znext`. Never `zm`/`zp`.
