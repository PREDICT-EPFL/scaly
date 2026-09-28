# Scaly

A pure-Python symbolic compiler for optimal-control problems: named `Function`s over one sparse
typed expression graph, typed derivative requests, first-class call nodes, preserved mapped
structure, mixed scalar/block lowering. It generates C, compiles through a universal C ABI on first
call, and caches the resulting shared library.

Read before changing anything:

- [Architecture](docs/how_it_works/architecture.md) — the pipeline, the two dialects, what each stage owns
- [The codebase](docs/dev/codebase.md) — the package map, the import-layer table, *Where to add things*
  (a new scalar operation touches seven files) and the rules that keep the tree in shape. These
  rules bind agents exactly as they bind human contributors; follow them, do not work around them.
- [Conventions](docs/dev/conventions.md) — naming, code style, which side of the
  tests-versus-benchmarks line a check belongs on
- [Contributing](docs/dev/contributing.md) — the checks, where tests live, known flakes

Everything under `docs/` is published to the documentation site, all of it, because Zensical has no
exclusion mechanism. Anything unpublished lives in `internal/`:

- `internal/todo.md` is the single actionable list. Each item links its rationale to
  `docs/results/fairness.md` instead of restating it.
- `internal/notes/` holds frozen design and migration notes, including the completed benchmark and
  solver-plugin build-out.

The published `docs/results/fairness.md` owns what comparisons hold constant, the measurement
protocol, the reference machine and the evidence behind each rule. Put rationale in `fairness.md` and put the corresponding one-line task in `todo.md`. Do not repeat the same prose.

## Commands

- `uv run pytest -n=auto` — the suite; it collects `tests/` and `plugins/`
- `uv run ruff format` · `uv run ruff check` · `uv run ty check` — always through `uv run`. A bare
  `ruff` or `ty` is likely a globally installed one at a different version, which will format the
  tree or report types differently from the pinned tools CI uses.
- `uv run --only-group docs zensical serve` — docs preview, `build` to render into `site/`
- `uv run path/to/script.py` runs a script. Not `uv run python path/to/script.py`: `uv run` takes
  the file directly, and the extra `python` buys nothing. Never activate the venv.

Run the whole suite for any change touching the IR, differentiation or code generation. Those paths
break subtly and are expensive to debug later.

A test needing a built solver is marked `@pytest.mark.solver("piqp"|"ipopt")` and the root
`conftest.py` handles the skip. Never hand-roll a "is the solver loadable" condition.

## Words

Use these; they were settled deliberately.

**Oracle** — a function that supplies a solver with a problem quantity or derivative: an objective,
constraint, Lagrangian, gradient, Jacobian or Hessian. Not "callback".

**Function evaluation** — all the work needed to evaluate the oracles a solver asks for, values and
derivatives together. Not "objective evaluation", when the whole set is meant.

Identifier spellings, several of which reach the generated C:

- `_grad`, `_jac`, `_hess` — never `_gradient`, `_jacobian`, `_hessian`. The long forms exist only
  as the user-facing wrappers `sc.gradient`, `sc.jacobian`, `sc.hessian`.
- `zprev`, `z`, `znext` for multistage stage variables — never `zm`/`zp`.
- `Expr*` for expression-dialect names, `Program*` for program-dialect names. There is no third
  vocabulary; "semantic IR" and the `P`-prefixed spellings are gone.
- Derived outputs are `{kind}_{of}_{wrt}`. These become C symbols and the sparsity-table prefixes in
  the generated header, so renaming one moves symbols in everyone's build.

## Traps

- **Never edit `benchmarks/problems/*/foxglove-layout.json`.** They are Foxglove Desktop exports,
  not source: undocumented panel ids, split trees and camera state that a hand edit gets wrong and
  that the next re-export silently discards. If a recorder change leaves one stale, say what to
  toggle in Desktop and let the user re-export.
- **Never cite a commit hash made on the current branch** — not in code, comments, docs, tests or
  commit messages. Branches land squashed through `wt merge`, so such a hash stops existing the
  moment the work merges. Name the file, function or change instead. Hashes already on `main` are
  safe to cite.
- **No `tinygrad` or `torch` imports.** NumPy and SciPy are the only runtime dependencies and that is
  worth defending; for PyTorch checkpoints use `scaly.utils.load_torch_state_dict`.
- **A new module needs an `IMPORT_LAYERS` entry in `tests/test_import_layering.py`** and a one-line docstring
  saying what it owns. Imports go down import layers, never up. A new public name needs a docstring too:
  the API reference is generated from them and `tests/test_import_boundaries.py` pins the surface.
- **Never let a benchmark be the only thing exercising an IR, AD or codegen path.** Copy a small
  self-contained reproduction into `tests/`, differential against an unrolled or NumPy reference,
  before changing or retiring the benchmark. Prove a new gate can fail by perturbing what it checks.
- **The vendored solver hooks are their own world.** The `$ORIGIN` escaping, the METIS
  `-march=native` strip, and the macOS `install_name` rewriting and re-signing are each explained in a
  comment beside the code in `plugins/*/hatch_build.py` — read them there before editing.
  `internal/notes/vendored_solvers.md` tracks what is still open. A cold rebuild is 5 to 8 minutes.
