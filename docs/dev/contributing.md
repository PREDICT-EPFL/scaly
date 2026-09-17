# Contributing

How to set up a checkout, which checks a change must pass, where tests live and what to read before
changing the compiler.

## Setup

```bash
git clone https://github.com/PREDICT-EPFL/scaly.git
cd scaly
uv sync
```

Use `uv run` for everything, for example `uv run pytest` or `uv run benchmarks/run.py`. `uv run`
takes a script path directly, so the `python` in `uv run python script.py` is redundant. Do not
activate the virtual environment by hand.

Scaly itself is pure Python, but the `scaly-piqp` and `scaly-ipopt` plugins vendor their solvers
and build them from source on the first sync. CMake comes from PyPI as a build requirement; a C++
compiler and a Fortran compiler have to be installed system-wide:

```bash
# macOS
brew install gcc

# Debian / Ubuntu
sudo apt-get install gfortran build-essential
```

Then:

```bash
SCALY_BUILD_SOLVERS=required uv sync
```

A cold build takes 5 to 8 minutes; later syncs reuse the cached artifacts. Without
`SCALY_BUILD_SOLVERS=required`, a missing native toolchain makes `uv sync` skip the solver
libraries instead of failing, and the solver tests skip with them; everything else works. See
[Environment variables](../guide/env_vars.md).

To force a clean rebuild, delete the plugin's `src/*/lib`, `src/*/include` and `third_party`
directories.

In the two vendoring plugins (`scaly-piqp`, `scaly-ipopt`), a new vendored dependency needs an entry
in `src/scaly_*/build_config.json`, which pins its version, and a row in the
`_write_third_party_notices` call of that plugin's `hatch_build.py`, which copies its license texts
into the wheel. The `test_*_notices.py` test in each of those plugins fails when a pinned dependency
has no license directory.

## The checks

```bash
uv run pytest -n=auto              # the suite, in parallel
uv run ruff check                  # lint
uv run ruff format --check         # formatting, as CI runs it; drop --check to fix
uv run ty check --error-on-warning # types, including expected-error assertions
```

Run all four before you consider a change done. `pytest` collects `tests/`, `plugins/`, and
`typing_playground/`. Strict type checking covers the assertions in `tests/typing/` and the
playground. An expected error that disappears leaves an unused ignore, which fails the check.

The root `conftest.py` checks the full-collection node-ID baseline. After adding, removing, or
renaming a test, regenerate it from the complete collection:

```bash
(
  set -e
  raw=$(mktemp)
  fresh=$(mktemp)
  trap 'rm -f "$raw" "$fresh"' EXIT
  uv run pytest --collect-only -q >"$raw" 2>&1 || true
  grep -E '^(tests|plugins|typing_playground)/[^:]+\.py::' "$raw" | LC_ALL=C sort >"$fresh"
  test -s "$fresh"
  mv "$fresh" tests/baseline/pytest_nodeids.txt
)
generation_status=$?
[ "$generation_status" -eq 0 ] && uv run pytest --collect-only -q
```

The first collection can exit nonzero because the existing baseline is stale or missing. The `grep`
keeps only pytest node IDs, including parameter IDs with spaces, and `sort` makes the file
deterministic. The final collection must pass.

A test that needs a built solver carries a marker:

```python
@pytest.mark.solver("piqp")
def test_something(): ...
```

The root `conftest.py` skips those when the library is absent, and CI splits the suite on
`-m solver` against `-m "not solver"`. Never hand-roll a "is the solver loadable" skip condition.

An xdist worker occasionally dies inside the isolated library load in the vendored-solver plugin
tests. It reproduces on unmodified checkouts, so a lone worker crash there is probably not yours.
Rerun before reading it as a failure.

## Where things live

`tests/` mirrors `src/scaly/` directory for directory, so a change to `src/scaly/passes/lowering.py`
has its tests in `tests/passes/test_lowering.py`. Outside the mirror:

- `tests/integration/` holds workload-shaped end-to-end checks.
- `tests/benchmarks/` tests the benchmark harness. The benchmark problems keep their own gates; see
  [Conventions](conventions.md#tests-against-benchmarks) for which side a check belongs on.
- `tests/typing/` holds the expected-error assertions that `ty check` covers.
- `tests/baseline/` holds `pytest_nodeids.txt` and the generated-C snapshots under `c/`, checked by
  the root-level `tests/test_c_snapshot.py`.

Two root-level tests are structural and permanent. `tests/test_import_layering.py` holds the
import-layer table, the two sanctioned exceptions and the acyclicity check; a new module needs an
entry in `IMPORT_LAYERS`. `tests/test_import_boundaries.py` pins the public names: that `sc.Expr` is
`scaly.ir.expr.Expr`, that both dialects verify through the same types, and that retired module
paths stay retired.

Some compiler paths are exercised only by workload-shaped fixtures, mainly the RK4 stage-transcription
Jacobian in `tests/integration/test_stage_transcription.py` and the chained-VMAP fixtures in
`tests/integration/test_vmap.py`. They build a mapped and a fully unrolled version of the same graph
and compare values, Jacobians and Hessians, so a coloring bug cannot hide behind a false structural
zero. Keep them working.

## Making a change

Read [the architecture](../how_it_works/architecture.md) first, then [The codebase](codebase.md).
[Where to add things](codebase.md#where-to-add-things) lists, for each kind of change, every
file it touches; adding a scalar operation touches seven.

Then read the surrounding code, follow what is already there, make a focused change, run the
narrowest relevant check, then widen. Match the style you find; see [Conventions](conventions.md).

Anything touching the IR, differentiation or code generation runs the full suite. Those paths break
subtly and are expensive to debug later.

Do not import `torch` or other libraries at run time. Scaly depends on NumPy and nothing else, and
a small local implementation is preferred, as with `scaly.utils.load_torch_state_dict`.

## Adding a solver backend

A backend is a separate distribution under `plugins/`, discovered by entry point, providing
packaging metadata and one code generation hook. The contract is in
[Solver plugins](solver_plugins.md).

## Documentation

The site is built with [Zensical](https://zensical.org):

```bash
uv run --only-group docs zensical serve    # live preview
uv run --only-group docs zensical build    # into site/
```

Everything under `docs/` is published. Zensical has no exclusion mechanism, so anything that should
stay unpublished lives in `internal/` at the repository root: the actionable list and the frozen
design notes.

The API reference is generated from docstrings, so a new public name needs one. Use Google style
and say what the thing is for instead of restating its signature.
