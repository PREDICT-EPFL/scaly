# Contributing

## Setup

```bash
git clone https://github.com/PREDICT-EPFL/alloy.git
cd alloy
uv sync
```

Use `uv run` for everything — `uv run pytest`, `uv run benchmarks/run.py`. `uv run` takes a script
path directly, so the `python` in `uv run python script.py` is redundant. Do not activate the
virtual environment by hand.

The vendored solvers (PIQP, IPOPT) build on the first sync and take 5 to 8 minutes cold. Without a
native toolchain the sync skips them and the solver tests skip with them; everything else works.
See [Installation](../guide/installation.md).

## The checks

```bash
uv run pytest -n=auto     # the suite, in parallel
uv run ruff check         # lint
uv run ruff format        # format
uv run ty check --error-on-warning  # types, including expected-error assertions
```

Run all four before you consider a change done. `pytest` collects `tests/`, `plugins/`, and
`typing_playground/`. Strict type checking covers the assertions in `tests/typing/` and the
playground: an expected error that disappears leaves an unused ignore, which must fail the check.

The full-collection node-ID baseline is checked by the root `conftest.py`. After adding, removing, or
renaming a test, regenerate it safely from the complete collection:

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

The first collection can exit nonzero because the existing baseline is stale or missing. The `grep` keeps only
pytest node IDs, including parameter IDs with spaces, and `sort` makes the file deterministic. The final
collection must pass.

A test needing a built solver is marked, not skipped by hand:

```python
@pytest.mark.solver("piqp")
def test_something(): ...
```

The root `conftest.py` skips those when the library is absent, and CI splits the suite on
`-m solver` against `-m "not solver"`. Never hand-roll a "is the solver loadable" skip condition.

One thing to know before you read a red run as a real failure: an xdist worker occasionally dies
inside the isolated library load in the vendored-solver plugin tests. It reproduces on unmodified
checkouts, so a lone worker crash there is probably not yours. Rerun before believing it.

## Where things live

`tests/` mirrors `src/alloy/` directory for directory, so a change to `src/alloy/passes/lowering.py`
has its tests in `tests/passes/test_lowering.py`. Beyond the mirror there are two extra
directories: `tests/integration/` for workload-shaped end-to-end checks, and `tests/benchmarks/`
for the benchmark *harness* — the benchmark *problems* keep their own gates. See
[Conventions](conventions.md#tests-against-benchmarks) for which side a check belongs on.

Two tests are structural rather than functional, and both are meant to be permanent:

- `tests/test_import_layering.py` holds the import-layer table, the two sanctioned exceptions and the
  acyclicity check. A new module needs an entry in `IMPORT_LAYERS`.
- `tests/test_import_boundaries.py` pins the public surface — that `al.Expr` really is
  `alloy.ir.expr.Expr`, that both dialects verify through the same types, and that retired module
  paths stay retired.

Some compiler paths are exercised only by workload-shaped fixtures — the RK4 stage-transcription
Jacobian in `tests/integration/test_stage_transcription.py` and the chained-VMAP fixtures in
`tests/integration/test_vmap.py` are the main ones. They build both a mapped and a fully unrolled version of
the same graph and hold the values, Jacobians and Hessians against each other, so a coloring bug
cannot hide behind a false structural zero. Keep them working.

## Making a change

Read [the architecture](../how_it_works/architecture.md) first if you have not. In particular
[Where to add things](../how_it_works/architecture.md#where-to-add-things) lists, for each kind of
change, every file it touches — adding a scalar operation touches six.

Then the ordinary discipline: read the surrounding code, follow what is already there, make a
focused change, run the narrowest relevant check, then widen. Match the style you find; see
[Conventions](conventions.md).

Two things specific to this repository:

**Anything touching the IR, differentiation or code generation runs the full suite.** Those paths
are cheap to break subtly and expensive to debug later.

**Do not add import to `torch` or other libraries.** Alloy depends on NumPy and nothing else at
run time, and that is a feature worth defending. Prioritize adding a small implementation if 
possible, like we did with `alloy.utils.load_torch_state_dict`.

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

`docs/` is what gets published — all of it. Zensical has no exclusion mechanism, so anything that
should stay unpublished lives in `internal/` at the repository root instead: the library roadmap
and the frozen design notes, kept for the record but off the site.

The API reference is generated from docstrings, so a new public name needs one. Google style, and
say what the thing is for rather than restating its signature.
