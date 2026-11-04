# Contributing

To contribute a change, set up a checkout and run the checks below. You do not need to build the
optional solvers to work on expressions, derivatives, or most compiler tests.

Use [Tracking work](issue_tracking.md) to find issues, interpret planning metadata, and check
dependencies and overlapping work before starting a change.

## Setup

```bash
git clone https://github.com/PREDICT-EPFL/scaly.git
cd scaly
uv sync
```

Run commands through `uv run` so they use the project's pinned environment. For example, use
`uv run pytest` for tests and `uv run benchmarks/run.py` for the benchmark script.

Scaly itself is pure Python, but the `scaly-piqp` and `scaly-ipopt` plugins vendor their solvers
and look first in the local solver cache, then in the newest PyPI release. A compatible release
wheel supplies its libraries, headers and licenses when its recorded native-input hash matches
the checkout. The hook verifies the download against PyPI's SHA-256 before copying any files.
An offline machine, a missing platform wheel or a different hash falls back to a source build.
Only source builds need a system C++ compiler and, for IPOPT, a Fortran compiler. CMake comes
from PyPI as a build requirement:

```bash
# macOS
brew install gcc

# Debian / Ubuntu
sudo apt-get install gfortran build-essential
```

On Windows, IPOPT source builds use MSYS2 UCRT64, which targets the Universal C Runtime. Install
its GFortran, CMake and Ninja packages with Git, make, patch, diffutils and pkgconf, as the
[CI build workflow](https://github.com/PREDICT-EPFL/scaly/blob/main/.github/workflows/ci-build.yml)
does, and add its `ucrt64/bin` and `usr/bin` directories to `PATH`. The hook runs source-build
commands through its MSYS2 shell.

Then:

```bash
SCALY_BUILD_SOLVERS=required uv sync
```

A source build takes 5 to 8 minutes on Linux and macOS, and about an hour on Windows, most of it
for OpenBLAS. Builds are shared by checkouts through `~/.cache/scaly/solvers`. The native-input
hash covers the plugin's `native_build.py` recipe, `build_config.json` and license texts.
The hook and shared lookup helper stay outside it, so lookup fixes reuse existing native libraries. Released
libraries are cached by this hash and platform alone. Source builds also include compiler
identity and build environment in their key. Without an explicit compiler selection, lookup can
reuse the recorded compiler build even when the machine's default compiler differs. Explicit
`CC`, `CXX` and `FC` selections must match the cached build. Builds unused for 30 days are deleted. A failed
build keeps its sources and build trees in the cache directory, at the path it prints.

`SCALY_BUILD_SOLVERS=required` makes a missing toolchain fatal when neither cache nor PyPI can
supply the libraries. The default `auto` lets an editable install skip them instead, and the
solver tests then skip. `SCALY_BUILD_SOLVERS=skip` skips native preparation entirely.

To force compilation even when cached or released libraries match, use
`SCALY_BUILD_SOLVERS=source uv sync --reinstall-package scaly-piqp --reinstall-package scaly-ipopt`.
The Monday nightly uses the same forcing value. Releases use the ordinary lookup order, so their
native libraries remain byte-identical until the recipe changes.

In the two vendoring plugins (`scaly-piqp`, `scaly-ipopt`), a new vendored dependency needs an entry
in `src/scaly_*/build_config.json`, which pins its version, and a row in the
`_write_third_party_notices` call of that plugin's `native_build.py`, which copies its license texts
into the wheel. The `test_*_notices.py` test in each of those plugins fails when a pinned dependency
has no license directory.

## Run the checks

```bash
uv run pytest -n=auto              # the suite, in parallel
uv run ruff check                  # lint
uv run ruff format --check         # formatting, as CI runs it; drop --check to fix
uv run ty check --error-on-warning # types, including expected-error assertions
```

Run all four before you consider a change done. `pytest` collects `tests/`, `plugins/`, and
`typing_playground/`. Strict type checking covers the assertions in `tests/typing/` and the
playground. An expected error that disappears leaves an unused ignore, which fails the check.

Type checking also covers the benchmarks. Their CasADi imports are treated as `Any`
because CasADi does not supply complete type information. This exception applies only
to benchmark files, so Scaly calls and the benchmark code's own annotations remain checked.

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

The root `conftest.py` skips those when the library is absent; CI installs both solver wheels and
runs everything. Never hand-roll a "is the solver loadable" skip condition.

## What CI runs

Pull requests and the merge queue run the same jobs of `ci.yml`, all on the Python version in
`.python-version`:

- `lint` runs Ruff, the format check, the node-ID baseline and ty three times: at the oldest
  supported Python, at 3.14 and for Windows. On a pull request it also rejects a title with a
  conventional-commit prefix, since the title becomes the commit title on main. Ruff forbids
  `sys.version_info`, because CI runs a single Python version and would never test the other branch.
- `docs` builds the site and fails on any warning, such as a broken link.
- `build / <platform>`, `test / <platform>` and `install / <platform>` run on linux-x86_64,
  linux-arm64, macos-arm64, macos-x86_64 and windows-x86_64. The build job makes the platform's
  wheels once. The test job runs the whole suite against them. The install job installs them into
  a clean environment, a slim container on Linux, and solves with every backend.
- `smoke / linux-x86_64` runs the benchmark and closed-loop smoke checks after `build`, in
  parallel with tests and install checks.
- `ci` passes only when every other job passed. It is the one required check, and its summary
  has a row per platform saying whether the native libraries were built or came from the cache,
  and the smoke result.

The native cache tracks each plugin's `native_build.py`, `build_config.json` and `licenses/`, so
edits to Python wrappers or lookup code reuse the libraries. Source builds also retain compiler
identities in their keys. Google Benchmark is built once per compiler and restored from the cache too. A push to main runs CI again unless it only touches
`internal/`, and stops after its first job when the merge queue already tested the same commit.

## Nightly and release checks

The nightly builds the same wheels as CI, then checks three independent groups:

- `toolchain / <image> <compiler>` runs the unit tests and benchmark smoke with each compiler
  and runner image.
- `python / <version>` runs only the unit tests on linux-x86_64 for every Python version in the
  package classifiers.
- `benchmarks / <platform>` runs the full benchmark smoke and every supported closed-loop pair
  once on each of the five platforms.

Nightly compile caches and Google Benchmark builds start cold. The baseline toolchain entries
save their compile caches for CI to restore through `SCALY_CACHE_DIR`. The nightly also saves
native libraries and Google Benchmark builds on main so pull requests can reuse them. On Mondays
the native builds also start without a cache and compile both solver plugins from source. A
manual nightly run accepts `rebuild-native` to exercise this source build on another day.
Scheduled runs skip an unchanged main except on Mondays.

A failing scheduled run opens or updates one issue with the labels `source:maintainer`, `kind:bug`
and `area:tooling`. It adds `area:benchmarks` while a benchmark entry fails and removes that label
when only other entries fail. The first passing run closes the issue.

A release reuses completed checks and their built files for the exact release commit on main.
It first looks for completed TestPyPI verification, then a completed full nightly. With neither
available, it builds and runs the full nightly suite. An unchanged nightly that skipped its tests does not
qualify, and missing or expired files force fresh validation. A fresh run on Monday also rebuilds
the native plugins from source.

A `pypi` release automatically uploads to TestPyPI, checks clean installations, then publishes the
same files to PyPI. It skips the TestPyPI checks when those files already passed them for this
commit. A `testpypi` release stops after those checks. Neither target pauses for manual approval,
and the run summary links to any earlier validation it reused.

## Where things live

`tests/` mirrors `src/scaly/` directory for directory, so a change to `src/scaly/passes/lowering/`
has its tests in `tests/passes/lowering/`. Outside the mirror:

- `tests/integration/` holds workload-shaped end-to-end checks.
- `tests/benchmarks/` tests the benchmark harness. The benchmark problems keep their own gates.
  [Conventions](conventions.md#tests-against-benchmarks) explains which side a check belongs on.
- `tests/typing/` holds the expected-error assertions that `ty check` covers.
- `tests/baseline/` holds `pytest_nodeids.txt` and the generated-C snapshots under `c/`, checked by
  the root-level `tests/test_c_snapshot.py`.

Two root-level tests are structural and permanent. `tests/test_import_layering.py` holds the
import-layer table, the two sanctioned exceptions and the acyclicity check. A new module needs an
entry in `IMPORT_LAYERS`. `tests/test_import_boundaries.py` pins the public names: that `sc.Expr` is
`scaly.ir.expr.Expr`, that both dialects verify through the same types, and that retired module
paths stay retired.

The integration tests include fourth-order Runge-Kutta stage derivatives in
`tests/integration/test_stage_transcription.py` and chained mapped functions in
`tests/integration/test_vmap.py`. They compare mapped and unrolled versions of the same calculation,
including values, Jacobians, and Hessians. These comparisons detect errors that would otherwise
appear as missing nonzero derivative entries.

## Making a change

Read [the architecture](../how_it_works/architecture.md) first, then [The codebase](codebase.md).
[Where to add things](codebase.md#where-to-add-things) lists, for each kind of change, every
file it touches. Adding a scalar operation touches seven.

Then read the surrounding code, follow what is already there, make a focused change, run the
narrowest relevant check, then widen. Match the style you find, as described in
[Conventions](conventions.md).

Anything touching the IR, differentiation or code generation runs the full suite. Those paths break
subtly and are expensive to debug later.

NumPy and SciPy are Scaly's only runtime dependencies. Do not import `torch`, `tinygrad`, or
another library at run time. Prefer a small local implementation, as with
`scaly.utils.load_torch_state_dict`.

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
stay unpublished lives in `internal/` at the repository root, including design notes and investigation records.
Actionable work lives in [GitHub Issues](https://github.com/PREDICT-EPFL/scaly/issues).

The API reference is generated from docstrings, so a new public name needs one. Use Google style
and say what the thing is for instead of restating its signature.
