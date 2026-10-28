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
and build them from source on the first sync. CMake comes from PyPI as a build requirement, but a
C++ compiler and a Fortran compiler have to be installed system-wide:

```bash
# macOS
brew install gcc

# Debian / Ubuntu
sudo apt-get install gfortran build-essential
```

On Windows, IPOPT source builds use MSYS2 UCRT64, which targets the Universal C Runtime.
The GCC and winpthreads package versions are pinned in the plugin's
[`build_config.json`](https://github.com/PREDICT-EPFL/scaly/blob/main/plugins/scaly-ipopt/src/scaly_ipopt/build_config.json).
The [wheel workflow](https://github.com/PREDICT-EPFL/scaly/blob/main/.github/workflows/wheels.yml)
shows their installation commands. The environment also needs Git, make, patch, diffutils,
pkgconf, and the UCRT64 CMake and Ninja packages. Add its `ucrt64/bin` and `usr/bin`
directories to `PATH`. The hook runs source-build commands through its MSYS2 shell.

Then:

```bash
SCALY_BUILD_SOLVERS=required uv sync
```

Each build lands in a cache shared by every checkout on the
machine, `~/.cache/scaly/solvers` by default, under a hash of everything that decides its result:
the plugin's `hatch_build.py`, its pins, its license texts, the platform and the compilers. A new
worktree therefore copies the build its branch needs from the cache, and builds only when no
checkout has built that version yet. Builds unused for 30 days are deleted. A failed build keeps its
sources and build trees in the cache directory, at the path it prints.

Without `SCALY_BUILD_SOLVERS=required`, a missing native toolchain makes `uv sync` skip the solver
libraries instead of failing. The solver tests then skip, and everything else works. See
[Environment variables](../guide/env_vars.md).

To force a clean rebuild, delete the plugin's entry in the cache (`piqp-*` or `ipopt-*`) and the
checkout's `plugins/scaly-*/.build_key`, then run `uv sync --reinstall-package scaly-piqp` or
`scaly-ipopt`.

In the two vendoring plugins (`scaly-piqp`, `scaly-ipopt`), a new vendored dependency needs an entry
in `src/scaly_*/build_config.json`, which pins its version, and a row in the
`_write_third_party_notices` call of that plugin's `hatch_build.py`, which copies its license texts
into the wheel. The `test_*_notices.py` test in each of those plugins fails when a pinned dependency
has no license directory.

### Local Ipopt archives

The Ipopt producer builds a native archive for the current host using the same source recipe as
the build hook. It needs the native toolchain described above. Run it from the repository root:

```bash
uv run plugins/scaly-ipopt/hatch_build.py bundle --work-dir build/ipopt-native --output build/ipopt-native.tar.gz
```

The command writes the archive, `ipopt-native.tar.gz.manifest.json` and
`ipopt-native.tar.gz.sha256`. It retains source checkouts, component build trees, `build.log`
and `commands.jsonl` under the work directory, including after failure. Rerunning the command
reuses those sources and incremental native builds, then recreates the payload. You can inspect
or rebuild a component there with its native build tools. The manifest records tracked source
patches, downloaded MUMPS source hashes, compiler identities and generated build settings.
Use a new work directory after changing the recipe, target or compiler environment.

A local Linux archive records its host's glibc floor and uses a `linux_*` tag. Producing a
manylinux archive requires a suitable compatibility environment and an explicit
`--platform-tag`, for example `manylinux_2_27_x86_64`. The command runs `auditwheel repair`
before archiving and includes any repaired `scaly_ipopt.libs` sibling. macOS archives record
the highest minimum OS version in their native libraries, with a minimum of macOS 15.

Select an archive explicitly for an editable install or wheel build. The path should be absolute.
In Bash:

```bash
export SCALY_IPOPT_BUNDLE="$PWD/build/ipopt-native.tar.gz"
uv sync --reinstall-package scaly-ipopt
uv build plugins/scaly-ipopt --wheel
```

In PowerShell:

```powershell
$env:SCALY_IPOPT_BUNDLE = (Resolve-Path build/ipopt-native.tar.gz).Path
uv sync --reinstall-package scaly-ipopt
uv build plugins/scaly-ipopt --wheel
```

The hook checks the archive's target, manifest, complete file inventory and SHA-256 hashes before
installing. It rejects unsafe paths and links. It copies the finished native bytes without repair
and uses the payload's platform tag for the wheel. The archive contains the matching headers,
Windows `ipopt.lib`, native dependencies and licenses. It contains no Python wrappers or wheel
metadata. Archive selection needs no native compiler and does not change checked-in source pins.

Archive installs record their SHA-256 and input selection in `scaly_ipopt/native_bundle.json`.
The benchmark provenance includes that identity and a SHA-256 of the selected Ipopt library,
alongside the existing native source configuration. Reading provenance downloads no native input.

After replacing an archive at the same path, rerun
`uv sync --reinstall-package scaly-ipopt`. The hook keys its cache and checkout stamp by the full
archive SHA-256, so it cannot reuse the previous archive's cache entry. `uv` does not track the
contents of a file named by an environment variable, which is why the reinstall flag is required.

To return to the default source build, unset `SCALY_IPOPT_BUNDLE` with
`unset SCALY_IPOPT_BUNDLE` in Bash or `Remove-Item Env:SCALY_IPOPT_BUNDLE` in PowerShell, then
run `uv sync --reinstall-package scaly-ipopt`. The hook replaces the archive payload with the
source build selected by its usual cache key. `SCALY_BUILD_SOLVERS=skip` skips archive installation
as well as source builds. An sdist build neither reads the selected archive nor builds a solver.

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
