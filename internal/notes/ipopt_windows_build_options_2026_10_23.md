# Windows Ipopt build options

Investigation for [#196](https://github.com/PREDICT-EPFL/scaly/issues/196), dated 2026-10-23.
Ted chose native Windows through MSYS2 UCRT64 as the first source-build route on 2026-10-23.
Keep the current solver sources and use MinGW/GFortran on the builder, with Zig for generated
callers. The native source build and installed-wheel results below establish the recipe.

## Chosen first route

Use MinGW-w64 GCC and GFortran for the native dependencies. Keep Scaly's existing Ipopt 3.14.19,
sequential MUMPS 5.8.2, METIS 5.2.1, GKlib and OpenBLAS 0.3.28 sources. MUMPS is the
Multifrontal Massively Parallel Sparse Solver. Keep Zig for the generated
C callers. The native probes below establish that a Zig caller can use a MinGW-built Ipopt
library through its C application binary interface, or ABI.

Prefer building on native Windows with an MSYS2 UCRT64 build environment. UCRT64 uses the
Universal C Runtime. MSYS2 is a build dependency only. End users receive a wheel containing the
solver and its runtime dynamic-link libraries, or DLLs.

This keeps Windows in the existing native wheel matrix. The numerical stack builds on native
Windows, and the installed wheel passes direct and nested solves on Python 3.12, 3.13 and 3.14.
The Linux cross-build alternative was not needed.

## What CasADi does

Scaly's lockfile selects CasADi 3.8.0. Its
[release workflow](https://github.com/casadi/casadi/blob/3.8.0/.github/workflows/binaries.yml)
builds the Windows core and Python wheel on Ubuntu through
`ghcr.io/jgillis/windows-shared-x64-posix:production`. The `core-dockcross` and
`python-dockcross` jobs use this image. Separate `test-python` jobs run the wheel on Windows.

The [image recipe](https://github.com/casadi/casadi/blob/3.8.0/misc/Dockerfile.windows) extends
`dockcross/windows-shared-x64-posix:20230108-c9273a2`. It uses MXE, a MinGW cross-build
environment, and the target `x86_64-w64-mingw32.shared.posix`. The installed `libipopt.la`
records GCC/GFortran 11.2.0 paths. Here, `posix` names the toolchain's threading model. The
resulting libraries are native Windows binaries and do not import an MSYS runtime.

The [CasADi CMake build](https://github.com/casadi/casadi/blob/3.8.0/CMakeLists.txt) orchestrates
Autotools builds for Ipopt and MUMPS. It forwards `--host`, passes
`lt_cv_deplibs_check_method=pass_all` for MinGW, and configures MUMPS with `--without-openmp`.
Its source defaults differ from ours:

- Ipopt comes from `jgillis/Ipopt-1`, branch `3.14.19.mod`.
- MUMPS comes from `jgillis/ThirdParty-Mumps`, branch `gate-unconditional-use-omp-lib`.
  Its download script selects MUMPS 5.8.2. The extra patch makes `USE OMP_LIB` conditional.
- METIS comes from `jgillis/ThirdParty-Metis`, branch `bugfix2`.
  Its [download script](https://github.com/jgillis/ThirdParty-Metis/blob/bugfix2/get.Metis)
  selects METIS 4.0.3.
- The release's OpenBLAS source default is v0.3.21.

CasADi's Ipopt fork includes an extra array element in the triplet-index buffers and changes
the dynamic-loader flags. Those changes need individual justification if a build exposes a
need for them. Copying the entire fork or reverting our METIS version is not part of the
recommendation. The current fork branches and the container's `production` tag are mutable,
so copying those references would not establish a reproducible Scaly build.

CasADi packages separate shared libraries for Ipopt, MUMPS, METIS and OpenBLAS. Scaly links the
numerical dependencies statically into its shared Ipopt library on Linux and Windows, while
bundling the compiler runtimes dynamically.

## Native Windows evidence

The locally installed CasADi 3.8.0 Windows wheel was inspected without importing CasADi.
Its Ipopt DLL has nine non-system DLLs in its transitive dependency chain, including itself:

```text
libipopt-3.dll
libcoinmumps-3.dll
libcoinmetis-2.dll
libcasadi-tp-openblas.dll
libstdc++-6.dll
libgfortran-5.dll
libgcc_s_seh-1.dll
libquadmath-0.dll
libwinpthread-1.dll
```

Together they occupy 32,164,171 bytes, about 30.7 MiB before wheel compression. All nine use
the Windows x86-64 binary format. Their remaining direct imports are `KERNEL32.dll`,
`ADVAPI32.dll` and `msvcrt.dll`, which Windows supplies. The inspection rejects unexamined
delay-load imports, and none occurred in this dependency chain.

A small C library compiled with the existing `zig-bin` 0.17.0 executable linked against
CasADi's `libipopt.dll.a`. It supplied exact objective, constraint and derivative oracles
for this problem:

```text
minimize x[0]^2 + x[1]^2
subject to x[0] + x[1] = 10
           -100 <= x[i] <= 100
start at [0, 0]
```

The library selected `linear_solver=mumps` and recorded the iteration count through the
intermediate C interface. A fresh Python 3.14.8 process loaded the probe by absolute path,
with an empty working directory, isolated Python imports and only
`C:\WINDOWS\System32;C:\WINDOWS` on `PATH`. No CasADi Python module was imported.

The solve returned Ipopt status 0, solution `[5.0, 5.0]`, objective 50.0 and 7 iterations.
`GetIpoptVersion` reported 3.14.19. Windows module-path queries confirmed that the probe and
all nine dependency DLLs loaded from the staged directory inside this worktree. Removing
the staged `libgfortran-5.dll` made a second fresh process fail to load the probe. Restoring
the DLL restored the staged files.

Local scratch files and evidence are under `build/ipopt-196/`:

```powershell
uv run --no-sync build/ipopt-196/inspect_casadi.py
uv run --no-sync build/ipopt-196/run_probe.py
```

`dll_evidence.json` records the import graph, byte counts and SHA-256 values.
`native_probe.json` records the solve and loaded paths. `negative_probe.txt` records the
missing-runtime failure. These scratch scripts read CasADi and Zig from the main checkout's
existing virtual environment at `C:/Users/tudor/Desktop/scaly/.venv`. They do not modify it.

This first probe established the C ABI and DLL-loading path before choosing the source build.

## Native source build and wheel evidence

The native stack built under MSYS2 UCRT64 with GCC/GFortran 16.2.0-6 and the
winpthreads packages 14.0.0.r495.g63ebe6cdb-2. These package versions are pinned in
`plugins/scaly-ipopt/src/scaly_ipopt/build_config.json`. The wheel workflow installs those
signed package archives. The build cache also includes the full `pacman -Q` manifest and
the versions of the DLL editing tools. Source versions remain unchanged.

The package transaction includes the matching `cc-libs` metapackage and `libatomic`.
Older MSYS2 snapshots carry `cc-libs` with exact dependencies on an earlier GCC runtime;
upgrading only the compiler and numerical runtimes fails dependency resolution. This was
reproduced in an isolated package database, where the complete transaction passed.

GKlib needs an explicit C11 setting because GCC 16 defaults to C23, where its definitions of
`bool`, `true` and `false` are invalid. METIS builds only its library on Windows. Its optional
command-line programs use Unix resource APIs that MinGW does not supply. Neither adjustment
changes the numerical algorithms. OpenBLAS retains `DYNAMIC_ARCH=1` with the common-code
`TARGET=PRESCOTT` baseline. MUMPS uses `--without-openmp`.

The first Scaly solve succeeded, but loading CasADi afterward failed with
`ImportError: DLL load failed while importing _casadi: The specified procedure could not be found.`
The two distributions' GCC DLLs had the same names. The hook now gives Scaly's Ipopt a private
name, rewrites runtime imports with the build-only `delvewheel` tool, and gives the five runtime
DLLs a prefix containing a hash of the bundle. `dlltool` generates `ipopt.lib` from the actual
exports, pointing at the private solver DLL. Zig's `-lipopt` search finds `ipopt.lib`, while it
does not find GNU's original `libipopt.dll.a` filename.

The resulting six DLLs are the private Ipopt library plus private copies of libgcc, libstdc++,
libgfortran, libquadmath and libwinpthread. MUMPS, METIS, GKlib and OpenBLAS are static within
Ipopt. Runtime collection reads the compiler's directory directly because Python's
`shutil.which()` does not find DLL files. Missing transitive dependencies fail the build.
The existing notices are retained, with a winpthreads notice copied from its pinned package.

`SCALY_BUILD_SOLVERS=required uv build plugins/scaly-ipopt --wheel` produced
`scaly_ipopt-0.1.0a3-py3-none-win_amd64.whl`, 10,621,256 bytes. The local artifact's SHA-256 is
`feb7dbb04f37470ed47621dacb666f5b11edee2c38ddbccd7e240a994b4ef0c3`.
The compiler executables and `SCALY_SOLVER_CACHE` pointed into this worktree's `build/` tree.

All 25 plugin tests passed, including numerical references, status/statistics, warm starts,
nested calls and both CasADi load orders in fresh processes. Fresh installations of the core,
Ipopt, PIQP and SQP wheels passed `.github/scripts/windows_install.py` on Python 3.12.11,
3.13.1 and 3.14.8 with Zig 0.17.0. The child processes had only their environment's `Scripts/`,
the Windows system directory and Windows itself on `PATH`. Direct and nested Ipopt solves
matched the analytic disk projection and reported successful status and nonzero iterations.
Module-path checks located every solver DLL under the installed wheel directories.

A separate extraction of the wheel loaded all six Ipopt DLLs with only Windows directories
on `PATH`. `GetIpoptVersion` returned 3.14.19. Removing its private libgfortran made a fresh
process fail to load the solver. The missing runtime could not be supplied by another toolchain
or CasADi. The DLL was restored after the check.

Editable installs skip an unsuitable Windows toolchain in automatic mode and fail in required
mode. Regression tests cover both standalone Fortran compilers and mismatched MSYS2 package
versions. Runtime collection prefers the pinned toolchain even if another installer put a
DLL with the same name in `System32`.

Enabling all solver tests also exposed Unix assumptions in the benchmark helpers. The Windows
path now reads DLL import symbols, exports the benchmark shim, resolves the actual Ipopt
function address with the Windows module API, and records CasADi's versioned DLL filename.
The related benchmark regression checks passed, 53 tests with 4 platform skips. The new import
inspection check fails when its symbol extraction is disabled. The runtime graph check fails
when transitive traversal is disabled.

## Alternatives

| Route | Evidence and cost | Position |
| --- | --- | --- |
| Native MinGW/GFortran through MSYS2 UCRT64 | Native source build and installed-wheel checks pass. Fits our native Windows wheel job. | Chosen. |
| Linux MinGW/GFortran cross-build | CasADi ships working Windows wheels this way. Reuses Unix build tools. Requires a target-aware hook, Windows wheel tagging and a build-artifact handoff to native Windows tests. | Strong alternative if we prefer a cross-build pipeline. |
| LLVM Flang with MUMPS | Keeps MUMPS but changes the Fortran compiler, runtime and build-system integration. No native source-build evidence collected here. | Defer unless GNU builds reveal a concrete obstacle. |
| Ipopt with Intel MKL PARDISO | Ipopt supports this solver and MKL supplies the numerical libraries, which can avoid compiling MUMPS. Changes the linear solver and binary dependencies. | Defer because it expands the scope beyond Windows packaging of the current solver. |

[Ipopt's installation guide](https://coin-or.github.io/Ipopt/INSTALL.html) documents MinGW,
GFortran, MUMPS and MKL PARDISO. [MSYS2 recommends UCRT64](https://www.msys2.org/docs/environments/)
for new GNU builds. Its documentation also explains that DLLs using different C runtimes can
interoperate when they do not exchange runtime-owned objects such as `FILE*`. Our numerical
C interface exchanges arrays, scalars and function pointers. All numerical dependencies in
our own source build should still use one consistent toolchain and C runtime.

[Flang's driver documentation](https://flang.llvm.org/docs/FlangDriver.html) describes its
separate runtime and mixed C++/Fortran linking requirements. This investigation does not
establish that Flang cannot build this stack.

## Acceptance and limits

The source build retains the numerical versions and CPU baseline, produces a matching DLL and
import library, and bundles the runtime dependency chain and notices. The plugin declares
`scaly_ipopt-3` as its Windows library stem, which lets existing core discovery find
`libscaly_ipopt-3.dll`. Renaming the DLL alone would leave callers importing the old name.

Compiler and runtime package versions are pinned. Other build-tool versions are recorded in
the cache identity. This establishes a repeatable source-build recipe and runtime bundle,
not byte-for-byte reproducibility across different Windows images and build directories.
MSYS2 can prune superseded package archives from its repository, so the package pins need
maintenance when their downloads stop being available. The hook and workflow read the same
package list from `build_config.json`.

`wt hook pre-merge --yes` passed on native Windows: 2,709 tests passed and 102 skipped,
with both solvers required. Type checking, formatting, lint, benchmark smoke, the documentation
build and the private-name check passed. The smoke includes the Ipopt oracle comparisons and
standalone C++ solver consumer. The newly enabled race-cars closed-loop checks also passed
for both compiled and interpreted CasADi oracles alongside Scaly.

The first Windows release needs new versions of core, PIQP and Ipopt together. Their published
`0.1.0a2` distributions predate Windows support. All three are bumped to `0.1.0a3`, with matching
minimum dependencies. SQP's published source is unchanged and its existing pure-Python wheel
works with the new versions.

The pull request's continuous-integration run remains the final platform check. Build and
release workflows include the Windows Ipopt wheel, require the backend in Windows tests and
run the isolated installation assertions on all three Python versions.

## Repository and tracking context

Read `plugins/scaly-ipopt/hatch_build.py`, its `build_config.json` and package metadata,
`src/scaly/solvers/paths.py`, the PIQP Windows hook, `.github/workflows/wheels.yml`,
`.github/scripts/windows_install.py`, the release workflow, `internal/notes/vendored_solvers.md`
and the Windows toolchain decisions. The issue has no recorded blockers or open linked pull
requests. #156, which supplied the existing Windows installation and CI path, is closed.

The active branch is `t3/ipopt-windows-build-196`. The investigation is recorded on #196.
Project status could not be inspected or changed because the GitHub token lacks `read:project`.
