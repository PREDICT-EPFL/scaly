# Windows Ipopt build options

Investigation for [#196](https://github.com/PREDICT-EPFL/scaly/issues/196), dated 2026-10-23.
Ted chose native Windows through MSYS2 UCRT64 as the first source-build route on 2026-10-23.
Keep the current solver sources and use MinGW/GFortran on the builder, with Zig for generated
callers. A complete source build and installed-wheel checks must establish the final recipe.

## Chosen first route

Use MinGW-w64 GCC and GFortran for the native dependencies. Keep Scaly's existing Ipopt 3.14.19,
sequential MUMPS 5.8.2, METIS 5.2.1, GKlib and OpenBLAS 0.3.28 sources. MUMPS is the
Multifrontal Massively Parallel Sparse Solver. Keep Zig for the generated
C callers. The native probe below establishes that a Zig caller can use a MinGW-built Ipopt
library through its C application binary interface, or ABI.

Prefer building on native Windows with an MSYS2 UCRT64 build environment. UCRT64 uses the
Universal C Runtime. MSYS2 is a build dependency only. End users receive a wheel containing the
solver and its runtime dynamic-link libraries, or DLLs.

This keeps Windows in the existing native wheel matrix and lets us validate a source build on
the target machine. CasADi's Linux cross-build is a credible alternative if native shell and
path handling becomes the larger maintenance burden. Neither source-build route has been
tested here yet. The recommendation is based on the existing packaging structure and the
binary compatibility evidence, not a completed Scaly Windows build.

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

CasADi packages separate shared libraries for Ipopt, MUMPS, METIS and OpenBLAS. Scaly currently
links the numerical dependencies statically into its shared Ipopt library on Linux. We can
try to retain that arrangement on Windows while bundling the compiler runtimes dynamically.
The final runtime list must come from the built binaries.

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

This proves the tested C ABI and DLL-loading path. It does not prove a Scaly source build,
a Scaly wheel installation, the complete Ipopt numerical references, or reproducibility.

## Alternatives

| Route | Evidence and cost | Position |
| --- | --- | --- |
| Native MinGW/GFortran through MSYS2 UCRT64 | Ipopt documents the GNU/MSYS2 route. Fits our native Windows wheel job. Requires explicit shell invocation, path conversion and a pinned build environment. | Recommended first source-build experiment. |
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

## What the agreed recipe must establish

The source build should retain the existing numerical versions and CPU baseline, produce a
matching DLL and import library, bundle every non-system dependency found recursively, and
include the existing third-party notices plus any new runtime notices. CasADi's versioned
`libipopt-3.dll` name is not recognized by Scaly's current Windows library discovery. The
DLL name, its import-library reference and discovery must agree. Renaming a DLL alone is
insufficient.

Build-tool and source identities must participate in the build cache key. Installing the
latest packages from MSYS2's rolling repositories alone does not pin the build environment.
The implementation needs an explicit versioned, verified toolchain source.

Acceptance remains #196's full installed-wheel numerical, status and statistics checks on
native Windows, without build paths, followed by Windows continuous-integration and release
matrix coverage. The source-build experiment comes before committing to the final recipe.

## Repository and tracking context

Read `plugins/scaly-ipopt/hatch_build.py`, its `build_config.json` and package metadata,
`src/scaly/solvers/paths.py`, the PIQP Windows hook, `.github/workflows/wheels.yml`,
`.github/scripts/windows_install.py`, the release workflow, `internal/notes/vendored_solvers.md`
and the Windows toolchain decisions. The issue has no recorded blockers or open linked pull
requests. #156, which supplied the existing Windows installation and CI path, is closed.

The active branch is `t3/ipopt-windows-build-196`. The investigation is recorded on #196.
Project status could not be inspected or changed because the GitHub token lacks `read:project`.
