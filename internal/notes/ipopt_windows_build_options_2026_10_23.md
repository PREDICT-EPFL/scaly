# Windows Ipopt build options

Investigation for [#196](https://github.com/PREDICT-EPFL/scaly/issues/196), dated 2026-10-23.
Ted chose native Windows through MSYS2 UCRT64 as the first source-build route on 2026-10-23.

## Chosen route

Build the existing Ipopt 3.14.19, sequential MUMPS 5.8.2 (the Multifrontal Massively Parallel
Sparse Solver), METIS 5.2.1, GKlib and OpenBLAS 0.3.28 sources with MinGW-w64 GCC and GFortran
under MSYS2 UCRT64, which targets the Universal C Runtime. MSYS2 is a build dependency only:
users receive a wheel with the solver and its runtime dynamic-link libraries, or DLLs. Generated
callers keep using Zig. A probe showed a Zig-compiled caller solving through a MinGW-built Ipopt
over its C application binary interface, or ABI.

The toolchain is not pinned, as on macOS: CI installs MSYS2's current GFortran, and the solver
cache key names the compiler version, so a toolchain change triggers a rebuild.

## What CasADi does

CasADi 3.8.0 cross-builds its Windows wheels on Ubuntu in an MXE (MinGW cross environment) image,
`x86_64-w64-mingw32.shared.posix` with GCC/GFortran 11.2.0, and tests them on Windows. Its CMake
build drives the Ipopt and MUMPS Autotools builds and configures MUMPS with `--without-openmp`.
It builds from mutable fork branches (`jgillis/Ipopt-1`, `jgillis/ThirdParty-Mumps`,
`jgillis/ThirdParty-Metis` with METIS 4.0.3) and OpenBLAS v0.3.21, so copying its references
would not give a reproducible build. CasADi ships Ipopt, MUMPS, METIS and OpenBLAS as separate
DLLs; Scaly links the numerical stack statically into one Ipopt DLL and bundles only the
compiler runtimes.

## Source adjustments

- GKlib builds with C11, because GCC 16 defaults to C23, where its `bool`, `true` and `false`
  definitions are invalid.
- METIS builds only its library. Its command-line programs use Unix resource APIs that MinGW
  lacks.
- MUMPS builds without OpenMP. OpenBLAS keeps `DYNAMIC_ARCH=1` with the `TARGET=PRESCOTT`
  baseline.

None of these changes the numerical algorithms.

## Private DLL names

Loading CasADi after a Scaly Ipopt solve first failed with `ImportError: DLL load failed while
importing _casadi: The specified procedure could not be found.` Windows reuses an already-loaded
DLL with the same base name, and both packages bundle `libgfortran-5.dll` and its siblings. The
hook therefore names the solver `libscaly_ipopt-3.dll`, copies the five GCC runtime DLLs
(libgcc_s_seh, libstdc++, libgfortran, libquadmath and libwinpthread) from the compiler's
directory with a `scaly_ipopt_` prefix, and rewrites the imports with the build-only
`delvewheel` tool. Renaming the solver makes libtool's `libipopt.dll.a` stale, so `dlltool`
generates `ipopt.lib` from the DLL's exports; Zig's `-lipopt` finds that name. The plugin
declares `scaly_ipopt-3` as its Windows library stem, so the core finds the renamed DLL.

Before the rebase onto the new CI layout, the wheel built on native Windows and its 27 Ipopt
plugin tests passed, including CasADi loaded before and after Scaly in fresh processes. Fresh
installations passed `.github/scripts/windows_install.py` on Python 3.12, 3.13 and 3.14 with
only Windows directories on `PATH`, and hiding the bundled libgfortran made the load fail.

## Alternatives

| Route | Evidence and cost | Position |
| --- | --- | --- |
| Native MinGW/GFortran through MSYS2 UCRT64 | Native source build and installed-wheel checks pass. Fits the native Windows wheel job. | Chosen. |
| Linux MinGW/GFortran cross-build | CasADi ships working Windows wheels this way. Needs a target-aware hook, Windows wheel tagging and a hand-off to native Windows tests. | Alternative if we prefer cross-building. |
| LLVM Flang with MUMPS | Changes the Fortran compiler, runtime and build integration. No build evidence collected. | Defer unless GNU builds hit an obstacle. |
| Ipopt with Intel MKL PARDISO | Avoids compiling MUMPS but changes the linear solver and binary dependencies. | Defer: beyond packaging the current solver. |

[Ipopt's installation guide](https://coin-or.github.io/Ipopt/INSTALL.html) documents MinGW,
GFortran, MUMPS and MKL PARDISO. [MSYS2 recommends UCRT64](https://www.msys2.org/docs/environments/)
for new GNU builds, and notes that DLLs on different C runtimes interoperate when they exchange
no runtime-owned objects such as `FILE*`. Scaly's C interface exchanges only arrays, scalars and
function pointers.
