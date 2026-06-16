# Native solver toolchain exploration

This note records the June 2026 `toolchain` branch experiment around avoiding source builds, using
conda-forge native prefixes, and repairing conda-vendored solver wheels. It is intentionally historical:
the project direction after the experiment is to keep Alloy focused on a small, first-party PIQP/IPOPT
source build rather than becoming a broad binary aggregation layer.

## Goals tested

- Keep Python dependency management on `uv` while using conda/micromamba only as a native binary prefix.
- Let editable installs and core tests run without building PIQP/IPOPT.
- Point solver tests at `ALLOY_SOLVER_PREFIX=/path/to/prefix` without activating conda.
- Try building wheels by copying PIQP/IPOPT from a conda prefix and repairing transitive dependencies with
  `delocate`/`auditwheel`.
- Compare the approach with CasADi's vendored solver wheel process.

## Conda-prefix runtime test

A local prefix was created with:

```bash
mamba create -y -p "$TMPDIR/alloy-native-conda-test" -c conda-forge \
  ipopt=3.14.19 piqp=0.6.2 pkg-config
```

Observations on macOS arm64:

- `libipopt.dylib` and `libpiqpc.dylib` are present under `lib/`.
- IPOPT headers are under `include/coin-or/`.
- PIQP's C headers are under `include/piqp_c/`, not `include/piqp/`.
- With discovery adjusted for `piqp_c`, `ALLOY_SOLVER_PREFIX=$PREFIX` was enough for both direct ctypes
  solver calls and nested solver JIT tests when the existing `src/alloy/lib` and `src/alloy/include`
  directories were temporarily moved out of the checkout.

This validated that conda-forge can be a useful local developer prefix, but it also showed that it adds a
second binary layout to support.

## Conda-prefix wheel vendoring test

A wheel was built by copying from the conda prefix into `src/alloy/lib` and `src/alloy/include`. The
unrepaired wheel contained only:

```text
alloy/lib/libipopt.dylib
alloy/lib/libpiqpc.dylib
alloy/include/coin-or/...
alloy/include/piqp_c/...
```

`delocate` does scan ordinary `.dylib` files inside a wheel; a Python extension module is not required.
However, the copied conda libraries depended on sibling conda libraries through `@rpath`, so delocate could
not resolve the transitive dependencies unless the top-level copied solver libraries carried a temporary
build-time rpath back to `<prefix>/lib`.

After adding that temporary rpath and ad-hoc re-signing the modified top-level dylibs, delocate copied a
large transitive closure into `alloy/.dylibs`, including MUMPS, OpenBLAS, libgfortran, libomp, BLASFEO,
libc++, compression libraries, and graph-partitioning libraries. The repaired wheel was roughly 11 MB.

A remaining macOS issue was OpenBLAS aliasing in the conda prefix:

```text
libblas.3.dylib   -> libopenblas.0.dylib
liblapack.3.dylib -> libopenblas.0.dylib
```

Plain upstream `delocate` copied `libopenblas.0.dylib` and rewrote some references, but left unresolved
`@rpath/liblapack.3.dylib` references in `libipopt.dylib` and `libdmumps_seq.dylib`. Manually patching
those references to the copied `libopenblas.0.dylib`, re-signing, and repacking produced a self-contained
wheel that passed:

- direct PIQP ctypes solve;
- direct IPOPT ctypes solve;
- nested PIQP JIT solve;
- nested IPOPT JIT solve;
- with the original conda prefix moved out of the way.

This means conda-prefix wheel vendoring can be made to work, but not as a simple, robust default path.

## CasADi comparison

CasADi's macOS wheels take a different route:

- They build a self-contained install tree first, with libraries laid out directly in the `casadi/` package
  directory and `@loader_path` install rpaths.
- Their global binary CI enables source-built IPOPT, MUMPS, and many other solvers.
- On macOS they disable the OpenBLAS source build (`WITH_BUILD_LAPACK=OFF`), so IPOPT/MUMPS link against
  Apple's Accelerate framework instead of OpenBLAS.
- They run a custom `jgillis/universal_grafter` action on the installed package tree before making the final
  wheel. That action creates a dummy wheel, repairs it with forked `delocate`/`auditwheel` tooling, unpacks
  the repaired tree back into place, then fixes install names and code signs/notarizes on macOS.

A local inspection of the PyPI `casadi==3.7.2` macOS arm64 wheel confirmed:

```text
casadi/libipopt.3.dylib -> /System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate
casadi/libcoinmumps.3.dylib -> /System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate
```

No OpenBLAS dylib is bundled in that wheel.

## Decision after the experiment

Alloy should not become an aggregate binary distribution for many solvers. The long-term direction is to
write optimization loops directly in Alloy, while PIQP/IPOPT provide a small out-of-the-box path for optimal
control users. For that reason:

- The primary vendored solver path should remain Alloy's own source build of PIQP and IPOPT.
- macOS solver builds should continue to link IPOPT/MUMPS against Accelerate for the smallest and most
  native wheel dependency set.
- Linux can keep the explicit OpenBLAS/source dependency story where needed.
- Conda-prefix discovery and conda-prefix wheel vendoring are useful research results, but should not be the
  main runtime/build interface unless we revisit binary distribution strategy later.
- The central `alloy.toolchain` module and environment-variable registry are still useful and independent of
  the vendoring strategy; they should remain the single place for JIT compiler/cache settings and vendored
  solver path diagnostics.
- CI should make the source-built vendored solvers reproducible and cached, then run core checks and solver
  checks as separate jobs for clearer failure granularity.
