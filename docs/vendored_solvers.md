# Vendored solver build — known issues

The `hatch_build.py` hook ships PIQP and IPOPT as shared libraries inside the wheel (`src/alloy/lib/`) plus C headers (`src/alloy/include/`). These notes track open issues we should fix before the wheels.yml workflow is exercised at scale. Historical notes from the conda-prefix/delocate experiment live in [`native_toolchain_exploration.md`](native_toolchain_exploration.md).

## 1. Static libgfortran linking — partially achieved

**Goal:** `libipopt.{dylib,so}` has no runtime dependency on libgfortran/libgcc/libstdc++, so external C++ consumers can link `-lipopt` without dragging a Fortran toolchain into their build.

**Current state on macOS (arm64, Apple Clang 21):** *not* met. `otool -L src/alloy/lib/libipopt.dylib` shows dynamic dependencies on `/opt/homebrew/opt/gcc/lib/gcc/current/libgfortran.5.dylib` and `libquadmath.0.dylib`.

**Why:** IPOPT's final libipopt link is driven by `clang++`, not `gfortran`. Apple Clang errors on `-static-libgfortran` (a gfortran driver flag), so we currently pass it through `FCFLAGS=` only. That covers Fortran-only objects but not the C++ link step that produces the dylib.

An earlier attempt put the static flags into `LDFLAGS=`. That broke the autoconf `configure` C-compiler conftest (Apple Clang refuses to even probe with the unknown flag), so the build never started. Hence the `FCFLAGS=` compromise in `_static_fortran_ldflags`.

**Likely fix:** locate `libgfortran.a` / `libquadmath.a` from the active gfortran installation and pass them as explicit static link inputs via `-Wl,-force_load <path>` (macOS) or `-Wl,-Bstatic -lgfortran -Wl,-Bdynamic` (Linux). The path resolution can use `gfortran -print-file-name=libgfortran.a` at hook time.

**Current state on Linux (GitHub `ubuntu-latest`):** source builds and solver tests pass, but the static dependency goal is also *not* met. CI `ldd src/alloy/lib/libipopt.so` still shows dynamic dependencies on `libgfortran.so.5`, `libstdc++.so.6`, and `libgcc_s.so.1`. `_static_fortran_ldflags` returns `-static-libgfortran -static-libgcc -static-libstdc++`, but the same `FCFLAGS`-only routing does not affect IPOPT's final C++ shared-library link.

## 2. METIS legacy-C compatibility

`ThirdParty-Metis` is pinned to `releases/2.0.1`. Despite the 2.x version number it still bundles upstream METIS 4.0.3 — a GPL-licensed K&R-era release whose `__GKfree` / implicit declarations modern Clang rejects by default. `_build_metis` patches around this by passing:

```
CFLAGS=-O2 -fPIC -Wno-implicit-function-declaration -Wno-implicit-int -Wno-int-conversion -Wno-error
```

If a future MUMPS pin requires METIS 5.x (Apache-2 licensed, modern C), we'll need to switch to a different ThirdParty branch or build METIS 5 directly. The current pin works on macOS and should work on the manylinux_2_28 image.

## 3. `install_name` rewriting on macOS

IPOPT's install dir contains `libipopt.3.dylib` (real file) and `libipopt.dylib` (symlink). `shutil.copy2` of the symlink resolves through it but preserves the original install_name (`/abs/path/to/.../libipopt.3.dylib`), which makes the lib non-relocatable.

Current fix: pick the versioned dylib directly, copy as `libipopt.dylib`, then run `install_name_tool -id @rpath/libipopt.dylib`. Same `-id` rewrite is applied to PIQP for consistency. Relevant code: `_build_ipopt_stack` near the end.

Linux keeps a SONAME-compatible copy next to the unversioned link target (for example `libipopt.so.3` next to `libipopt.so`) so JIT-built solver callers with `DT_NEEDED=libipopt.so.3` can resolve through their rpath. A future wheel repair pass may still prefer setting/changing SONAMEs explicitly with `patchelf`.

## 4. Wheel platform tag and editable build mode

**Goal:** each built wheel correctly declares its platform (and is manylinux- / delocate-compatible) so installers resolve it correctly and PyPI accepts upload, while local editable installs do not fail before Python-only development can start.

**Current state:** `hatch_build.py` marks wheel builds that include vendored solver libraries as impure:

```python
build_data["pure_python"] = False
build_data["infer_tag"] = True
```

ABI tag stays `none` because the vendored libs are loaded via `ctypes`, not linked as a CPython extension module — there's no Python ABI to bind to. Output becomes `alloy-0.1.0-py3-none-macosx_14_0_arm64.whl` / `alloy-0.1.0-py3-none-linux_x86_64.whl` per build host.

Editable installs use `ALLOY_BUILD_SOLVERS=auto` by default: if the native toolchain is present, `uv sync` builds the solver stack; if it is missing (for example no `gfortran`), the hook skips the missing solver libraries and solver tests are skipped. CI sets `ALLOY_BUILD_SOLVERS=required` so missing toolchains/build regressions remain fatal. `ALLOY_BUILD_SOLVERS=skip` is available for intentionally Python-only syncs.

**Still required before distribution:** a correct tag is necessary but not sufficient; PyPI rejects raw `linux_*` and the wheel may still pull in host-specific shared libs.

- **Linux:** run `auditwheel repair` on the wheel. It rewrites `linux_x86_64` → the lowest manylinux baseline that the binary actually satisfies (target: `manylinux_2_28_x86_64`) and bundles / patchelfs any non-allowlisted shared libs into the wheel. This still blocks on either fixing the Linux half of issue #1 (static libgfortran/libgcc/libstdc++) or deliberately letting auditwheel vendor those runtime libraries.
- **macOS:** run `delocate-wheel`. Equivalent operation: copies dylib dependencies into the wheel and rewrites install names against `@loader_path`. Currently would pull in Homebrew `libgfortran.5.dylib` and `libquadmath.0.dylib` — blocks on the macOS half of issue #1.
- **Matrix:** arm64 / x86_64 on each OS are separate wheels; build each on its native runner (or via `cibuildwheel` in the eventual `wheels.yml`) and upload the full set.

**Renaming is not a substitute.** The wheel's `*.dist-info/WHEEL` file records a `Tag:` line that installers cross-check against the filename. A wheel renamed from `py3-none-any.whl` to `py3-none-macosx_14_0_arm64.whl` still claims `any` internally and fails strict validation. Independently, PyPI refuses uploads with raw `linux_*` tags — only `manylinux_*` / `musllinux_*` are accepted, and those tags are contracts about glibc baseline and bundled deps, not free-form labels.

## 5. Things not yet exercised

- **CI cache key.** Keyed on OS, architecture, and `hashFiles('hatch_build.py')`. If we later split the hook into multiple files, update the key. Cold IPOPT build is ~5-8 min, so a stale cache hides a lot.
- **Static OpenBLAS install.** `_build_openblas` builds with `NO_SHARED=1 USE_OPENMP=0 DYNAMIC_ARCH=1`; pass the same flags to `make install` or OpenBLAS tries to install a shared `libopenblas*.so` that was never built.
- **Static link flags.** Linux uses static OpenBLAS and METIS. Keep OpenBLAS' dependent `-lm -lpthread -lgfortran` in the LAPACK lflags, and keep `-lm` in both the MUMPS `--with-metis-lflags` and IPOPT `--with-mumps-lflags`; otherwise configure/link checks fail on Linux.
- **AOT solver harness.** `al.qp(...)` (PIQP) and `al.nlp(...)` (IPOPT) are wired through `ctypes` for direct Python calls and through generated C for nested JIT/AOT use; see [`solvers.md`](solvers.md). What is still TODO before distribution: a CI-level standalone C/C++ harness that links `-lpiqpc`/`-lipopt` directly outside Python and exercises the exact AOT path the static-libgfortran work is meant to unblock.

## 6. ThirdParty version pins (as of 2026-05-19)

```
IPOPT_BRANCH    = "releases/3.14.19"
MUMPS_BRANCH    = "releases/3.0.12"
METIS_BRANCH    = "releases/2.0.1"
OPENBLAS_BRANCH = "v0.3.28"
```

These were the latest tags at extraction time. The spec originally listed `releases/2.2.0` for METIS, which does not exist — only the 2.x tags up to 2.0.1 are published.
