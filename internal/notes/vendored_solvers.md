# Vendored solver build — known issues

The `plugins/scaly-{piqp,ipopt}/hatch_build.py` hooks ship each solver's shared library and C headers inside its plugin wheel. These notes track open issues we should fix before the wheels.yml workflow is exercised at scale. Historical notes from the conda-prefix/delocate experiment live in [`native_toolchain_exploration.md`](native_toolchain_exploration.md).

## 1. Runtime bundling replaced static linking — done, and enforced

**Original goal:** `libipopt.{dylib,so}` with no runtime dependency on libgfortran/libgcc/libstdc++, so an external C++ consumer could link `-lipopt` without dragging a Fortran toolchain into its build.

**Why it was abandoned:** IPOPT's final `libipopt` link is driven by `clang++`/`g++`, not `gfortran`, and it reads `LDFLAGS`, not `FCFLAGS`. Apple Clang errors on `-static-libgfortran`, so an early version routed the static flags through `FCFLAGS=`; that covered the Fortran objects and never touched the C++ link that produces the shared library, so it achieved nothing. It also did active harm: COIN-OR's configure sets its default with `: ${FCFLAGS:="-O2 $ADD_FCFLAGS"}`, so passing `FCFLAGS=` on the command line silently dropped `-O2` from the MUMPS and IPOPT Fortran. (`ADD_FCFLAGS` is the variable for adding to those defaults.) Putting the flags in `LDFLAGS=` instead broke autoconf's C-compiler conftest — Apple Clang refuses to even probe with the unknown flag — so the build never started.

**What replaced it:** the hook vendors the Fortran runtime next to `libipopt` and makes the library find it relocatably. `_static_fortran_ldflags` is gone.

- **macOS:** `_bundle_macos_runtime` walks the `otool -L` closure, copies every non-OS dependency into `lib/`, rewrites each load command to `@rpath/`, adds an `@loader_path` rpath, and re-signs — arm64 refuses to load an image whose load commands changed after signing.
- **Linux:** `_bundle_linux_runtime` does the same job with `$ORIGIN`. IPOPT links with `LDFLAGS=-Wl,-rpath,\$$ORIGIN` and the hook copies `libgfortran` (plus `libquadmath` when the closure needs it) next to `libipopt.so`, so `ldd` resolves them to the siblings. `libgcc_s` and `libstdc++` deliberately stay on the system: they are in every glibc distribution's base install, and bundling `libstdc++` risks pinning an old one onto other C++ libraries in the same process.

The `$ORIGIN` escaping in `LINUX_RPATH_LDFLAGS` is load-bearing and explained in a comment beside it. Both layers eat a `$`: configure copies `LDFLAGS` into the Makefiles verbatim where make turns `$$` back into `$`, and the recipe shell needs the backslash or it expands `$ORIGIN` to nothing — leaving a bare `-Wl,-rpath` that swallows the next argument and fails the link with `cannot find libipopt.so.3`.

The packaging consequence: **the whole `lib/` directory is what has to travel**, never just `libipopt.dylib`.

This is checked rather than assumed. `_macos_self_contained` / `_linux_self_contained` assert that every vendored dependency in the closure resolves to a sibling under `lib/`. `_ipopt_built` treats a library that fails the check as unbuilt and rebuilds it, so a stale library from before the bundling pass cannot survive a sync; the Linux path additionally raises if the bundle comes out incomplete.

**What the original goal would still have bought:** a consumer linking `-lipopt` from C++ outside Python needs `lib/` on its rpath, where a fully static library would have needed nothing. That is a weaker guarantee, but it no longer blocks distribution.

## 2. METIS 5 and GKlib

METIS comes from `KarypisLab/METIS` at tag `v5.2.1` (Apache-2.0), not from COIN-OR's `ThirdParty-Metis`, whose `get.Metis` fetches METIS 4.0.3 under a license that forbids redistribution. METIS 5 no longer bundles GKlib, so `_build_metis` builds `KarypisLab/GKlib` first, pinned to a commit because that repository has no release tags, and installs both into `metis_install`. Consequences for the rest of the stack:

- MUMPS and IPOPT link `-lmetis -lGKlib -lm`; `ThirdParty-Mumps` reads `METIS_VER_MAJOR` from `metis.h` and switches its Fortran to the METIS 5 `METIS_NodeND` entry point on its own.
- MUMPS insists on `idx_t` being a plain `int`, so the hook writes `IDXTYPEWIDTH 32` into the generated `build/xinclude/metis.h`, exactly what upstream's `make config` would do.
- METIS' `conf/gkbuild.cmake` hardcodes `-march=native` under GCC. The hook strips it after cloning; otherwise the Linux wheel would be tied to the build host while OpenBLAS goes to the trouble of `DYNAMIC_ARCH=1`.
- GKlib's `LICENSES.md` lists two glibc-derived headers under LGPL-2.1-or-later and one BSD-3-Clause file next to the Apache-2.0 default. `_write_third_party_notices` copies all four texts into `licenses/gklib/`.

The METIS 4 legacy-C warning flags are gone with it; METIS 5 is modern C.

## 3. `install_name` rewriting on macOS

IPOPT's install dir contains `libipopt.3.dylib` (real file) and `libipopt.dylib` (symlink). `shutil.copy2` of the symlink resolves through it but preserves the original install_name (`/abs/path/to/.../libipopt.3.dylib`), which makes the lib non-relocatable.

Current fix: pick the versioned dylib directly, copy as `libipopt.dylib`, then run `install_name_tool -id @rpath/libipopt.dylib`. Same `-id` rewrite is applied to PIQP for consistency. Relevant code: `_build_ipopt_stack` near the end, which then hands off to `_bundle_macos_runtime` for the rest of the closure (issue #1).

Linux keeps a SONAME-compatible copy next to the unversioned link target (for example `libipopt.so.3` next to `libipopt.so`) so JIT-built solver callers with `DT_NEEDED=libipopt.so.3` can resolve through their rpath. A future wheel repair pass may still prefer setting/changing SONAMEs explicitly with `patchelf`.

## 4. Wheel platform tag and editable build mode

**Goal:** each built wheel correctly declares its platform (and is manylinux- / delocate-compatible) so installers resolve it correctly and PyPI accepts upload, while local editable installs do not fail before Python-only development can start.

**Current state:** both plugin `hatch_build.py` hooks mark wheel builds that include vendored solver libraries as impure:

```python
build_data["pure_python"] = False
build_data["infer_tag"] = True
```

ABI tag stays `none` because the vendored libs are loaded via `ctypes`, not linked as a CPython extension module — there's no Python ABI to bind to. Output becomes `scaly-0.1.0-py3-none-macosx_14_0_arm64.whl` / `scaly-0.1.0-py3-none-linux_x86_64.whl` per build host.

Editable installs use `SCALY_BUILD_SOLVERS=auto` by default: if the native toolchain is present, `uv sync` builds the solver stack; if it is missing (for example no `gfortran`), the hook skips the missing solver libraries and solver tests are skipped. CI sets `SCALY_BUILD_SOLVERS=required` so missing toolchains/build regressions remain fatal. `SCALY_BUILD_SOLVERS=skip` is available for intentionally Python-only syncs.

**Still required before distribution:** a correct tag is necessary but not sufficient; PyPI rejects raw `linux_*` and the wheel may still pull in host-specific shared libs.

- **Linux:** cibuildwheel's `auditwheel repair` produces `manylinux_2_24` (PIQP) and `manylinux_2_27` (IPOPT) wheels and leaves the `$ORIGIN` siblings under `lib/` alone. It still grafts a second `libquadmath` into `scaly_ipopt.libs/`, and `libipopt.so` ships twice (also as `libipopt.so.3`): about 28 MB of duplicates in the IPOPT wheel, open.
- **macOS:** `delocate-wheel` finds nothing left to move after `_bundle_macos_runtime`, and the installed wheels load and solve on arm64 and x86_64. The IPOPT wheel requires macOS 15, the minimum of the bundled Homebrew runtime.
- **Matrix:** `ci.yml` builds one wheel per OS and architecture on its native runner with cibuildwheel.

**Renaming is not a substitute.** The wheel's `*.dist-info/WHEEL` file records a `Tag:` line that installers cross-check against the filename. A wheel renamed from `py3-none-any.whl` to `py3-none-macosx_14_0_arm64.whl` still claims `any` internally and fails strict validation. Independently, PyPI refuses uploads with raw `linux_*` tags — only `manylinux_*` / `musllinux_*` are accepted, and those tags are contracts about glibc baseline and bundled deps, not free-form labels.

## 5. Things not yet exercised

- **CI cache key.** Keyed on OS, architecture, and both plugin `hatch_build.py` files. Cold IPOPT build is ~5-8 min, so a stale cache hides a lot.
- **Static OpenBLAS install.** `_build_openblas` builds with `NO_SHARED=1 USE_OPENMP=0 DYNAMIC_ARCH=1`; pass the same flags to `make install` or OpenBLAS tries to install a shared `libopenblas*.so` that was never built.
- **Static link flags.** Linux uses static OpenBLAS, METIS and GKlib. Keep OpenBLAS' dependent `-lm -lpthread -lgfortran` in the LAPACK lflags, and keep `-lm` in both the MUMPS `--with-metis-lflags` and IPOPT `--with-mumps-lflags`; otherwise configure/link checks fail on Linux.
- **AOT solver harness.** `sc.qp(...)` (PIQP) and `sc.nlp(...)` (IPOPT) are wired through `ctypes` for direct Python calls and through generated C for nested JIT/AOT use; see [`solvers.md`](solvers.md). What is still TODO before distribution: a CI-level standalone C/C++ harness that links `-lpiqpc`/`-lipopt` directly outside Python and exercises the exact AOT path the static-libgfortran work is meant to unblock.

## 6. ThirdParty version pins (as of 2026-05-19)

```
IPOPT_BRANCH    = "releases/3.14.19"
MUMPS_BRANCH    = "releases/3.0.12"
METIS_TAG       = "v5.2.1"
GKLIB_COMMIT    = "3b7d61b9f885063c89901f3901fb4426f9cfb58f"
OPENBLAS_BRANCH = "v0.3.28"
```

IPOPT, MUMPS and OpenBLAS were the latest tags at extraction time. METIS moved from `ThirdParty-Metis releases/2.0.1` (METIS 4.0.3 inside) to upstream 5.2.1 in September 2026, see section 2.
