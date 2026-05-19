# Vendored solver build — known issues

The `hatch_build.py` hook ships PIQP and IPOPT as shared libraries inside the wheel (`src/alloy/lib/`) plus C headers (`src/alloy/include/`). These notes track open issues we should fix before the wheels.yml workflow is exercised at scale.

## 1. Static libgfortran linking — partially achieved

**Goal:** `libipopt.{dylib,so}` has no runtime dependency on libgfortran/libgcc/libstdc++, so external C++ consumers can link `-lipopt` without dragging a Fortran toolchain into their build.

**Current state on macOS (arm64, Apple Clang 21):** *not* met. `otool -L src/alloy/lib/libipopt.dylib` shows dynamic dependencies on `/opt/homebrew/opt/gcc/lib/gcc/current/libgfortran.5.dylib` and `libquadmath.0.dylib`.

**Why:** IPOPT's final libipopt link is driven by `clang++`, not `gfortran`. Apple Clang errors on `-static-libgfortran` (a gfortran driver flag), so we currently pass it through `FCFLAGS=` only. That covers Fortran-only objects but not the C++ link step that produces the dylib.

An earlier attempt put the static flags into `LDFLAGS=`. That broke the autoconf `configure` C-compiler conftest (Apple Clang refuses to even probe with the unknown flag), so the build never started. Hence the `FCFLAGS=` compromise in `_static_fortran_ldflags`.

**Likely fix:** locate `libgfortran.a` / `libquadmath.a` from the active gfortran installation and pass them as explicit static link inputs via `-Wl,-force_load <path>` (macOS) or `-Wl,-Bstatic -lgfortran -Wl,-Bdynamic` (Linux). The path resolution can use `gfortran -print-file-name=libgfortran.a` at hook time.

**Linux:** untested. `_static_fortran_ldflags` returns `-static-libgfortran -static-libgcc -static-libstdc++` for Linux but the same `FCFLAGS`-only routing applies, so the same C++-link gap probably exists there too. First Linux CI run will tell us.

## 2. METIS legacy-C compatibility

`ThirdParty-Metis` is pinned to `releases/2.0.1`. Despite the 2.x version number it still bundles upstream METIS 4.0.3 — a GPL-licensed K&R-era release whose `__GKfree` / implicit declarations modern Clang rejects by default. `_build_metis` patches around this by passing:

```
CFLAGS=-O2 -fPIC -Wno-implicit-function-declaration -Wno-implicit-int -Wno-int-conversion -Wno-error
```

If a future MUMPS pin requires METIS 5.x (Apache-2 licensed, modern C), we'll need to switch to a different ThirdParty branch or build METIS 5 directly. The current pin works on macOS and should work on the manylinux_2_28 image.

## 3. `install_name` rewriting on macOS

IPOPT's install dir contains `libipopt.3.dylib` (real file) and `libipopt.dylib` (symlink). `shutil.copy2` of the symlink resolves through it but preserves the original install_name (`/abs/path/to/.../libipopt.3.dylib`), which makes the lib non-relocatable.

Current fix: pick the versioned dylib directly, copy as `libipopt.dylib`, then run `install_name_tool -id @rpath/libipopt.dylib`. Same `-id` rewrite is applied to PIQP for consistency. Relevant code: `_build_ipopt_stack` near the end.

The Linux equivalent (SONAME via `patchelf --set-soname`) is not implemented yet — needed before the manylinux wheel works.

## 4. Things not yet exercised

- **CI cache key.** Keyed on `hashFiles('hatch_build.py')`. If we later split the hook into multiple files, update the key. Cold IPOPT build is ~5-8 min, so a stale cache hides a lot.
- **Solver bindings.** The hatch hook ships the libraries; Phase 5's actual `al.qp(...)` / `al.nlp(...)` bindings on top of them are still TODO.

## 5. ThirdParty version pins (as of 2026-05-19)

```
IPOPT_BRANCH    = "releases/3.14.19"
MUMPS_BRANCH    = "releases/3.0.12"
METIS_BRANCH    = "releases/2.0.1"
OPENBLAS_BRANCH = "v0.3.28"
```

These were the latest tags at extraction time. The spec originally listed `releases/2.2.0` for METIS, which does not exist — only the 2.x tags up to 2.0.1 are published.
