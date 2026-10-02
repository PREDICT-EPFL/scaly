# Windows support: the investigation so far

Background for [#84], written 2026-10-02. Everything here was read from CasADi's sources and wheel,
or tried by cross-compiling from macOS with `ziglang` 0.16.0. Nothing has run on a Windows machine
yet, and the first step of [#84] is to do so. This note records what to try, not how to build it.

## How CasADi does it

- Its Windows wheels are cross-compiled on Linux with MinGW-w64 GCC (the `core-dockcross` job,
  target `windows-shared-x64-posix`, in
  [`binaries.yml`](https://github.com/casadi/casadi/blob/main/.github/workflows/binaries.yml)).
  IPOPT, MUMPS, METIS, OpenBLAS and PIQP use the same toolchain, and the wheel ships
  `libgfortran`, `libquadmath`, `libgcc_s_seh`, `libstdc++`, `libwinpthread` and `libatomic` as
  DLLs. These load into the standard MSVC-built CPython without trouble.
- Its JIT bundles no compiler. On Windows the `shell` compiler runs `cl.exe` and `link.exe` from
  `PATH` ([`shell_compiler.cpp`](https://github.com/casadi/casadi/blob/main/casadi/solvers/shell_compiler.cpp)),
  so Python must be started from a Visual Studio developer prompt. "Compilation failed. Tried
  cl.exe" is its most common Windows issue
  ([#2746](https://github.com/casadi/casadi/issues/2746),
  [#3964](https://github.com/casadi/casadi/issues/3964)). Each build uses random temporary names
  and is deleted afterwards, so it never overwrites a loaded DLL and has no on-disk cache.
- Generated C marks exports `__declspec(dllexport)` on Windows and uses `long long` for integers.

Scaly should not copy the JIT side: a compiler on `PATH` is exactly what CasADi's users trip over.

## zig as the Windows toolchain

- `ziglang` has `win_amd64` and `win_arm64` wheels on [PyPI](https://pypi.org/project/ziglang/),
  about 100 MB each.
- Its default Windows target is windows-gnu: the output depends only on the Universal C Runtime,
  which CPython also uses, and `KERNEL32`; the MinGW parts and libc++ are linked statically, so a
  zig-built PIQP would need no extra runtime DLLs. `-target ...-msvc` needs MSVC installed.
- `sincos` and `-march=native` work. `-shared` also writes a `.lib` and a 1 to 2 MB `.pdb` beside
  the DLL.
- The first compile took about 6 s while zig built its C runtime once; later ones about 0.1 s.
- zig has no Fortran, so `scaly-ipopt` cannot share the toolchain; it would follow CasADi's MinGW
  gfortran build. That is why it is not part of the first Windows target.

## What to try on a Windows machine

1. A zig-built DLL loaded through ctypes and called through Scaly's entry signature, and what the
   linker exports without `__declspec(dllexport)`, which `codegen/` never emits today.
2. The JIT cache: a loaded DLL cannot be overwritten, so files must be content-named, never
   rewritten, and published through `os.replace` with an existing target counted as success; the
   260-character path limit; compiler commands passed to `subprocess` as argument lists.
3. The plugins' vendored DLLs found through `os.add_dll_directory` or full paths, since Python 3.8
   no longer searches `PATH` for dependencies.
4. PIQP and BLASFEO built with zig as the CMake C and C++ compiler; BLASFEO's assembly kernels
   under zig on Windows are unverified.
5. The first-compile cost in a fresh environment, and whether it needs a warm-up or only a
   documented note.

[#84]: https://github.com/PREDICT-EPFL/scaly/issues/84
