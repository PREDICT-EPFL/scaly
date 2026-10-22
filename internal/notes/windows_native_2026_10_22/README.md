# Native Windows: compiler, core, JIT cache and PIQP wheel

Evidence for #152, #153, #154 and #155, gathered on October 22, 2026 for #187 on the
maintainer's Windows machine. The decisions they test are in
[the Windows toolchain investigation](../windows_toolchain_options_2026_10_09.md#decisions). The
SQP closed loops on the same machine are in `../windows_zig_sqp_2026_10_17/`.

## Machine

- Intel Core i5-14600KF (Raptor Lake: AVX2, no AVX-512), Windows 11 Pro build 26200.
- Python 3.14.8 for the suite, and 3.12.11 and 3.14.8 for the wheel checks, all from uv.
- `zig-bin` 0.17.0, whose `zig cc` is Clang 22.1.8 targeting `x86_64-unknown-windows-gnu`.
- No Visual Studio, MSYS2 or MinGW compiler is installed. Git for Windows' `mingw64` directory
  holds only a runtime DLL.

## Compiler (#152)

`probe.py` is self-contained: `uv run internal/notes/windows_native_2026_10_22/probe.py` installs
`zig-bin` and NumPy into a temporary environment. It compiles a DLL shaped like Scaly's generated
code, with the entry signature, GNU vector types passed by value to inline helpers, libm calls and
a module-definition file for its exports, using `-O2 -march=native -fno-math-errno`. The compiler
runs with a PATH of only the Windows system directories. The DLL loads through `ctypes` from an
ordinary Python process. `probe.json` holds the full output.

| Check | Result |
| --- | --- |
| Entry status, lanes chosen by the native macros | 0, 4 |
| Largest elementwise difference from NumPy, difference of the sum | 7.1e-15, 1.5e-11 |
| Imported DLLs | `KERNEL32.dll` and the `api-ms-win-crt-*` API sets of the Universal C Runtime (UCRT) |
| Compilation with an empty zig cache, median of 5 | 10.6 s |
| The same compilation with that cache warm, median of 5 | 0.09 s |

Every import is part of Windows 10 and later. Zig links its C++ runtime statically, so neither a
JIT module nor `piqpc.dll` needs a DLL from the toolchain.

`first_compile.py` times Scaly's first JIT call in a fresh process, which also probes the
compiler's version and macros:

| First call | Mean (s) | Min (s) | Max (s) |
|---|---:|---:|---:|
| zig cc, empty zig cache | 12.696 | 12.338 | 13.138 |
| zig cc, warm zig cache | 0.434 | 0.429 | 0.439 |

The empty-cache cost is zig building its MinGW-w64 C runtime and compiler support for the native
CPU into its global cache, once per machine and zig version. It is about four times the 2 to 3 s
measured on Linux and macOS for #158. Recommendation: `zig-bin==0.17.*` is sufficient on native
Windows x86-64 and is the compiler the core Windows support uses. Document the first-compilation
delay. A warm-up mechanism would only move the same one-time cost elsewhere, since an installer
cannot run it.

## Core (#153)

What makes the core work on Windows:

- Scaly writes a module-definition file listing the exported entry points beside each generated
  source. Zig, like MinGW, exports every non-static function only until one symbol is marked
  `dllexport`, and the CasADi-generated oracles that the SQP plugin embeds for the benchmark
  comparison carry that mark. Zig also ignores `--export-all-symbols`.
- Functions that print flush through `ucrtbase.dll`, the C runtime that zig's Windows target links
  and Python shares. `test_c_and_python_output_stay_in_order_through_a_pipe` fails when the flush
  is removed.
- `SCALY_CC` and `CC` accept a command, such as a quoted path to `zig.exe` followed by `cc`.
- Solver link flags omit the ELF rpath. A JIT module that calls a solver is loaded with the
  solver library directories added through `os.add_dll_directory`.

Tests that compile generated C by hand now use the JIT's compiler through `tests/compilers.py`,
and `test_vectors.py` adds zig to its gcc and clang matrices. The full suite on this machine, with
the toolchain extra and every plugin installed, gives 2466 passed, 146 skipped and 0 failed. The
skips are:

| Count | Reason |
| ---: | --- |
| 86 | the gcc and clang entries of the compiler matrices; their zig entries run |
| 39 | IPOPT, which has no Windows build (#196) |
| 14 | glibc-only: libmvec in `test_vectors.py` and `test_jit.py` |
| 5 | benchmark tooling: binutils `size` and `nm`, and the C++ Google Benchmark driver |
| 1 | malloc interposition through the ELF or Mach-O loader |
| 1 | a compiler wrapper written as a shell script; `test_compiler_fingerprint_changes_with_the_compiler_and_host` covers the same cache-key logic |

`test_jit_compiles_into_a_cache_path_with_spaces` compiles into a cache path containing spaces,
and the wheel check below compiles solver calls whose include and library directories contain
spaces. Compiler commands are argument lists throughout.

## JIT cache (#154)

Windows refuses to replace or delete a DLL that any process has loaded, and `ctypes` never unloads
one. The cache keeps its content-addressed entries and handles that refusal in two places:

- Publication renames a finished file into the cache. When Windows refuses because the target is
  open, another process published the same content first and is compiling or running it, so its
  file stays and the temporary file is deleted.
- Invalidation, through `recompile()`, removes what it can of the entry on every platform. When a
  loaded DLL survives, this process compiles the function into another entry, whose key adds its
  process id and a per-function counter. Old and new handles both stay callable.

| Test | What it shows | Fails when |
| --- | --- | --- |
| `test_processes_share_a_cold_cache` | two processes released together on an empty cache both evaluate correctly; a third reuses the entry without compiling | publication stops tolerating the refusal, or every call compiles |
| `test_publication_leaves_a_loaded_library_callable` | publishing onto a loaded DLL keeps it and its handle | publication stops tolerating the refusal |
| `test_old_handle_stays_callable_after_recompilation` | both handles evaluate correctly after `recompile()` | recompilation reuses the locked path |
| `test_recompile_clears_cache_and_recompiles` | only the loaded DLL survives invalidation on Windows, nothing on Linux and macOS | invalidation leaves the entry in place |

The tests run on every platform. The first one only meets the refusal when the two processes
collide, which they did here: it failed when the tolerance was removed.

Limitations: a recompiled function's previous DLL stays in the cache while any process has it
loaded, and the entry compiled by that process stays after it exits, since no later process derives
its key. The JIT cache has no automatic pruning on any platform, so clearing it by hand after the
processes exit removes both.

## PIQP wheel (#155)

On Windows the PIQP build hook defaults `CC` and `CXX` to `zig cc` and `zig c++`, from a
`zig-bin==0.17.*` build requirement. CMake and Ninja also come from the build requirements, so the
build needs only git beside Python. A toolchain file hands CMake zig's C, C++, assembler, `ar` and
`ranlib`. The PIQP target is renamed to `piqpc.dll`, and the import library `libpiqpc.dll.a` ships
beside it for the JIT's `-lpiqpc`. The build cache key names the compiler without its path, because
an isolated build environment lives in a new temporary directory each time.

BLASFEO keeps its `X64_INTEL_HASWELL` kernels, as on Linux x86-64, so the wheel requires AVX2.
BLASFEO's CMake defines `OS_WINDOWS` for a native Windows build, and its assembly kernels use the
Windows x64 calling convention under that define. PIQP's dense solves, the default mode in the
plugin tests, run on these kernels.

A from-scratch build, with empty solver and zig caches, through an sdist:

```bash
SCALY_BUILD_SOLVERS=required SCALY_SOLVER_CACHE=<empty> ZIG_GLOBAL_CACHE_DIR=<empty> uv build --package scaly-piqp -o <dist>
uv build --package scaly --wheel -o <dist>
uv build --package scaly-sqp --wheel -o <dist>
```

The PIQP build took 1 min 50 s, including the clones of Eigen, BLASFEO and PIQP. It produced
`scaly_piqp-0.1.0a2-py3-none-win_amd64.whl` with `piqpc.dll`, `libpiqpc.dll.a`, the two C
headers, `THIRD_PARTY_NOTICES.md` and the license texts of PIQP, LDL, Eigen and BLASFEO. The DLL
imports only `KERNEL32.dll` and UCRT API sets.

`verify_wheels.py <dist> 3.12 3.14` installs `scaly[toolchain]` and both plugins from those wheels
into a new environment outside the checkout, for each Python version. It then runs, in isolated
mode from an empty directory, with a PATH of only that environment and the Windows system
directories:

- a PIQP solve called from Python, against the analytic solution for two parameters, with status
  and iteration count
- the same QP inside a Scaly function, whose compiled module calls PIQP, with its solver status
- an SQP solve of a nonlinear constrained problem, against its analytic solution for two
  parameters, with status, SQP and QP iteration counts
- that the `piqpc.dll` loaded in the process comes from the environment's `site-packages`

All passed on Python 3.12.11 and 3.14.8.

## Left to #156, #196 and #197

Windows CI jobs that install the built wheels, Windows in the build and release matrices, and
`zig-bin` as a Windows dependency of the core package are #156. IPOPT is #196. Installation
instructions, the first-compilation delay and the limitations above go on the site with #197.
