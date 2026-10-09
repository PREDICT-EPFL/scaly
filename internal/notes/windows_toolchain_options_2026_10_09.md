# Windows toolchain options after Zig 0.17

Settled with the maintainer on 2026-10-09 for [#82] and [#84]. None of the native Windows checks
below have run. The earlier [Windows investigation](windows_support_2026_10_02.md) records
cross-compilation results, not native Windows validation. The compiler policy and release scope
are recorded in the [core roadmap](core_compiler_roadmap.md#windows-toolchain-and-support).

## What the missing wheel means

The [PyPI release of ziglang 0.17.0](https://pypi.org/project/ziglang/0.17.0/), uploaded on October 8,
currently has Linux and macOS wheels only. However,
[ziglang 0.16.0](https://pypi.org/project/ziglang/0.16.0/) still has `win_amd64`, `win_arm64` and
`win32` wheels. The current [packaging script](https://codeberg.org/ziglang/zig-pypi/src/branch/main/make_wheels.py)
still includes all three Windows targets. The [0.17 release discussion](https://codeberg.org/ziglang/zig-pypi/issues/51)
records workflow-permission problems and a failing Linux portability check, not a decision to drop
Windows. A publishing problem is the likely explanation, but the specific Windows cause remains
unverified.

Two local `uv pip install --dry-run` checks targeted Windows x86-64 and Python 3.12. Both an
explicit `ziglang==0.16.0` request and an unpinned `ziglang` request resolved to 0.16.0. This proves
package resolution only, not that the compiler runs on Windows. A third dry run targeting Windows
ARM64 also resolved the pinned version.

Zig itself also publishes [portable Windows 0.17.0 ZIPs](https://ziglang.org/download/), and its
[installation guide](https://ziglang.org/learn/getting-started/) documents WinGet. Even if future
Python wheels remain unavailable, a separate Zig installation remains a candidate. A missing
Python wheel does not by itself require Microsoft Visual C++ (MSVC) or a separate MinGW toolchain.

## Decisions

Keep #82 and use `ziglang==0.16.0` as the initial shared compiler version across supported
platforms. #82 pins the toolchain extra. #156 adds the same version as an automatic dependency
of the core package on Windows through its platform-conditional package metadata.
Upgrade after package availability and the compiler and solver checks pass. An unpinned Windows
install can currently select 0.16.0 itself, but a deliberate pin also fixes the compiler used for
validation and avoids depending on installer behavior across platforms.

Keep #84 as the shared parent for #82 and the Windows work. Its initial Windows scope is the
core, `scaly-piqp` and `scaly-sqp` on native Windows x86-64. ARM64 and 32-bit Windows wait for
their own native checks. IPOPT remains later because its Fortran dependencies need a separate
build investigation.

#82 validates the compiler switch with the core and installed solvers on Linux and macOS. Its
kernel comparison on the reference machine remains required. Native Windows solver acceptance
belongs to #84's Windows issues, so it cannot block the compiler switch needed by those issues.
This replaces #82's requirement to validate solver linking on all three operating systems before
changing the default. The scope was separated on 2026-10-09 to remove that circular prerequisite.

If the wheel route stops being usable, try the official portable Zig distribution before changing
compiler families. Compiler commands are already tuples in `codegen/toolchain.py`, so selecting
`(path_to_zig, "cc")` fits the current representation. Discovery and configuration still need
implementation. In particular, today's `SCALY_CC` accepts an executable, not `"zig cc"`.

If Zig itself fails the required Windows builds, the next candidate is the portable Universal C
Runtime (UCRT) distribution of [LLVM-MinGW](https://github.com/mstorsjo/llvm-mingw). It ships Clang,
Windows headers and libraries together and installs by unpacking an archive. This preserves the
compiler flag syntax and GNU vector extensions Scaly already uses. It still requires testing
solver linking, runtime dependencies and distribution.

MSVC is possible, but is a larger change for the present compiler. Scaly probes with `-dM -E`,
uses GNU compiler flags, and emits GNU vector attributes in its native recipe. Supporting
`cl.exe` therefore needs work on both compiler invocation and the generated-code policy. Microsoft's
[command-line build guide](https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170)
also requires an initialized build environment. Prefer one tested Windows toolchain initially.

## What the Windows machine should settle

Start with a small standalone C dynamic-link library (DLL) compiled with the 0.16 wheel and
loaded through `ctypes`. Exercise Scaly's pointer entry signature, explicit exports and math calls
from an ordinary Python process without a Visual Studio developer environment. Measure a fresh
Zig cache separately from later compiles. Cross-compilation cannot establish this result.
Document the first-compilation delay unless these measurements justify a warm-up mechanism.

Then test the core end to end, including values and derivatives, mapped calls and the native
vector recipe. Exercise paths containing spaces, concurrent first calls, a second process using
the cache, and forced recompilation while the old function remains callable. The current cache
uses content-derived directories, but publishing with `Path.replace` and invalidating with
`shutil.rmtree` still need tests against Windows' loaded-DLL restrictions.

The core checks also cover `sc.print` once #77 lands. Its open pull request #149 proposes a flush
of C standard output through `ctypes.CDLL(None).fflush(None)` on Linux and macOS, as recorded
on #84. Windows needs the C runtime used by the generated DLL, rather than a symbol lookup on
the process handle.

Finally build PIQP and BLASFEO with an explicitly selected compiler and CMake generator. Run a
PIQP solve, a generated call into PIQP, then a `scaly-sqp` solve. Inspect the wheel's import
libraries and DLL dependencies, and install it in an environment without the build tools. Users
should receive built solver wheels, rather than compiling C++ merely to install a solver. Capture
the successful checks in Windows continuous integration before claiming support.

Try the existing BLASFEO kernels first. If their Windows build or calling convention fails, use
its generic implementation and record that limitation. Kernel optimization can follow separately.

Use Python for the checks and `subprocess` argument lists for compiler and build commands. Keep
PowerShell to environment setup and launching Python. Test native Windows Python, not Python
inside Windows Subsystem for Linux, which would exercise the existing Linux path.

## Issue ownership

[#84] is the shared delivery parent for [#82] and [#152] through [#156]. [#82] owns the compiler
switch, the exact pin and the Linux/macOS checks, including its reference-machine comparison.
[#152] owns native compiler evidence and the first-compilation recommendation. It can start from
main and writes a separate dated evidence note instead of editing these unmerged planning files.
[#153] owns the core compilation, loading and printing behavior; [#154] owns loaded-DLL cache
behavior. [#155] owns PIQP packaging, DLL dependencies and the BLASFEO implementation choice.
[#156] owns the complete installed SQP path, the automatic Windows compiler dependency, supported
Python checks, Windows workflows and public installation instructions. #153 through #155 record
manual runs on the native Windows machine in their own internal notes; #156 puts those checks in CI. GitHub
records the prerequisites and status.

#83 retains its #82 prerequisite and Clang flag convention. #85 verifies the completed Windows
matrices and release artifacts and writes the release notes; workflow implementation belongs to
[#156]. Parent closure requires the installed package combination to pass together, rather than
only the component issues to close. Issue links use the published planning branch until these
notes land on main. Repoint them to main before deleting that branch.

## Repository context

The recommendation follows the existing Clang flag decision in
`core_compiler_roadmap.md`, especially #82 before `sc.CLibrary` in #83. Relevant implementation
is `src/scaly/codegen/{toolchain,jit,c,abi}.py`, `src/scaly/solvers/paths.py`,
`src/scaly/utils/env.py`, `plugins/scaly-piqp/hatch_build.py`, the SQP plugin metadata, and
`.github/workflows/wheels.yml`. DLL extensions are already recognized. Solver link flags still
emit Unix rpaths, the PIQP build hook explicitly rejects Windows, and the wheel workflow has no
Windows runner. Changing compiler discovery alone therefore cannot complete #84.

[#82]: https://github.com/PREDICT-EPFL/scaly/issues/82
[#84]: https://github.com/PREDICT-EPFL/scaly/issues/84
[#152]: https://github.com/PREDICT-EPFL/scaly/issues/152
[#153]: https://github.com/PREDICT-EPFL/scaly/issues/153
[#154]: https://github.com/PREDICT-EPFL/scaly/issues/154
[#155]: https://github.com/PREDICT-EPFL/scaly/issues/155
[#156]: https://github.com/PREDICT-EPFL/scaly/issues/156
